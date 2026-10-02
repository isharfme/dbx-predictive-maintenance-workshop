# Grants Databricks read access to the two published AWS flows through Unity
# Catalog, so notebooks reference only UC object names — never AWS keys. UC vends
# short-lived STS credentials at read time.
#
# - S3 (settings files) is object storage → storage credential + external location.
# - Kinesis (sensor stream) is a streaming service, not storage → service
#   credential (usable from serverless, where instance profiles are not).
#
# Both IAM roles use the Unity Catalog self-assuming trust pattern, with the
# Databricks account ID as the external ID.

data "aws_caller_identity" "current" {}

locals {
  suffix             = var.bucket_suffix != "" ? "-${var.bucket_suffix}" : ""
  data_bucket        = "${var.prefix}-data${local.suffix}"
  account_id         = data.aws_caller_identity.current.account_id
  kinesis_stream_arn = "arn:aws:kinesis:${var.aws_region}:${local.account_id}:stream/${var.prefix}-sensors"
  s3_role_name       = "${var.prefix}-uc-s3"
  kinesis_role_name  = "${var.prefix}-uc-kinesis"
}

# ---------------------------------------------------------------------------
# S3: storage credential + external location
# ---------------------------------------------------------------------------

data "databricks_aws_unity_catalog_assume_role_policy" "s3" {
  aws_account_id = local.account_id
  role_name      = local.s3_role_name
  external_id    = var.databricks_account_id
}

data "databricks_aws_unity_catalog_policy" "s3" {
  aws_account_id = local.account_id
  bucket_name    = local.data_bucket
  role_name      = local.s3_role_name
}

resource "aws_iam_role" "uc_s3" {
  name               = local.s3_role_name
  assume_role_policy = data.databricks_aws_unity_catalog_assume_role_policy.s3.json
}

resource "aws_iam_role_policy" "uc_s3" {
  name   = local.s3_role_name
  role   = aws_iam_role.uc_s3.id
  policy = data.databricks_aws_unity_catalog_policy.s3.json
}

# AWS needs time to propagate a freshly created IAM role — enough that the role
# can even assume itself (the self-assuming Unity Catalog pattern). Until then,
# UC's read-validation of the external location gets a 403, so wait before
# creating the credential. Only delays a fresh create, not steady-state applies.
resource "time_sleep" "uc_s3_propagation" {
  depends_on      = [aws_iam_role.uc_s3, aws_iam_role_policy.uc_s3]
  create_duration = "60s"
}

resource "databricks_storage_credential" "s3" {
  name = local.s3_role_name
  aws_iam_role {
    role_arn = aws_iam_role.uc_s3.arn
  }
  comment = "Read the published engine-settings bucket."

  depends_on = [time_sleep.uc_s3_propagation]
}

resource "databricks_external_location" "data" {
  name            = "${var.prefix}-data"
  url             = "s3://${local.data_bucket}/"
  credential_name = databricks_storage_credential.s3.id
  comment         = "Workshop published static data."
}

resource "databricks_grants" "data" {
  external_location = databricks_external_location.data.id
  grant {
    principal  = var.workshop_principal
    privileges = ["READ_FILES"]
  }
}

# ---------------------------------------------------------------------------
# Kinesis: service credential
# ---------------------------------------------------------------------------

data "databricks_aws_unity_catalog_assume_role_policy" "kinesis" {
  aws_account_id = local.account_id
  role_name      = local.kinesis_role_name
  external_id    = var.databricks_account_id
}

resource "aws_iam_role" "uc_kinesis" {
  name               = local.kinesis_role_name
  assume_role_policy = data.databricks_aws_unity_catalog_assume_role_policy.kinesis.json
}

resource "aws_iam_role_policy" "uc_kinesis" {
  name = local.kinesis_role_name
  role = aws_iam_role.uc_kinesis.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "kinesis:GetRecords",
          "kinesis:GetShardIterator",
          "kinesis:DescribeStream",
          "kinesis:DescribeStreamSummary",
          "kinesis:ListShards",
          "kinesis:SubscribeToShard",
        ]
        Resource = local.kinesis_stream_arn
      },
      # Unity Catalog vends read credentials by having this role assume itself, so
      # the role needs permission to do that. The S3 role gets this statement from
      # the UC-generated policy; this hand-written policy must add it explicitly.
      {
        Effect   = "Allow"
        Action   = ["sts:AssumeRole"]
        Resource = aws_iam_role.uc_kinesis.arn
      },
    ]
  })
}

resource "databricks_credential" "kinesis" {
  name = local.kinesis_role_name
  aws_iam_role {
    role_arn = aws_iam_role.uc_kinesis.arn
  }
  purpose = "SERVICE"
  comment = "Read the sensor Kinesis stream."
}

resource "databricks_grants" "kinesis" {
  credential = databricks_credential.kinesis.id
  grant {
    principal  = var.workshop_principal
    privileges = ["ACCESS"]
  }
}

# ---------------------------------------------------------------------------
# Config volume: Terraform's outputs, published for the notebook to read
# ---------------------------------------------------------------------------
#
# Terraform knows the environment-specific names the notebook needs (catalog,
# credential and location names, the AWS stream and bucket). deploy.sh writes
# them as a JSON file into this volume so the notebook reads UC object names at
# runtime — never Terraform state or hardcoded values. A volume (not a Delta
# table) holds the file because writing a file needs no compute, keeping
# deploy.sh to the AWS, terraform, and databricks CLIs with no SQL warehouse.
#
# The catalog already exists and is never a Terraform resource here, so neither
# apply nor destroy can create or delete it. The schema is created with
# force_destroy so `terraform destroy` removes it (and the managed volume and its
# file) without a manual cleanup step.

resource "databricks_schema" "config" {
  catalog_name  = var.catalog
  name          = var.config_schema
  comment       = "Deploy config the notebook reads at runtime."
  force_destroy = true
}

resource "databricks_volume" "config" {
  catalog_name = var.catalog
  schema_name  = databricks_schema.config.name
  name         = var.config_volume
  volume_type  = "MANAGED"
  comment      = "Holds deploy.json, written by deploy.sh."
}

# Reading a volume file needs USE_SCHEMA on its schema and READ_VOLUME on the
# volume, so grant both to the principal on these Terraform-created objects.
# Catalog-level access (USE_CATALOG) is deliberately not managed here: the
# principal already holds it through the catalog's default grant, and an
# authoritative databricks_grants on the shared catalog would overwrite that
# whole ACL. Leaving the catalog grants alone also means destroy never revokes
# catalog access mid-teardown.
resource "databricks_grants" "config_schema" {
  schema = databricks_schema.config.id
  grant {
    principal  = var.workshop_principal
    privileges = ["USE_SCHEMA"]
  }
}

resource "databricks_grants" "config_volume" {
  volume = databricks_volume.config.id
  grant {
    principal  = var.workshop_principal
    privileges = ["READ_VOLUME"]
  }
}

# ---------------------------------------------------------------------------
# Data schema + ops volume: the medallion pipeline's home
# ---------------------------------------------------------------------------
#
# The notebook builds its bronze/silver/gold tables in this schema and keeps
# streaming checkpoints in the ops volume. Both are created here, with the same
# force_destroy on the schema as the config schema above, so `terraform destroy`
# reclaims the schema and cascades every table and volume the notebook wrote
# into it — no orphaned UC objects, and no SQL warehouse needed to clean up.

resource "databricks_schema" "data" {
  catalog_name  = var.catalog
  name          = var.data_schema
  comment       = "Workshop medallion tables (bronze/silver/gold)."
  force_destroy = true
}

resource "databricks_volume" "ops" {
  catalog_name = var.catalog
  schema_name  = databricks_schema.data.name
  name         = var.ops_volume
  volume_type  = "MANAGED"
  comment      = "Streaming checkpoints and Auto Loader schema tracking."
}
