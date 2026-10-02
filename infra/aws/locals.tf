data "aws_caller_identity" "current" {}

locals {
  suffix      = var.bucket_suffix != "" ? "-${var.bucket_suffix}" : ""
  data_bucket = "${var.prefix}-data${local.suffix}"
  account_id  = data.aws_caller_identity.current.account_id

  # IoT topic rule names allow only alphanumerics and underscores.
  rule_name    = "${replace(var.prefix, "-", "_")}_sensors"
  sensor_topic = "${var.prefix}/sensors"
}
