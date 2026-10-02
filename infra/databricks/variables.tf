variable "aws_profile" {
  description = "Local AWS CLI/SSO profile to authenticate with."
  type        = string
}

variable "aws_region" {
  description = "AWS region of the S3 bucket and Kinesis stream."
  type        = string
}

variable "prefix" {
  description = "Name prefix shared with the infra/aws assets."
  type        = string
  default     = "pdm-ws"
}

variable "bucket_suffix" {
  description = "Optional suffix if infra/aws set one on the data bucket name."
  type        = string
  default     = ""
}

variable "owner_email" {
  description = "Owner tag: the resource owner's email."
  type        = string
}

variable "remove_after" {
  description = "RemoveAfter tag: expiry date in YYYY-MM-DD."
  type        = string
}

variable "project" {
  description = "Project tag value."
  type        = string
  default     = "pdm-workshop"
}

variable "databricks_profile" {
  description = "~/.databrickscfg profile for the target workspace."
  type        = string
}

variable "databricks_account_id" {
  description = "Databricks account ID, used as the external ID in the IAM role trust policy."
  type        = string
}

variable "workshop_principal" {
  description = "UC group (or user) that gets read access to the flows, e.g. a workshop group."
  type        = string
}

variable "catalog" {
  description = "Existing Unity Catalog the workshop uses; not created here."
  type        = string
  default     = "pdm_ws_catalog"
}

variable "data_schema" {
  description = "Schema holding the workshop medallion tables and the ops volume."
  type        = string
  default     = "ai"
}

variable "ops_volume" {
  description = "Volume the notebook uses for streaming checkpoints and Auto Loader schema tracking."
  type        = string
  default     = "ops"
}

variable "config_schema" {
  description = "Schema holding the config volume with the deploy JSON."
  type        = string
  default     = "config"
}

variable "config_volume" {
  description = "Volume the deploy JSON is written to and the notebook reads."
  type        = string
  default     = "outputs"
}
