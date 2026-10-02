variable "aws_profile" {
  description = "Local AWS CLI/SSO profile to authenticate with."
  type        = string
}

variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
}

variable "prefix" {
  description = "Name prefix for all assets."
  type        = string
  default     = "pdm-ws"
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

variable "bucket_suffix" {
  description = "Optional suffix if the global S3 bucket names are already taken."
  type        = string
  default     = ""
}
