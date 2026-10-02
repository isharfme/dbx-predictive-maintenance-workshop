output "external_location_name" {
  description = "External location the notebook reads settings files from."
  value       = databricks_external_location.data.name
}

output "s3_role_arn" {
  value = aws_iam_role.uc_s3.arn
}

output "kinesis_service_credential_name" {
  description = "Service credential name for the notebook's serviceCredential option."
  value       = databricks_credential.kinesis.name
}

output "kinesis_role_arn" {
  value = aws_iam_role.uc_kinesis.arn
}

output "catalog" {
  description = "Catalog the notebook reads and writes workshop data under."
  value       = var.catalog
}

output "data_schema" {
  description = "Schema name the notebook creates for the workshop data."
  value       = var.data_schema
}

output "config_volume_path" {
  description = "UC volume path the deploy JSON is written to."
  value       = "/Volumes/${var.catalog}/${var.config_schema}/${var.config_volume}"
}
