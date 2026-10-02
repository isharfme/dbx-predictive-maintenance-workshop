output "data_bucket" {
  description = "Published static-metadata bucket."
  value       = aws_s3_bucket.data.bucket
}

output "kinesis_stream_name" {
  value = aws_kinesis_stream.sensors.name
}

output "kinesis_stream_arn" {
  value = aws_kinesis_stream.sensors.arn
}

output "iot_thing_name" {
  value = aws_iot_thing.producer.name
}

output "iot_endpoint" {
  description = "MQTT data endpoint host for the producer."
  value       = data.aws_iot_endpoint.data_ats.endpoint_address
}

output "sensor_topic_prefix" {
  description = "Publish to <prefix>/<engine_family>."
  value       = local.sensor_topic
}

output "certificate_path" {
  value = local_sensitive_file.certificate.filename
}

output "private_key_path" {
  value = local_sensitive_file.private_key.filename
}
