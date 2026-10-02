# High-frequency sensor events. On-demand capacity so the demo needs no shard
# sizing; Databricks reads this stream directly via format("kinesis").
resource "aws_kinesis_stream" "sensors" {
  name             = "${var.prefix}-sensors"
  retention_period = 24

  stream_mode_details {
    stream_mode = "ON_DEMAND"
  }
}
