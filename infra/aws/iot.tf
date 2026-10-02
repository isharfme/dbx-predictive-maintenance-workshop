# The MQTT data endpoint the producer connects to.
data "aws_iot_endpoint" "data_ats" {
  endpoint_type = "iot:Data-ATS"
}

# Simulated producer device, with an AWS-generated X.509 keypair.
resource "aws_iot_thing" "producer" {
  name = "${var.prefix}-producer"
}

resource "aws_iot_certificate" "producer" {
  active = true
}

resource "aws_iot_thing_principal_attachment" "producer" {
  thing     = aws_iot_thing.producer.name
  principal = aws_iot_certificate.producer.arn
}

# Least-privilege: connect as a pdm-ws-* client, publish only under the
# sensor topic tree.
resource "aws_iot_policy" "producer" {
  name = "${var.prefix}-producer"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["iot:Connect"]
        Resource = "arn:aws:iot:${var.aws_region}:${local.account_id}:client/${var.prefix}-*"
      },
      {
        Effect   = "Allow"
        Action   = ["iot:Publish"]
        Resource = "arn:aws:iot:${var.aws_region}:${local.account_id}:topic/${local.sensor_topic}/*"
      },
    ]
  })
}

resource "aws_iot_policy_attachment" "producer" {
  policy = aws_iot_policy.producer.name
  target = aws_iot_certificate.producer.arn
}

# Role the IoT rule assumes to write into Kinesis.
data "aws_iam_policy_document" "iot_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["iot.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "iot_to_kinesis" {
  name               = "${var.prefix}-iot-kinesis"
  assume_role_policy = data.aws_iam_policy_document.iot_assume.json
}

resource "aws_iam_role_policy" "iot_to_kinesis" {
  name = "${var.prefix}-iot-kinesis"
  role = aws_iam_role.iot_to_kinesis.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["kinesis:PutRecord"]
        Resource = aws_kinesis_stream.sensors.arn
      },
    ]
  })
}

# Route every sensor ping to Kinesis; newuuid() spreads records across shards.
resource "aws_iot_topic_rule" "sensors" {
  name        = local.rule_name
  enabled     = true
  sql         = "SELECT * FROM '${local.sensor_topic}/#'"
  sql_version = "2016-03-23"

  kinesis {
    role_arn      = aws_iam_role.iot_to_kinesis.arn
    stream_name   = aws_kinesis_stream.sensors.name
    partition_key = "$${newuuid()}"
  }
}

# Write the device credentials locally for the producer (gitignored).
resource "local_sensitive_file" "certificate" {
  filename = "${path.module}/certs/${var.prefix}-producer.cert.pem"
  content  = aws_iot_certificate.producer.certificate_pem
}

resource "local_sensitive_file" "private_key" {
  filename = "${path.module}/certs/${var.prefix}-producer.private.key"
  content  = aws_iot_certificate.producer.private_key
}
