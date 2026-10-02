provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = {
      Owner       = var.owner_email
      RemoveAfter = var.remove_after
      Project     = var.project
    }
  }
}
