# providers.terraform

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "AudioPredictiveMaintenance"
      Environment = "Dev"
      ManagedBy   = "Terraform"
      Owner       = "AyushPradhan"
    }
  }
}