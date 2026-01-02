# terraform/versions.tf

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # remote state configuration
  backend "s3" {
    bucket = "ayush-tfstate-audio-mlops"
    key = "audio-mlops/terraform.tfstate"
    region = "eu-west-1"
    dynamodb_table = "terraform-locks"
    encrypt = true
  }
}