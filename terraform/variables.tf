# terraform/variables.tf

variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "eu-west-1"
}

variable "project_name" {
  description = "Project name prefix for resources"
  type        = string
  default     = "audio-mlops"
}

variable "ecr_repo_name" {
  description = "Name of the ECR repository"
  type        = string
  default     = "audio-predictive-maintenance"
}