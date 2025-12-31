# outputs.tf

output "ecr_repository_url" {
  description = "The URL of the ECR repository"
  value       = aws_ecr_repository.app_repo.repository_url
}

output "ecr_registry_id" {
  description = "The registry ID"
  value       = aws_ecr_repository.app_repo.registry_id
}

