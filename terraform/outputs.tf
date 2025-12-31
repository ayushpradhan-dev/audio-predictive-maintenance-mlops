# terraform/outputs.tf

output "ecr_repository_url" {
  description = "The URL of the ECR repository"
  value       = aws_ecr_repository.app_repo.repository_url
}

output "ecr_registry_id" {
  description = "The registry ID"
  value       = aws_ecr_repository.app_repo.registry_id
}

output "api_gateway_url" {
  description = "Base URL for the API Gateway stage"
  value       = aws_apigatewayv2_stage.default.invoke_url
}

