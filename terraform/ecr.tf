# terraform/main.tf

# ECR repository
resource "aws_ecr_repository" "app_repo" {
  name                 = var.ecr_repo_name
  image_tag_mutability = "MUTABLE" # allow tag overwriting
  force_delete         = false     # can't delete repo if it has an image

  # scan images for vulnerabilities on push
  image_scanning_configuration {
    scan_on_push = true
  }
}

# lifecycle policy for cost management
# auto delete untagged or old images
resource "aws_ecr_lifecycle_policy" "cleanup_policy" {
  repository = aws_ecr_repository.app_repo.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only the last 2 tagged images"

        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 2
        }

        action = {
          type = "expire"
        }
      }
    ]
  })
}