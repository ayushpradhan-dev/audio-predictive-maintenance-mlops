# terraform/api_gateway.tf

# IAM role for lambda, allows lambda to assume identity
resource "aws_iam_role" "lambda_exec_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# attach policies to role, allow writing logs to CloudWatch
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# allows pulling images from ECR
resource "aws_iam_policy" "ecr_pull_policy" {
  name        = "${var.project_name}-ecr-pull"
  description = "Allow lambda to pull images from ECR"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability"
        ]
        Resource = aws_ecr_repository.app_repo.arn
      },
      {
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_ecr_pull" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = aws_iam_policy.ecr_pull_policy.arn
}

# the lambda function
resource "aws_lambda_function" "api_lambda" {
  function_name = "${var.project_name}-api"
  role          = aws_iam_role.lambda_exec_role.arn
  timeout       = 300
  memory_size   = 3008 # 3GB
  package_type  = "Image"

  # link to ECR image, use latest tag for initial setup
  image_uri = "${aws_ecr_repository.app_repo.repository_url}:${var.image_tag}"

  environment {
    variables = {
      ENV = "production"
    }
  }
}