# terraform/api_gateway.tf

# API gateway
resource "aws_apigatewayv2_api" "http_api" {
  name          = "${var.project_name}-gateway"
  protocol_type = "HTTP"
}

# the stage, $default means changes are deployed immediately
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http_api.id
  name        = "$default"
  auto_deploy = true
}

# integration, connects API gateway to the lambda function
resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id           = aws_apigatewayv2_api.http_api.id
  integration_type = "AWS_PROXY"

  integration_uri        = aws_lambda_function.api_lambda.invoke_arn
  integration_method     = "POST" # lambda invocation requires POST
  payload_format_version = "2.0"
}

# the route, routes all traffic to the lambda function
# FastAPI handles internal routing (/predict, /docs)
resource "aws_apigatewayv2_route" "any" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "ANY /{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# permission for API gateway to invoke lambda
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api_lambda.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}