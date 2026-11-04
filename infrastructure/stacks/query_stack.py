from pathlib import Path

from aws_cdk import (
    Stack,
    Duration,
    CfnOutput,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_apigatewayv2 as apigwv2,
    aws_apigatewayv2_integrations as integrations,
    aws_apigatewayv2_authorizers as authorizers,
    aws_cognito as cognito,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cloudwatch_actions,
)

from constructs import Construct


class AIKnowledgeAssistantQueryStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        region = self.region
        account = self.account

        # Existing S3 Vectors index
        vector_index_arn = (
            f"arn:aws:s3vectors:{region}:{account}:"
            "bucket/ai-knowledge-assistant-vectors/index/knowledge-index"
        )

        # Lambda deployment package
        project_root = Path(__file__).resolve().parents[2]
        lambda_package_path = project_root / "lambda_package"

        # RAG query Lambda
        query_function = lambda_.Function(
            self,
            "KnowledgeAssistantQueryFunction",
            function_name="ai-knowledge-assistant-query",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="query_handler.lambda_handler",
            code=lambda_.Code.from_asset(str(lambda_package_path)),
            timeout=Duration.seconds(60),
            memory_size=512,
            log_retention=logs.RetentionDays.ONE_MONTH,
            environment={
                "VECTOR_BUCKET_NAME": "ai-knowledge-assistant-vectors",
                "VECTOR_INDEX_NAME": "knowledge-index",
                "MODEL_ID": "eu.amazon.nova-lite-v1:0",
                "TOP_K": "3",
                "MAX_DISTANCE": "0.60",
            },
            description=(
                "Answers questions using retrieved policy documents, "
                "Amazon Bedrock, and Amazon S3 Vectors."
            ),
        )

        # Allow retrieval from the existing S3 Vectors index
        query_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "s3vectors:QueryVectors",
                    "s3vectors:GetVectors",
                ],
                resources=[vector_index_arn],
            )
        )

        # Allow Titan embeddings and Nova Lite inference
        query_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    (
                        f"arn:aws:bedrock:{region}::foundation-model/"
                        "amazon.titan-embed-text-v2:0"
                    ),
                    (
                        f"arn:aws:bedrock:{region}:{account}:"
                        "inference-profile/eu.amazon.nova-lite-v1:0"
                    ),
                    (
                        "arn:aws:bedrock:*::foundation-model/"
                        "amazon.nova-lite-v1:0"
                    ),
                ],
            )
        )

        # Cognito User Pool
        user_pool = cognito.UserPool(
            self,
            "KnowledgeAssistantUserPool",
            user_pool_name="ai-knowledge-assistant-users",
            sign_in_aliases=cognito.SignInAliases(email=True),
            self_sign_up_enabled=False,
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
        )

        # Cognito application client
        user_pool_client = user_pool.add_client(
            "KnowledgeAssistantAppClient",
            user_pool_client_name="ai-knowledge-assistant-client",
            generate_secret=False,
            auth_flows=cognito.AuthFlow(
                user_password=True,
                user_srp=True,
            ),
            prevent_user_existence_errors=True,
        )

        # JWT authorizer for API Gateway HTTP API
        jwt_authorizer = authorizers.HttpJwtAuthorizer(
            "KnowledgeAssistantJwtAuthorizer",
            jwt_issuer=user_pool.user_pool_provider_url,
            jwt_audience=[user_pool_client.user_pool_client_id],
        )

        # Create HTTP API
        http_api = apigwv2.HttpApi(
            self,
            "KnowledgeAssistantHttpApi",
            api_name="ai-knowledge-assistant-api",
            description=(
                "Authenticated HTTP API for the AI Knowledge Assistant."
            ),
        )

        # POST /ask requires a valid Cognito JWT
        http_api.add_routes(
            path="/ask",
            methods=[apigwv2.HttpMethod.POST],
            integration=integrations.HttpLambdaIntegration(
                "KnowledgeAssistantQueryIntegration",
                query_function,
            ),
            authorizer=jwt_authorizer,
        )

        # Outputs
        CfnOutput(
            self,
            "QueryApiEndpoint",
            value=f"{http_api.api_endpoint}/ask",
            description="Authenticated API endpoint.",
        )

        CfnOutput(
            self,
            "CognitoUserPoolId",
            value=user_pool.user_pool_id,
            description="Cognito User Pool ID.",
        )

        CfnOutput(
            self,
            "CognitoAppClientId",
            value=user_pool_client.user_pool_client_id,
            description="Cognito application client ID.",
        )

        # CloudWatch monitoring dashboard
        dashboard = cloudwatch.Dashboard(
            self,
            "KnowledgeAssistantDashboard",
            dashboard_name="AI-Knowledge-Assistant",
        )

        # Lambda operational metrics
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lambda Invocations",
                left=[
                    query_function.metric_invocations(
                        period=Duration.minutes(5),
                    )
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Lambda Errors",
                left=[
                    query_function.metric_errors(
                        period=Duration.minutes(5),
                    )
                ],
                width=12,
            ),
        )

        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lambda Duration",
                left=[
                    query_function.metric_duration(
                        statistic="p95",
                        period=Duration.minutes(5),
                    )
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Lambda Throttles",
                left=[
                    query_function.metric_throttles(
                        period=Duration.minutes(5),
                    )
                ],
                width=12,
            ),
        )

        # Alarm when Lambda reports errors
        error_alarm = cloudwatch.Alarm(
            self,
            "KnowledgeAssistantLambdaErrorAlarm",
            metric=query_function.metric_errors(
                period=Duration.minutes(5),
            ),
            threshold=1,
            evaluation_periods=1,
            alarm_description=(
                "The Knowledge Assistant query Lambda encountered an error."
            ),
        )

        # Alarm when Lambda approaches its 60-second timeout
        duration_alarm = cloudwatch.Alarm(
            self,
            "KnowledgeAssistantLambdaDurationAlarm",
            metric=query_function.metric_duration(
                statistic="p95",
                period=Duration.minutes(5),
            ),
            threshold=45000,
            evaluation_periods=2,
            alarm_description=(
                "The Knowledge Assistant query Lambda p95 duration "
                "exceeded 45 seconds."
            ),
        )