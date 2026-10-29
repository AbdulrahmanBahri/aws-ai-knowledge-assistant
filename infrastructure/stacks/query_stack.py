from pathlib import Path

from aws_cdk import (
    Stack,
    Duration,
    aws_iam as iam,
    aws_lambda as lambda_,
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

        # Query Lambda
        query_function = lambda_.Function(
            self,
            "KnowledgeAssistantQueryFunction",
            function_name="ai-knowledge-assistant-query",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="query_handler.lambda_handler",
            code=lambda_.Code.from_asset(str(lambda_package_path)),
            timeout=Duration.seconds(60),
            memory_size=512,
            environment={
                "VECTOR_BUCKET_NAME": "ai-knowledge-assistant-vectors",
                "VECTOR_INDEX_NAME": "knowledge-index",
                "MODEL_ID": "eu.amazon.nova-lite-v1:0",
                "TOP_K": "3",
                "MAX_DISTANCE": "0.35",
            },
            description=(
                "Answers questions using retrieved policy documents, "
                "Amazon Bedrock, and Amazon S3 Vectors."
            ),
        )

        # Allow retrieval from the existing S3 Vectors index.
        query_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "s3vectors:QueryVectors",
                    "s3vectors:GetVectors",
                ],
                resources=[vector_index_arn],
            )
        )

        # Allow Titan Text Embeddings V2 and Nova Lite inference.
        query_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    # Titan embedding model
                    (
                        f"arn:aws:bedrock:{region}::foundation-model/"
                        "amazon.titan-embed-text-v2:0"
                    ),

                    # EU Nova Lite inference profile
                    (
                        f"arn:aws:bedrock:{region}:{account}:"
                        "inference-profile/eu.amazon.nova-lite-v1:0"
                    ),

                    # Nova Lite foundation model resources
                    # used by the inference profile.
                    (
                        "arn:aws:bedrock:*::foundation-model/"
                        "amazon.nova-lite-v1:0"
                    ),
                ],
            )
        )