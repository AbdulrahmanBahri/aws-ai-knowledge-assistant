from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_s3vectors as s3vectors,
)
from constructs import Construct


class StorageStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        super().__init__(
            scope,
            construct_id,
            **kwargs,
        )

        # ---------------------------------------------------------
        # S3 DOCUMENT BUCKET
        # ---------------------------------------------------------

        self.document_bucket = s3.Bucket(
            self,
            "DocumentBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # ---------------------------------------------------------
        # DYNAMODB DOCUMENT REGISTRY
        # ---------------------------------------------------------

        self.document_registry = dynamodb.Table(
            self,
            "DocumentRegistry",
            partition_key=dynamodb.Attribute(
                name="documentId",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ---------------------------------------------------------
        # S3 VECTOR BUCKET
        # ---------------------------------------------------------

        vector_bucket_name = "ai-knowledge-assistant-vectors"

        self.vector_bucket = s3vectors.CfnVectorBucket(
            self,
            "KnowledgeVectorBucket",
            vector_bucket_name=vector_bucket_name,
            tags=[
                {
                    "key": "Project",
                    "value": "AIKnowledgeAssistant",
                },
                {
                    "key": "Environment",
                    "value": "dev",
                },
            ],
        )

        # ---------------------------------------------------------
        # S3 VECTOR INDEX
        # ---------------------------------------------------------

        self.vector_index = s3vectors.CfnIndex(
            self,
            "KnowledgeVectorIndex",
            vector_bucket_name=vector_bucket_name,
            index_name="knowledge-index",
            data_type="float32",
            dimension=1024,
            distance_metric="cosine",
            metadata_configuration=(
                s3vectors.CfnIndex.MetadataConfigurationProperty(
                    non_filterable_metadata_keys=[
                        "text",
                        "source",
                    ]
                )
            ),
        )

        self.vector_index.add_dependency(
            self.vector_bucket
        )

        # ---------------------------------------------------------
        # DOCUMENT INGESTION LAMBDA
        # ---------------------------------------------------------

        self.ingestion_function = lambda_.Function(
            self,
            "DocumentIngestionFunction",
            runtime=lambda_.Runtime.PYTHON_3_13,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("lambda_package"),
            environment={
                "DOCUMENT_REGISTRY_TABLE": (
                    self.document_registry.table_name
                ),
                "VECTOR_BUCKET_NAME": vector_bucket_name,
                "VECTOR_INDEX_NAME": "knowledge-index",
            },
            timeout=Duration.seconds(60),
        )

        # ---------------------------------------------------------
        # LAMBDA PERMISSIONS
        # ---------------------------------------------------------

        # Allow reading and writing document registry records.
        # The handler needs GetItem, UpdateItem, Scan, and
        # DeleteItem in addition to its existing write operations.
        self.document_registry.grant_read_write_data(
            self.ingestion_function
        )

        # Allow the Lambda to read uploaded documents.
        self.document_bucket.grant_read(
            self.ingestion_function
        )

        # Allow writing vectors to the configured index.
        self.ingestion_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "s3vectors:PutVectors",
                ],
                resources=[
                    self.vector_index.attr_index_arn,
                ],
            )
        )

        # Allow deleting obsolete vectors from the same index.
        self.ingestion_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "s3vectors:DeleteVectors",
                ],
                resources=[
                    self.vector_index.attr_index_arn,
                ],
            )
        )

        # Allow invocation of Titan Text Embeddings V2.
        self.ingestion_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                ],
                resources=[
                    (
                        f"arn:{self.partition}:bedrock:"
                        f"{self.region}::foundation-model/"
                        "amazon.titan-embed-text-v2:0"
                    ),
                ],
            )
        )

        # ---------------------------------------------------------
        # S3 EVENT NOTIFICATION
        # ---------------------------------------------------------

        self.document_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(
                self.ingestion_function
            ),
        )

        # ---------------------------------------------------------
        # OUTPUTS
        # ---------------------------------------------------------

        CfnOutput(
            self,
            "DocumentBucketName",
            value=self.document_bucket.bucket_name,
            description=(
                "S3 bucket containing uploaded documents"
            ),
        )

        CfnOutput(
            self,
            "DocumentRegistryTableName",
            value=self.document_registry.table_name,
            description=(
                "DynamoDB table tracking document lifecycle"
            ),
        )

        CfnOutput(
            self,
            "VectorBucketName",
            value=self.vector_bucket.ref,
            description=(
                "S3 Vectors bucket containing knowledge vectors"
            ),
        )

        CfnOutput(
            self,
            "VectorIndexName",
            value=self.vector_index.ref,
            description=(
                "S3 Vectors index used for semantic search"
            ),
        )