from aws_cdk import (
    RemovalPolicy,
    Stack,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    aws_s3vectors as s3vectors,
    CfnOutput,
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
                {"key": "Project", "value": "AIKnowledgeAssistant"},
                {"key": "Environment", "value": "dev"},
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
        # OUTPUTS
        # ---------------------------------------------------------

        CfnOutput(
            self,
            "DocumentBucketName",
            value=self.document_bucket.bucket_name,
            description="S3 bucket containing uploaded documents",
        )

        CfnOutput(
            self,
            "DocumentRegistryTableName",
            value=self.document_registry.table_name,
            description="DynamoDB table tracking document lifecycle",
        )

        CfnOutput(
            self,
            "VectorBucketName",
            value=self.vector_bucket.ref,
            description="S3 Vector bucket containing knowledge vectors",
        )

        CfnOutput(
            self,
            "VectorIndexName",
            value=self.vector_index.ref,
            description="S3 Vector index used for semantic search",
        )