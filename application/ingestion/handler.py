
import json
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3

from chunking import build_chunk_records
from embeddings import generate_embedding


# ---------------------------------------------------------
# AWS CLIENTS
# ---------------------------------------------------------

REGION = os.environ.get("AWS_REGION", "eu-central-1")

s3 = boto3.client("s3", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
s3vectors = boto3.client("s3vectors", region_name=REGION)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

TABLE_NAME = os.environ["DOCUMENT_REGISTRY_TABLE"]
VECTOR_BUCKET_NAME = os.environ["VECTOR_BUCKET_NAME"]
VECTOR_INDEX_NAME = os.environ["VECTOR_INDEX_NAME"]

table = dynamodb.Table(TABLE_NAME)


# ---------------------------------------------------------
# DOCUMENT INGESTION HANDLER
# ---------------------------------------------------------

def lambda_handler(event, context):
    records = event.get("Records", [])
    print(f"Received {len(records)} S3 record(s)")

    for record in records:
        bucket_name = record["s3"]["bucket"]["name"]
        object_key = unquote_plus(record["s3"]["object"]["key"])

        document_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        # Create a registry record before processing.
        table.put_item(
            Item={
                "documentId": document_id,
                "filename": object_key.split("/")[-1],
                "s3Key": object_key,
                "s3Bucket": bucket_name,
                "status": "PROCESSING",
                "createdAt": created_at,
            }
        )

        try:
            # -------------------------------------------------
            # 1. VALIDATE FILE TYPE
            # -------------------------------------------------

            if not object_key.lower().endswith(".txt"):
                raise ValueError(
                    "Unsupported file type. Only .txt is supported."
                )

            # -------------------------------------------------
            # 2. READ AND EXTRACT TEXT FROM S3
            # -------------------------------------------------

            response = s3.get_object(
                Bucket=bucket_name,
                Key=object_key,
            )

            file_bytes = response["Body"].read()
            extracted_text = file_bytes.decode("utf-8-sig")

            if not extracted_text.strip():
                raise ValueError("The uploaded text file is empty.")

            # -------------------------------------------------
            # 3. SPLIT TEXT INTO CHUNKS
            # -------------------------------------------------

            chunk_records = build_chunk_records(
                text=extracted_text,
                document_id=document_id,
                version=1,
                chunk_size=500,
                overlap=100,
            )

            if not chunk_records:
                raise ValueError("No chunks were produced.")

            print(
                f"Chunk preparation succeeded: "
                f"documentId={document_id}, "
                f"chunkCount={len(chunk_records)}"
            )

            # -------------------------------------------------
            # 4. GENERATE EMBEDDINGS AND PREPARE VECTORS
            # -------------------------------------------------

            vector_records = []

            for chunk in chunk_records:
                embedding = generate_embedding(chunk["text"])

                vector_records.append(
                    {
                        "key": chunk["chunkId"],
                        "data": {
                            "float32": embedding,
                        },
                        "metadata": {
                            "text": chunk["text"],
                            "source": object_key.split("/")[-1],
                            "documentId": document_id,
                            "documentVersion": chunk[
                                "documentVersion"
                            ],
                            "chunkIndex": chunk["chunkIndex"],
                        },
                    }
                )

            # -------------------------------------------------
            # 5. STORE VECTORS IN S3 VECTORS
            # -------------------------------------------------

            batch_size = 100

            for start in range(
                0, len(vector_records), batch_size
            ):
                batch = vector_records[
                    start:start + batch_size
                ]

                s3vectors.put_vectors(
                    vectorBucketName=VECTOR_BUCKET_NAME,
                    indexName=VECTOR_INDEX_NAME,
                    vectors=batch,
                )

            print(
                f"Vector storage succeeded: "
                f"documentId={document_id}, "
                f"vectorCount={len(vector_records)}"
            )

            # -------------------------------------------------
            # 6. MARK DOCUMENT AS COMPLETED
            # -------------------------------------------------

            table.update_item(
                Key={"documentId": document_id},
                UpdateExpression=(
                    "SET #status = :status, chunkCount = :count"
                ),
                ExpressionAttributeNames={
                    "#status": "status",
                },
                ExpressionAttributeValues={
                    ":status": "COMPLETED",
                    ":count": len(vector_records),
                },
            )

            print(
                f"Document processing completed: "
                f"documentId={document_id}"
            )

        except Exception as exc:
            print(
                f"Document processing failed: "
                f"documentId={document_id}, "
                f"error={type(exc).__name__}: {exc}"
            )

            table.update_item(
                Key={"documentId": document_id},
                UpdateExpression=(
                    "SET #status = :status, errorMessage = :error"
                ),
                ExpressionAttributeNames={
                    "#status": "status",
                },
                ExpressionAttributeValues={
                    ":status": "FAILED",
                    ":error": str(exc)[:1000],
                },
            )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {"message": "S3 ingestion event processed"}
        ),
    }
