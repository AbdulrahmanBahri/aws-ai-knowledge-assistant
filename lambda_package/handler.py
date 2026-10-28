import hashlib
import json
import os
from datetime import datetime, timezone
from urllib.parse import unquote_plus
from botocore.exceptions import ClientError

import boto3
from boto3.dynamodb.conditions import Attr

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

PUT_BATCH_SIZE = 100
DELETE_BATCH_SIZE = 500


# ---------------------------------------------------------
# DOCUMENT IDENTITY
# ---------------------------------------------------------

def make_document_id(bucket_name, object_key):
    """Create a stable ID for an S3 object."""

    identity = f"{bucket_name}\0{object_key}"

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def make_content_hash(text):
    """Identify a specific revision by its content."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# VECTOR CLEANUP
# ---------------------------------------------------------

def delete_vector_keys(vector_keys):
    """Delete vectors by exact key, in batches."""

    unique_keys = list(dict.fromkeys(vector_keys))

    for start in range(0, len(unique_keys), DELETE_BATCH_SIZE):
        batch = unique_keys[
            start:start + DELETE_BATCH_SIZE
        ]

        if not batch:
            continue

        s3vectors.delete_vectors(
            vectorBucketName=VECTOR_BUCKET_NAME,
            indexName=VECTOR_INDEX_NAME,
            keys=batch,
        )

        print(
            f"Deleted {len(batch)} obsolete vector(s)"
        )


def find_legacy_records(bucket_name, object_key, document_id):
    """
    Find registry records created by the previous
    UUID-based implementation for this S3 object.
    """

    legacy_records = []
    scan_kwargs = {}

    while True:
        response = table.scan(**scan_kwargs)

        for item in response.get("Items", []):
            if (
                item.get("s3Bucket") == bucket_name
                and item.get("s3Key") == object_key
                and item.get("documentId") != document_id
            ):
                legacy_records.append(item)

        last_key = response.get("LastEvaluatedKey")

        if not last_key:
            break

        scan_kwargs["ExclusiveStartKey"] = last_key

    return legacy_records


def get_legacy_vector_keys(legacy_records):
    """
    Reconstruct vector keys used by the previous handler.

    The previous handler always used version=1 and the
    format: <UUID>:v1:chunk-0001
    """

    keys = []

    for item in legacy_records:
        document_id = item.get("documentId")
        chunk_count = int(item.get("chunkCount", 0))

        if not document_id or chunk_count <= 0:
            continue

        for index in range(1, chunk_count + 1):
            keys.append(
                f"{document_id}:v1:chunk-{index:04d}"
            )

    return keys


# ---------------------------------------------------------
# DYNAMODB REGISTRY
# ---------------------------------------------------------

def mark_processing(
    document_id,
    bucket_name,
    object_key,
    content_hash,
    created_at,
):
    """Update the existing registry item without erasing
    previously stored vectorKeys.
    """

    table.update_item(
        Key={"documentId": document_id},
        UpdateExpression=(
            "SET #status = :status, "
            "s3Bucket = :bucket, "
            "s3Key = :key, "
            "filename = :filename, "
            "contentHash = :content_hash, "
            "updatedAt = :updated_at"
        ),
        ExpressionAttributeNames={
            "#status": "status",
        },
        ExpressionAttributeValues={
            ":status": "PROCESSING",
            ":bucket": bucket_name,
            ":key": object_key,
            ":filename": object_key.split("/")[-1],
            ":content_hash": content_hash,
            ":updated_at": created_at,
        },
    )


def mark_failed(document_id, error):
    """Record a failure while preserving existing attributes."""

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
            ":error": str(error)[:1000],
        },
    )


# ---------------------------------------------------------
# DOCUMENT PROCESSING
# ---------------------------------------------------------

def process_document(bucket_name, object_key):
    document_id = make_document_id(
        bucket_name,
        object_key,
    )

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    # -----------------------------------------------------
    # 1. VALIDATE FILE TYPE
    # -----------------------------------------------------

    if not object_key.lower().endswith(".txt"):
        raise ValueError(
            "Unsupported file type. Only .txt is supported."
        )

    # -----------------------------------------------------
    # 2. READ TEXT FROM S3
    # -----------------------------------------------------

    response = s3.get_object(
        Bucket=bucket_name,
        Key=object_key,
    )

    file_bytes = response["Body"].read()
    extracted_text = file_bytes.decode("utf-8-sig")

    if not extracted_text.strip():
        raise ValueError("The uploaded text file is empty.")

    content_hash = make_content_hash(extracted_text)

    # -----------------------------------------------------
    # 3. READ CURRENT REGISTRY STATE
    # -----------------------------------------------------

    existing_response = table.get_item(
        Key={"documentId": document_id},
        ConsistentRead=True,
    )

    existing_item = existing_response.get("Item", {})

    old_vector_keys = existing_item.get("vectorKeys", [])

    # Avoid re-embedding identical content when the registry
    # already records a completed revision.
    if (
        existing_item.get("status") == "COMPLETED"
        and existing_item.get("contentHash") == content_hash
        and old_vector_keys
    ):
        print(
            f"Document unchanged; skipping: "
            f"documentId={document_id}"
        )
        return

    # Find records from the previous UUID-based handler.
    legacy_records = find_legacy_records(
        bucket_name,
        object_key,
        document_id,
    )

    mark_processing(
        document_id=document_id,
        bucket_name=bucket_name,
        object_key=object_key,
        content_hash=content_hash,
        created_at=created_at,
    )

    try:
        # -------------------------------------------------
        # 4. SPLIT TEXT INTO CHUNKS
        # -------------------------------------------------

        chunk_records = build_chunk_records(
            text=extracted_text,
            document_id=document_id,
            version=content_hash,
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
        # 5. GENERATE EMBEDDINGS
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
                        "s3Bucket": bucket_name,
                        "s3Key": object_key,
                        "documentId": document_id,
                        "documentVersion": content_hash,
                        "chunkIndex": chunk["chunkIndex"],
                    },
                }
            )

        new_vector_keys = [
            record["key"]
            for record in vector_records
        ]

        # -------------------------------------------------
        # 6. STORE NEW VECTORS
        # -------------------------------------------------

        for start in range(
            0,
            len(vector_records),
            PUT_BATCH_SIZE,
        ):
            batch = vector_records[
                start:start + PUT_BATCH_SIZE
            ]

            s3vectors.put_vectors(
                vectorBucketName=VECTOR_BUCKET_NAME,
                indexName=VECTOR_INDEX_NAME,
                vectors=batch,
            )

        print(
            f"Vector storage succeeded: "
            f"documentId={document_id}, "
            f"vectorCount={len(new_vector_keys)}"
        )

        # -------------------------------------------------
        # 7. DELETE PREVIOUS VECTORS
        # -------------------------------------------------

        legacy_vector_keys = get_legacy_vector_keys(
            legacy_records
        )

        obsolete_keys = (
            old_vector_keys + legacy_vector_keys
        )

        # Never delete keys belonging to the new revision.
        obsolete_keys = [
            key for key in obsolete_keys
            if key not in set(new_vector_keys)
        ]

        delete_vector_keys(obsolete_keys)

        # -------------------------------------------------
        # 8. MARK THE NEW REVISION COMPLETED
        # -------------------------------------------------

        table.update_item(
            Key={"documentId": document_id},
            UpdateExpression=(
                "SET #status = :status, "
                "chunkCount = :count, "
                "vectorKeys = :vector_keys, "
                "contentHash = :content_hash, "
                "documentVersion = :version, "
                "completedAt = :completed_at "
                "REMOVE errorMessage"
            ),
            ExpressionAttributeNames={
                "#status": "status",
            },
            ExpressionAttributeValues={
                ":status": "COMPLETED",
                ":count": len(new_vector_keys),
                ":vector_keys": new_vector_keys,
                ":content_hash": content_hash,
                ":version": content_hash,
                ":completed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
        )

        # Remove obsolete registry entries after successful
        # indexing and cleanup.
        for item in legacy_records:
            table.delete_item(
                Key={"documentId": item["documentId"]}
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

        mark_failed(document_id, exc)

        raise


# ---------------------------------------------------------

def delete_document(bucket_name, object_key):
    """
    Remove indexed content when an S3 delete marker
    makes the document unavailable.

    If the document has already been uploaded again,
    leave its registry entry and vectors untouched.
    """

    document_id = make_document_id(
        bucket_name,
        object_key,
    )

    # S3 notifications may arrive out of order.
    # If the current object exists again, do not clean it up.
    try:
        s3.head_object(
            Bucket=bucket_name,
            Key=object_key,
        )

        print(
            "Skipping deletion cleanup because the "
            f"object exists again: {object_key}"
        )
        return

    except ClientError as exc:
        error_code = exc.response.get(
            "Error", {}
        ).get("Code", "")

        if error_code not in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            raise

    # Read the current document registry record.
    response = table.get_item(
        Key={"documentId": document_id},
        ConsistentRead=True,
    )

    item = response.get("Item")

    if not item:
        print(
            "No registry record found for deleted "
            f"document: {object_key}"
        )
        return

    # Delete vectors first. If this fails, retain the
    # registry record so cleanup can be retried.
    vector_keys = item.get("vectorKeys", [])

    if vector_keys:
        delete_vector_keys(vector_keys)

    # Delete the registry record only after vector cleanup.
    table.delete_item(
        Key={"documentId": document_id}
    )

    print(
        "Document deletion cleanup completed: "
        f"documentId={document_id}, "
        f"deletedVectors={len(vector_keys)}"
    )

# ---------------------------------------------------------
# LAMBDA HANDLER
# ---------------------------------------------------------

def lambda_handler(event, context):
    records = event.get("Records", [])

    print(f"Received {len(records)} S3 record(s)")

    for record in records:
        event_name = record.get("eventName", "")

        # Ignore records that aren't object-created or
        # delete-marker-created events.
        if event_name.startswith("ObjectCreated:"):
            bucket_name = record["s3"]["bucket"]["name"]

            object_key = unquote_plus(
                record["s3"]["object"]["key"]
            )

            process_document(
                bucket_name=bucket_name,
                object_key=object_key,
            )

        elif event_name == "ObjectRemoved:DeleteMarkerCreated":
            bucket_name = record["s3"]["bucket"]["name"]

            object_key = unquote_plus(
                record["s3"]["object"]["key"]
            )

            delete_document(
                bucket_name=bucket_name,
                object_key=object_key,
            )

        else:
            print(
                f"Ignoring unsupported event type: "
                f"{event_name}"
            )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {"message": "S3 document event processed"}
        ),
    }