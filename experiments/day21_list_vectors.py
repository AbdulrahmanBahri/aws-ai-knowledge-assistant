import json
import boto3

REGION = "eu-central-1"
VECTOR_BUCKET_NAME = "ai-knowledge-assistant-vectors"
VECTOR_INDEX_NAME = "knowledge-index"

s3vectors = boto3.client(
    "s3vectors",
    region_name=REGION,
)


def list_vectors():
    """List stored vector keys and their metadata."""

    next_token = None
    total = 0

    while True:
        params = {
            "vectorBucketName": VECTOR_BUCKET_NAME,
            "indexName": VECTOR_INDEX_NAME,
            "maxResults": 100,
            "returnMetadata": True,
        }

        if next_token:
            params["nextToken"] = next_token

        response = s3vectors.list_vectors(**params)

        for vector in response.get("vectors", []):
            metadata = vector.get("metadata") or {}

            result = {
                "key": vector.get("key"),
                "source": metadata.get("source"),
                "documentId": metadata.get("documentId"),
                "documentVersion": metadata.get("documentVersion"),
                "chunkIndex": metadata.get("chunkIndex"),
                "text": metadata.get("text"),
            }

            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("-" * 60)

            total += 1

        next_token = response.get("nextToken")

        if not next_token:
            break

    print(f"\nTotal vectors listed: {total}")


if __name__ == "__main__":
    list_vectors()