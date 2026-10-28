import boto3

REGION = "eu-central-1"
VECTOR_BUCKET_NAME = "ai-knowledge-assistant-vectors"
VECTOR_INDEX_NAME = "knowledge-index"

# Exact key copied from today's vector listing.
STALE_VECTOR_KEY = (
    "1fe39e98-417d-4a5f-841c-9fd2fe68dc69"
    ":v1:chunk-0001"
)

s3vectors = boto3.client(
    "s3vectors",
    region_name=REGION,
)


def main():
    confirmation = input(
        "Delete the day12-test.txt vector? "
        "Type DELETE to confirm: "
    )

    if confirmation != "DELETE":
        print("Cancelled. No vectors were deleted.")
        return

    response = s3vectors.delete_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=VECTOR_INDEX_NAME,
        keys=[STALE_VECTOR_KEY],
    )

    print("Deletion request succeeded.")
    print(response)


if __name__ == "__main__":
    main()