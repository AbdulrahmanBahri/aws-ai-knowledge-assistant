
import json
import math
import boto3

REGION = "eu-central-1"
MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
)


def generate_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")

    request_body = {
        "inputText": text,
        "dimensions": DIMENSIONS,
        "normalize": True,
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(request_body),
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    embedding = response_body.get("embedding")

    if not isinstance(embedding, list):
        raise ValueError("Bedrock response contains no valid embedding.")

    if len(embedding) != DIMENSIONS:
        raise ValueError(
            f"Expected {DIMENSIONS} dimensions, "
            f"received {len(embedding)}."
        )

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in embedding
    ):
        raise ValueError("Embedding contains invalid numeric values.")

    return embedding
