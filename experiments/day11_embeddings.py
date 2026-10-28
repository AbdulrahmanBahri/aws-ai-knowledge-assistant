
import json
import math

import boto3


REGION = "eu-central-1"
MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024


def generate_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")

    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
    )

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
    embedding = response_body["embedding"]

    if len(embedding) != DIMENSIONS:
        raise ValueError(
            f"Expected {DIMENSIONS} dimensions, "
            f"received {len(embedding)}."
        )

    if not all(math.isfinite(value) for value in embedding):
        raise ValueError("Embedding contains invalid numbers.")

    return embedding


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cannot compare a zero-magnitude vector.")

    return dot_product / (magnitude_a * magnitude_b)


if __name__ == "__main__":
    remote_work_a = (
        "Employees may work remotely up to two days per week "
        "with manager approval."
    )
    remote_work_b = (
        "Staff can work from home twice a week "
        "if their manager approves."
    )
    insurance = (
        "Employees receive health insurance through the company."
    )

    vector_a = generate_embedding(remote_work_a)
    vector_b = generate_embedding(remote_work_b)
    vector_c = generate_embedding(insurance)

    print(
        "A vs B (related):",
        round(cosine_similarity(vector_a, vector_b), 4),
    )
    print(
        "A vs C (different topics):",
        round(cosine_similarity(vector_a, vector_c), 4),
    )
