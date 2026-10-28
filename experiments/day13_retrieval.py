
import json
import sys
from pathlib import Path

import boto3

# Make the existing ingestion embedding module importable
# when this script is run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INGESTION_DIR = PROJECT_ROOT / "application" / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))

from embeddings import generate_embedding


REGION = "eu-central-1"
VECTOR_BUCKET_NAME = "ai-knowledge-assistant-vectors"
VECTOR_INDEX_NAME = "knowledge-index"
TOP_K = 3

s3vectors = boto3.client(
    "s3vectors",
    region_name=REGION,
)


# Six evaluation questions for the multi-document corpus.
QUESTIONS = [
    {
        "id": "remote_work",
        "question": "How many days per week can staff work remotely?",
        "expected_source": "remote_work_policy.txt",
        "answerable": True,
    },
    {
        "id": "health_insurance",
        "question": "Does the company cover employees' health insurance?",
        "expected_source": "employee_benefits_policy.txt",
        "answerable": True,
    },
    {
        "id": "annual_leave",
        "question": "Who approves annual leave requests?",
        "expected_source": "annual_leave_policy.txt",
        "answerable": True,
    },
    {
        "id": "family_coverage",
        "question": "Are employees' dependents covered by health insurance?",
        "expected_source": None,
        "answerable": False,
    },
    {
        "id": "leave_carryover",
        "question": "Can unused annual leave be carried over?",
        "expected_source": None,
        "answerable": False,
    },
    {
        "id": "international_remote_work",
        "question": "Can I work remotely from another country?",
        "expected_source": None,
        "answerable": False,
    },
    {
       "id": "international_remote_work_approval",
       "question": "What approval is required for international remote work?",
       "expected_source": "day21-lifecycle-test-new.txt",
       "answerable": True,
    },
]


def retrieve_chunks(question: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the nearest stored document chunks for a question."""

    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100.")

    # 1. Embed the question using the same model as ingestion.
    query_embedding = generate_embedding(question)

    if len(query_embedding) != 1024:
        raise ValueError(
            f"Expected 1024 embedding dimensions, "
            f"received {len(query_embedding)}."
        )

    # 2. Search the existing vector index.
    response = s3vectors.query_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=VECTOR_INDEX_NAME,
        queryVector={"float32": query_embedding},
        topK=top_k,
        returnDistance=True,
        returnMetadata=True,
    )

    # 3. Return the fields needed to evaluate retrieval.
    results = []

    for vector in response.get("vectors", []):
        metadata = vector.get("metadata") or {}

        results.append({
            "chunk_id": vector.get("key"),
            "distance": vector.get("distance"),
            "text": metadata.get("text"),
            "source": metadata.get("source"),
            "document_id": metadata.get("documentId"),
            "document_version": metadata.get("documentVersion"),
            "chunk_index": metadata.get("chunkIndex"),
        })

    return results


def evaluate_question(case: dict) -> dict:
    """Retrieve evidence and report basic source-matching results."""

    question = case["question"]
    matches = retrieve_chunks(question)
    expected_evidence_found = any(
      match.get("source") == case["expected_source"]
      and "international remote work requires explicit written approval"
      in (match.get("text") or "").lower()
      for match in matches
    )

    expected_source = case["expected_source"]

    # Source matching is a useful first check, not a complete
    # judgment of whether the evidence actually answers the question.
    expected_source_found = (
        expected_source is not None
        and any(
            (match.get("source") or "").lower()
            == expected_source.lower()
            for match in matches
        )
    )

    result = {
        "id": case["id"],
        "question": question,
        "answerable": case["answerable"],
        "expected_source": expected_source,
        "expected_source_found": expected_source_found,
        "retrieved_count": len(matches),
        "matches": matches,
        "expected_evidence_found": expected_evidence_found,
    }

    return result


def main():
    all_results = []

    for case in QUESTIONS:
        print("\n" + "=" * 72)
        print(f"Case: {case['id']}")
        print(f"Question: {case['question']}")
        print(f"Expected answerable: {case['answerable']}")
        print(f"Expected source: {case['expected_source']}")
        print()

        try:
            result = evaluate_question(case)
        except Exception as exc:
            print(f"Retrieval failed: {type(exc).__name__}: {exc}")
            all_results.append({
                "id": case["id"],
                "question": case["question"],
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        matches = result["matches"]

        if not matches:
            print("No matching vectors were returned.")
        else:
            for rank, match in enumerate(matches, start=1):
                print(f"--- Retrieved chunk {rank} ---")
                print(f"Source: {match['source']}")
                print(f"Distance: {match['distance']}")
                print(f"Chunk ID: {match['chunk_id']}")
                print("Text:")
                print(match["text"])
                print()

        print(
            "Expected source found: "
            f"{result['expected_source_found']}"
        )
        print(
            "Expected evidence found: "
            f"{result.get('expected_evidence_found', False)}"
        )

        all_results.append(result)

    # Save the retrieval evidence so it can be reviewed later.
    output_path = (
        PROJECT_ROOT
        / "tests"
        / "evaluation"
        / "day20_retrieval_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    successful_queries = sum(
        "error" not in result for result in all_results
    )
    source_checks = [
        result for result in all_results
        if "error" not in result
        and result["expected_source"] is not None
    ]
    correct_source_matches = sum(
        result["expected_source_found"]
        for result in source_checks
    )

    print("\n" + "=" * 72)
    print("DAY 20 RETRIEVAL SUMMARY")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Queries completed: {successful_queries}")
    print(
        "Supported questions with expected source retrieved: "
        f"{correct_source_matches}/{len(source_checks)}"
    )
    print(f"Saved results: {output_path}")


if __name__ == "__main__":
    main()
