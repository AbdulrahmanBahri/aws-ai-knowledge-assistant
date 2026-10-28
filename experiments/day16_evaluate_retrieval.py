
import json
from pathlib import Path

from day13_retrieval import retrieve_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "tests"
    / "evaluation"
    / "retrieval_cases.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "tests"
    / "evaluation"
    / "day17_results.json"
)


def evaluate_case(case: dict) -> dict:
    matches = retrieve_chunks(case["question"], top_k=3)

    print("\n" + "=" * 70)
    print(f"Case: {case['id']}")
    print(f"Question: {case['question']}")
    print(f"Expected answerable: {case['answerable']}")
    print(f"Retrieved chunks: {len(matches)}")

    retrieved_evidence = []

    for rank, match in enumerate(matches, start=1):
        text = match.get("text") or ""
        source = match.get("source") or "Unknown source"
        distance = match.get("distance")

        print(f"\n--- Retrieved chunk {rank} ---")
        print(f"Source: {source}")
        print(f"Distance: {distance}")
        print(f"Chunk ID: {match.get('chunk_id')}")
        print(f"Text:\n{text or '[No text returned]'}")

        retrieved_evidence.append({
            "rank": rank,
            "chunk_id": match.get("chunk_id"),
            "source": source,
            "distance": distance,
            "text": text,
        })

    print("\nJudge the retrieved evidence for this question.")
    print("RELEVANT   = evidence supports answering the question")
    print("IRRELEVANT = evidence does not support answering it")
    print("REVIEW     = uncertain; needs further inspection")

    valid_judgments = {"RELEVANT", "IRRELEVANT", "REVIEW"}

    while True:
        judgment = input(
            "Your judgment [RELEVANT/IRRELEVANT/REVIEW]: "
        ).strip().upper()

        if judgment in valid_judgments:
            break

        print("Please enter RELEVANT, IRRELEVANT, or REVIEW.")

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_answerable": case["answerable"],
        "retrieved_count": len(matches),
        "evidence": retrieved_evidence,
        "manual_judgment": judgment,
    }


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    results = []

    for case in cases:
        results.append(evaluate_case(case))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    relevant = sum(
        result["manual_judgment"] == "RELEVANT"
        for result in results
    )
    irrelevant = sum(
        result["manual_judgment"] == "IRRELEVANT"
        for result in results
    )
    review = sum(
        result["manual_judgment"] == "REVIEW"
        for result in results
    )

    print("\n" + "=" * 70)
    print("DAY 17 EVALUATION SUMMARY")
    print(f"Cases reviewed: {len(results)}")
    print(f"Relevant: {relevant}")
    print(f"Irrelevant: {irrelevant}")
    print(f"Needs review: {review}")
    print(f"Saved results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
