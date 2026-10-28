
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "tests"
    / "evaluation"
    / "day17_results.json"
)

# Experimental values, not production recommendations.
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]


def evaluate_threshold(results: list[dict], threshold: float) -> dict:
    true_accepts = 0
    false_rejections = 0
    false_acceptances = 0
    true_rejections = 0

    for case in results:
        evidence = case.get("evidence", [])

        # The original retrieval returned the nearest chunk first.
        distance = (
            evidence[0].get("distance")
            if evidence
            else None
        )

        judgment = case["manual_judgment"]

        if judgment == "REVIEW":
           continue

        actually_relevant = judgment == "RELEVANT"

        # No evidence, or distance above the threshold: reject.
        accepted = (
            distance is not None
            and distance <= threshold
        )

        if actually_relevant and accepted:
            true_accepts += 1
        elif actually_relevant and not accepted:
            false_rejections += 1
        elif not actually_relevant and accepted:
            false_acceptances += 1
        else:
            true_rejections += 1

    return {
        "threshold": threshold,
        "relevant_accepted": true_accepts,
        "relevant_rejected": false_rejections,
        "irrelevant_accepted": false_acceptances,
        "irrelevant_rejected": true_rejections,
    }


def main():
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    print("DAY 18: DISTANCE THRESHOLD EXPERIMENT")
    print(f"Evaluation cases: {len(results)}")
    print("Accept a result when distance <= threshold.\n")

    for threshold in THRESHOLDS:
        metrics = evaluate_threshold(results, threshold)

        print(f"--- Threshold: {threshold:.2f} ---")
        print(
            "Relevant evidence accepted: "
            f"{metrics['relevant_accepted']}"
        )
        print(
            "False rejections (relevant evidence rejected): "
            f"{metrics['relevant_rejected']}"
        )
        print(
            "False acceptances (irrelevant evidence accepted): "
            f"{metrics['irrelevant_accepted']}"
        )
        print(
            "Irrelevant evidence rejected: "
            f"{metrics['irrelevant_rejected']}"
        )
        print()


if __name__ == "__main__":
    main()
