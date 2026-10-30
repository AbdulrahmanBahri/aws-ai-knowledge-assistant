import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_ENDPOINT = (
    "https://5qk1tknmo0.execute-api.eu-central-1.amazonaws.com/ask"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "tests"
    / "evaluation"
    / "day28_api_evaluation_results.json"
)

TIMEOUT_SECONDS = 90

INSUFFICIENT_EVIDENCE_PHRASE = (
    "I couldn't find enough information in the available "
    "documents to answer that confidently."
)


# --------------------------------------------------
# Evaluation dataset
# --------------------------------------------------

TEST_CASES = [
    {
        "id": "remote_work",
        "question": "How many days per week can staff work remotely?",
        "expected_source": "remote_work_policy.txt",
        "expected_answerable": True,
    },
    {
        "id": "health_insurance",
        "question": "Does the company cover employees' health insurance?",
        "expected_source": "employee_benefits_policy.txt",
        "expected_answerable": True,
    },
    {
        "id": "annual_leave",
        "question": (
            "Who should employees submit annual leave requests through?"
        ),
        "expected_source": "annual_leave_policy.txt",
        "expected_answerable": True,
    },
    {
        "id": "family_coverage",
        "question": (
            "Are employees' dependents covered by health insurance?"
        ),
        "expected_source": None,
        "expected_answerable": False,
    },
    {
        "id": "leave_carryover",
        "question": "Can unused annual leave be carried over?",
        "expected_source": None,
        "expected_answerable": False,
    },
    {
        "id": "international_remote_work",
        "question": "Can I work remotely from another country?",
        "expected_source": "international_remote_work_policy.txt",
        "expected_answerable": True,
    },
    {
        "id": "international_remote_work_approval",
        "question": (
            "What approval is required for international remote work?"
        ),
        "expected_source": "international_remote_work_policy.txt",
        "expected_answerable": True,
    },
]


# --------------------------------------------------
# API client
# --------------------------------------------------

def call_api(question: str) -> dict:
    """Send one question to the deployed API."""

    payload = json.dumps(
        {"question": question}
    ).encode("utf-8")

    request = Request(
        API_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=TIMEOUT_SECONDS,
        ) as response:
            status_code = response.status
            response_body = response.read().decode("utf-8")

    except HTTPError as exc:
        status_code = exc.code
        response_body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"API request failed: {exc}"
        ) from exc

    try:
        body = json.loads(response_body)
    except json.JSONDecodeError:
        body = {"raw_response": response_body}

    return {
        "http_status": status_code,
        "body": body,
    }


# --------------------------------------------------
# Evaluation logic
# --------------------------------------------------

def evaluate_case(case: dict) -> dict:
    """Evaluate one end-to-end API response."""

    question = case["question"]

    result = {
        "id": case["id"],
        "question": question,
        "expected_answerable": case["expected_answerable"],
        "expected_source": case["expected_source"],
        "request_succeeded": False,
        "response_valid": False,
        "expected_source_found": None,
        "refusal_behavior_correct": None,
        "answer_correctness": "MANUAL_REVIEW_REQUIRED",
    }

    try:
        api_result = call_api(question)

    except Exception as exc:
        result["error"] = str(exc)
        return result

    result["http_status"] = api_result["http_status"]
    result["response_body"] = api_result["body"]

    body = api_result["body"]

    result["request_succeeded"] = (
        api_result["http_status"] == 200
    )

    if not isinstance(body, dict):
        result["error"] = "API response body is not a JSON object."
        return result

    answer = body.get("answer")
    sources = body.get("sources")

    result["answer"] = answer
    result["sources"] = sources
    result["grounded"] = body.get("grounded")

    result["response_valid"] = (
        result["request_succeeded"]
        and isinstance(answer, str)
        and bool(answer.strip())
        and isinstance(sources, list)
        and isinstance(body.get("grounded"), bool)
    )

    if not result["response_valid"]:
        result["error"] = (
            "Unexpected response structure, empty answer, "
            "or non-successful HTTP status."
        )
        return result

    returned_documents = [
        source.get("document", "").lower()
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("document"), str)
    ]

    expected_source = case["expected_source"]

    if expected_source is not None:
        result["expected_source_found"] = (
            expected_source.lower() in returned_documents
        )

    is_refusal = (
        INSUFFICIENT_EVIDENCE_PHRASE.lower()
        in answer.lower()
    )

    if case["expected_answerable"]:
        result["refusal_behavior_correct"] = not is_refusal
    else:
        result["refusal_behavior_correct"] = is_refusal

    return result


# --------------------------------------------------
# Reporting
# --------------------------------------------------

def main():
    results = []

    print("=" * 72)
    print("DAY 28 — END-TO-END API EVALUATION")
    print("=" * 72)
    print(f"Endpoint: {API_ENDPOINT}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    for number, case in enumerate(TEST_CASES, start=1):
        print(
            f"[{number}/{len(TEST_CASES)}] "
            f"{case['id']}"
        )
        print(f"Question: {case['question']}")

        result = evaluate_case(case)
        results.append(result)

        print(
            f"HTTP request succeeded: "
            f"{result['request_succeeded']}"
        )
        print(
            f"Response structure valid: "
            f"{result['response_valid']}"
        )
        print(
            f"Expected source found: "
            f"{result['expected_source_found']}"
        )
        print(
            f"Refusal behavior correct: "
            f"{result['refusal_behavior_correct']}"
        )

        if result.get("error"):
            print(f"Error: {result['error']}")

        if result.get("answer"):
            print(f"Answer: {result['answer']}")

        print("-" * 72)

        # Avoid sending requests in a rapid burst.
        if number < len(TEST_CASES):
            time.sleep(1)

    completed = [
        result
        for result in results
        if result["response_valid"]
    ]

    source_checks = [
        result
        for result in completed
        if result["expected_source"] is not None
    ]

    source_matches = sum(
        result["expected_source_found"] is True
        for result in source_checks
    )

    refusal_checks = [
        result
        for result in completed
        if result["refusal_behavior_correct"] is not None
    ]

    correct_refusal_behavior = sum(
        result["refusal_behavior_correct"] is True
        for result in refusal_checks
    )

    summary = {
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "endpoint": API_ENDPOINT,
        "total_cases": len(TEST_CASES),
        "valid_responses": len(completed),
        "successful_http_requests": sum(
            result["request_succeeded"]
            for result in results
        ),
        "expected_source_matches": source_matches,
        "expected_source_checks": len(source_checks),
        "correct_refusal_behavior": correct_refusal_behavior,
        "refusal_behavior_checks": len(refusal_checks),
        "answer_correctness": (
            "Requires manual review against source documents."
        ),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Valid API responses: {len(completed)}")
    print(
        f"Successful HTTP requests: "
        f"{summary['successful_http_requests']}"
    )
    print(
        f"Expected sources found: "
        f"{source_matches}/{len(source_checks)}"
    )
    print(
        f"Correct refusal behavior: "
        f"{correct_refusal_behavior}/{len(refusal_checks)}"
    )
    print("Answer correctness: requires manual review.")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()