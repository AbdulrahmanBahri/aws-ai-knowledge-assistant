import json
import re
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
    / "day30_quality_evaluation_results.json"
)

TIMEOUT_SECONDS = 90
REQUEST_DELAY_SECONDS = 1

REFUSAL_PHRASES = (
    "couldn't find enough information",
    "could not find enough information",
    "don't have enough information",
    "do not have enough information",
    "insufficient information",
    "unable to answer confidently",
    "cannot answer confidently",
    "can't answer confidently",
    "not enough information in the available documents",
)


# --------------------------------------------------
# Test dataset
# --------------------------------------------------

TEST_CASES = [
    {
        "id": "remote_work",
        "question": "How many days per week can staff work remotely?",
        "expected_source": "remote_work_policy.txt",
        "expected_answerable": True,
        "required_any": [
            "two days",
            "2 days",
        ],
        "required_all": [
            "manager",
        ],
        "forbidden": [
            "three days per week",
            "unlimited remote work",
        ],
    },
    {
        "id": "health_insurance",
        "question": "Does the company cover employees' health insurance?",
        "expected_source": "employee_benefits_policy.txt",
        "expected_answerable": True,
        "required_any": [
            "covers employees",
            "cover employees",
            "employees' health insurance",
            "employee health insurance",
        ],
        "required_all": [],
        "forbidden": [
            "dependents are definitely covered",
            "all family members are covered",
        ],
    },
    {
        "id": "annual_leave",
        "question": (
            "Who should employees submit annual leave requests through?"
        ),
        "expected_source": "annual_leave_policy.txt",
        "expected_answerable": True,
        "required_any": [
            "manager",
        ],
        "required_all": [
            "leave",
        ],
        "forbidden": [
            "submit directly to the ceo",
        ],
    },
    {
        "id": "family_coverage",
        "question": (
            "Are employees' dependents covered by health insurance?"
        ),
        "expected_source": None,
        "expected_answerable": False,
        "required_any": [
            "couldn't find enough information",
            "could not find enough information",
            "insufficient information",
            "not specify if family members",
            "does not specify whether family members",
            "does not specify if family members",
            "does not state whether dependents",
            "not mention whether dependents",
        ],
        "required_all": [],
        "forbidden": [
            "dependents are definitely covered",
            "all family members are covered",
            "dependents receive full insurance coverage",
        ],
    },
    {
        "id": "leave_carryover",
        "question": "Can unused annual leave be carried over?",
        "expected_source": None,
        "expected_answerable": False,
        "required_any": [
            "couldn't find enough information",
            "could not find enough information",
            "insufficient information",
            "cannot determine",
            "not specified",
            "not mentioned",
        ],
        "required_all": [],
        "forbidden": [
            "unused leave can be carried over",
            "unused leave is automatically carried over",
            "unused leave always carries over",
        ],
    },
    {
        "id": "international_remote_work",
        "question": "Can I work remotely from another country?",
        "expected_source": "international_remote_work_policy.txt",
        "expected_answerable": True,
        "required_any": [
            "written approval",
            "written permission",
        ],
        "required_all": [
            "hr",
        ],
        "forbidden": [
            "no approval is required",
            "approval is unnecessary",
            "you can work from any country without approval",
        ],
    },
    {
        "id": "international_remote_work_approval",
        "question": (
            "What approval is required for international remote work?"
        ),
        "expected_source": "international_remote_work_policy.txt",
        "expected_answerable": True,
        "required_any": [
            "written approval",
            "written permission",
        ],
        "required_all": [
            "hr",
        ],
        "forbidden": [
            "no approval is required",
            "approval is unnecessary",
        ],
    },
]


# --------------------------------------------------
# Text helpers
# --------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize text for simple, case-insensitive phrase matching."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_response(answer: str) -> str:
    """Classify a response using common refusal phrases."""
    normalized = normalize_text(answer)

    has_refusal = any(
        phrase in normalized
        for phrase in REFUSAL_PHRASES
    )

    if not has_refusal:
        return "ANSWERED"

    information_markers = (
        "requires",
        "must",
        "should",
        "approval",
        "according to the policy",
        "the policy states",
    )

    has_information = any(
        marker in normalized
        for marker in information_markers
    )

    if has_information:
        return "PARTIAL_REFUSAL"

    return "REFUSED"


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
# Answer quality checks
# --------------------------------------------------

def evaluate_answer_quality(case: dict, answer: str) -> dict:
    """
    Apply deterministic phrase checks to the generated answer.

    These checks are useful regression signals, not proof of
    factual correctness or a replacement for human review.
    """

    normalized = normalize_text(answer)

    required_any = case.get("required_any", [])
    required_all = case.get("required_all", [])
    forbidden = case.get("forbidden", [])

    matched_required_any = [
        phrase
        for phrase in required_any
        if normalize_text(phrase) in normalized
    ]

    missing_required_all = [
        phrase
        for phrase in required_all
        if normalize_text(phrase) not in normalized
    ]

    matched_forbidden = [
        phrase
        for phrase in forbidden
        if normalize_text(phrase) in normalized
    ]

    checks = {
        "required_any_passed": (
            not required_any or bool(matched_required_any)
        ),
        "required_all_passed": (
            not missing_required_all
        ),
        "no_forbidden_phrases": (
            not matched_forbidden
        ),
    }

    classification = classify_response(answer)

    if case["expected_answerable"]:
        checks["answerability_behavior_passed"] = (
            classification == "ANSWERED"
        )
    else:
        checks["answerability_behavior_passed"] = (
            classification == "REFUSED"
        )

    return {
        "classification": classification,
        "checks": checks,
        "quality_checks_passed": all(checks.values()),
        "matched_required_any": matched_required_any,
        "missing_required_all": missing_required_all,
        "matched_forbidden_phrases": matched_forbidden,
        "manual_review_required": True,
    }


# --------------------------------------------------
# Evaluate one test case
# --------------------------------------------------

def evaluate_case(case: dict) -> dict:
    """Call the API and evaluate its response."""

    result = {
        "id": case["id"],
        "question": case["question"],
        "expected_answerable": case["expected_answerable"],
        "expected_source": case["expected_source"],
        "request_succeeded": False,
        "response_valid": False,
        "expected_source_found": None,
        "quality_checks_passed": False,
        "manual_review_required": True,
    }

    try:
        api_result = call_api(case["question"])

    except Exception as exc:
        result["error"] = str(exc)
        return result

    body = api_result["body"]

    result["http_status"] = api_result["http_status"]
    result["response_body"] = body

    result["request_succeeded"] = (
        api_result["http_status"] == 200
    )

    if not isinstance(body, dict):
        result["error"] = "API response body is not a JSON object."
        return result

    answer = body.get("answer")
    sources = body.get("sources")
    grounded = body.get("grounded")

    result["answer"] = answer
    result["sources"] = sources
    result["grounded"] = grounded

    result["response_valid"] = (
        result["request_succeeded"]
        and isinstance(answer, str)
        and bool(answer.strip())
        and isinstance(sources, list)
        and isinstance(grounded, bool)
    )

    if not result["response_valid"]:
        result["error"] = (
            "Invalid response structure, empty answer, "
            "or unsuccessful HTTP status."
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

    quality = evaluate_answer_quality(case, answer)

    result["response_classification"] = quality["classification"]
    result["quality_checks"] = quality["checks"]
    result["quality_checks_passed"] = quality["quality_checks_passed"]
    result["matched_required_any"] = quality["matched_required_any"]
    result["missing_required_all"] = quality["missing_required_all"]
    result["matched_forbidden_phrases"] = (
        quality["matched_forbidden_phrases"]
    )

    # Source retrieval is tracked separately from answer quality.
    if expected_source is not None:
        result["expected_source_check_passed"] = (
            result["expected_source_found"] is True
        )
    else:
        result["expected_source_check_passed"] = True

    result["evaluation_passed"] = (
        result["response_valid"]
        and result["expected_source_check_passed"]
        and result["quality_checks_passed"]
    )

    return result


# --------------------------------------------------
# Main evaluation and report
# --------------------------------------------------

def main():
    results = []

    print("=" * 72)
    print("DAY 30 — RAG ANSWER QUALITY EVALUATION")
    print("=" * 72)
    print(f"Endpoint: {API_ENDPOINT}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    for number, case in enumerate(TEST_CASES, start=1):
        print(f"[{number}/{len(TEST_CASES)}] {case['id']}")
        print(f"Question: {case['question']}")

        result = evaluate_case(case)
        results.append(result)

        print(f"HTTP request succeeded: {result['request_succeeded']}")
        print(f"Response structure valid: {result['response_valid']}")
        print(f"Expected source found: {result['expected_source_found']}")
        print(
            f"Response classification: "
            f"{result.get('response_classification', 'NOT_CLASSIFIED')}"
        )
        print(
            f"Quality checks passed: "
            f"{result['quality_checks_passed']}"
        )
        print(
            f"Overall evaluation passed: "
            f"{result.get('evaluation_passed', False)}"
        )

        if result.get("missing_required_all"):
            print(
                "Missing required terms: "
                f"{result['missing_required_all']}"
            )

        if result.get("matched_forbidden_phrases"):
            print(
                "Forbidden phrases found: "
                f"{result['matched_forbidden_phrases']}"
            )

        if result.get("error"):
            print(f"Error: {result['error']}")

        if result.get("answer"):
            print(f"Answer: {result['answer']}")

        print("-" * 72)

        if number < len(TEST_CASES):
            time.sleep(REQUEST_DELAY_SECONDS)

    successful_requests = sum(
        result["request_succeeded"]
        for result in results
    )

    valid_responses = sum(
        result["response_valid"]
        for result in results
    )

    source_checks = [
        result
        for result in results
        if result["expected_source"] is not None
        and result["response_valid"]
    ]

    source_matches = sum(
        result["expected_source_found"] is True
        for result in source_checks
    )

    quality_checks = [
        result
        for result in results
        if result["response_valid"]
    ]

    quality_passes = sum(
        result["quality_checks_passed"]
        for result in quality_checks
    )

    overall_passes = sum(
        result.get("evaluation_passed", False)
        for result in results
    )

    summary = {
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "endpoint": API_ENDPOINT,
        "total_cases": len(TEST_CASES),
        "successful_http_requests": successful_requests,
        "valid_responses": valid_responses,
        "expected_source_matches": source_matches,
        "expected_source_checks": len(source_checks),
        "quality_checks_passed": quality_passes,
        "quality_checks_total": len(quality_checks),
        "overall_evaluation_passed": overall_passes,
        "overall_evaluation_total": len(TEST_CASES),
        "evaluation_method": (
            "Deterministic phrase checks; manual factual review "
            "is still required."
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
    print("DAY 30 SUMMARY")
    print("=" * 72)
    print(f"Successful HTTP requests: {successful_requests}/7")
    print(f"Valid API responses: {valid_responses}/7")
    print(
        f"Expected source matches: "
        f"{source_matches}/{len(source_checks)}"
    )
    print(
        f"Answer quality checks passed: "
        f"{quality_passes}/{len(quality_checks)}"
    )
    print(
        f"Overall evaluation passed: "
        f"{overall_passes}/{len(TEST_CASES)}"
    )
    print("Manual factual review is still required.")
    print(f"Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()