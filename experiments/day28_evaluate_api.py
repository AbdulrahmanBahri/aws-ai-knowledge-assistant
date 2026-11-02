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
    / "day29_api_evaluation_results.json"
)

TIMEOUT_SECONDS = 90
REQUEST_DELAY_SECONDS = 1


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
# Response classification
# --------------------------------------------------

def classify_response(answer: str) -> str:
    """
    Classify a response based on common refusal phrases.

    ANSWERED:
        No recognized refusal phrase was found.

    REFUSED:
        A refusal was found without an obvious policy-information marker.

    PARTIAL_REFUSAL:
        A refusal was found alongside a marker suggesting the answer
        also provides some policy information.

    This is a heuristic, not a semantic correctness evaluation.
    """

    text = answer.lower()

    refusal_phrases = (
        "couldn't find enough information",
        "could not find enough information",
        "don't have enough information",
        "do not have enough information",
        "insufficient information",
        "cannot determine from the provided",
        "can't determine from the provided",
        "not enough information in the provided",
        "unable to answer confidently",
        "cannot answer confidently",
        "can't answer confidently",
        "the available documents do not",
        "the documents do not provide",
        "not mentioned in the available documents",
        "not specified in the available documents",
    )

    contains_refusal = any(
        phrase in text for phrase in refusal_phrases
    )

    if not contains_refusal:
        return "ANSWERED"

    information_markers = (
        "requires",
        "must",
        "should",
        "policy states",
        "policy requires",
        "approval",
        "according to the policy",
        "the policy specifies",
        "the document states",
    )

    contains_information = any(
        marker in text for marker in information_markers
    )

    if contains_information:
        return "PARTIAL_REFUSAL"

    return "REFUSED"


# --------------------------------------------------
# RAG API evaluator
# --------------------------------------------------

class RAGEvaluator:
    """Evaluate the deployed RAG API against a fixed test dataset."""

    def __init__(
        self,
        endpoint: str = API_ENDPOINT,
        output_path: Path = OUTPUT_PATH,
        timeout_seconds: int = TIMEOUT_SECONDS,
        request_delay_seconds: int = REQUEST_DELAY_SECONDS,
    ):
        self.endpoint = endpoint
        self.output_path = output_path
        self.timeout_seconds = timeout_seconds
        self.request_delay_seconds = request_delay_seconds

    # --------------------------------------------------
    # API client
    # --------------------------------------------------

    def call_api(self, question: str) -> dict:
        """Send one question to the deployed API."""

        payload = json.dumps(
            {"question": question}
        ).encode("utf-8")

        request = Request(
            self.endpoint,
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
                timeout=self.timeout_seconds,
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
    # Single-case evaluation
    # --------------------------------------------------

    def evaluate_case(self, case: dict) -> dict:
        """Evaluate one end-to-end API response."""

        result = {
            "id": case["id"],
            "question": case["question"],
            "expected_answerable": case["expected_answerable"],
            "expected_source": case["expected_source"],
            "request_succeeded": False,
            "response_valid": False,
            "expected_source_found": None,
            "response_classification": None,
            "refusal_behavior_correct": None,
            "answer_correctness": "MANUAL_REVIEW_REQUIRED",
        }

        try:
            api_result = self.call_api(case["question"])
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
            result["error"] = (
                "API response body is not a JSON object."
            )
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
                "Unexpected response structure, empty answer, "
                "or non-successful HTTP status."
            )
            return result

        # Check whether the expected document was retrieved.
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

        # Classify the answer's refusal behavior.
        classification = classify_response(answer)

        result["response_classification"] = classification

        if case["expected_answerable"]:
            result["refusal_behavior_correct"] = (
                classification == "ANSWERED"
            )
        else:
            result["refusal_behavior_correct"] = (
                classification == "REFUSED"
            )

        return result

    # --------------------------------------------------
    # Run complete evaluation
    # --------------------------------------------------

    def run(self, test_cases: list = None) -> dict:
        """Run the dataset and return a complete evaluation summary."""

        cases = TEST_CASES if test_cases is None else test_cases
        results = []

        print("=" * 72)
        print("DAY 29 — RAG API EVALUATION")
        print("=" * 72)
        print(f"Endpoint: {self.endpoint}")
        print(f"Test cases: {len(cases)}")
        print()

        for number, case in enumerate(cases, start=1):
            print(f"[{number}/{len(cases)}] {case['id']}")
            print(f"Question: {case['question']}")

            result = self.evaluate_case(case)
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
                f"Response classification: "
                f"{result['response_classification']}"
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

            if number < len(cases):
                time.sleep(self.request_delay_seconds)

        # Aggregate results.
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

        classification_counts = {
            classification: sum(
                result["response_classification"] == classification
                for result in completed
            )
            for classification in (
                "ANSWERED",
                "REFUSED",
                "PARTIAL_REFUSAL",
            )
        }

        summary = {
            "evaluated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "endpoint": self.endpoint,
            "total_cases": len(cases),
            "valid_responses": len(completed),
            "successful_http_requests": sum(
                result["request_succeeded"]
                for result in results
            ),
            "expected_source_matches": source_matches,
            "expected_source_checks": len(source_checks),
            "correct_refusal_behavior": correct_refusal_behavior,
            "refusal_behavior_checks": len(refusal_checks),
            "response_classification_counts": classification_counts,
            "answer_correctness": (
                "Requires manual review against source documents."
            ),
            "results": results,
        }

        # Save JSON report.
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

        # Print summary.
        print()
        print("=" * 72)
        print("EVALUATION SUMMARY")
        print("=" * 72)
        print(f"Total test cases: {len(cases)}")
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
        print(f"Classification counts: {classification_counts}")
        print("Answer correctness: requires manual review.")
        print(f"Results saved to: {self.output_path}")

        return summary


# --------------------------------------------------
# Entry point
# --------------------------------------------------

def main():
    evaluator = RAGEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()