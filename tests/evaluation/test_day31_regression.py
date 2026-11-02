import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATOR_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "day30_evaluate_quality.py"
)

spec = importlib.util.spec_from_file_location(
    "day30_evaluate_quality",
    EVALUATOR_PATH,
)

evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class TestRAGRegression(unittest.TestCase):

    def evaluate(self, case_id, answer):
        case = next(
            case
            for case in evaluator.TEST_CASES
            if case["id"] == case_id
        )

        return evaluator.evaluate_answer_quality(case, answer)

    def test_remote_work_correct_answer(self):
        answer = (
            "Employees may work remotely up to two days per week "
            "with manager approval."
        )

        result = self.evaluate("remote_work", answer)

        self.assertTrue(result["quality_checks_passed"])
        self.assertEqual(result["classification"], "ANSWERED")

    def test_remote_work_missing_approval_context(self):
        answer = (
            "Employees may work remotely up to two days per week."
        )

        result = self.evaluate("remote_work", answer)

        self.assertFalse(result["quality_checks_passed"])
        self.assertIn("manager", result["missing_required_all"])

    def test_remote_work_unsupported_claim(self):
        answer = "Employees have unlimited remote work."

        result = self.evaluate("remote_work", answer)

        self.assertFalse(result["quality_checks_passed"])
        self.assertTrue(result["matched_forbidden_phrases"])

    def test_employee_health_insurance_answer(self):
        answer = (
            "The company covers employees' health insurance. "
            "The policy does not specify dependent coverage."
        )

        result = self.evaluate("health_insurance", answer)

        self.assertTrue(result["quality_checks_passed"])

    def test_annual_leave_answer_mentions_manager(self):
        answer = (
            "Employees must submit annual leave requests "
            "through their manager."
        )

        result = self.evaluate("annual_leave", answer)

        self.assertTrue(result["quality_checks_passed"])

    def test_unknown_dependent_coverage_is_refused(self):
        answer = (
            "I couldn't find enough information in the available "
            "documents to answer whether employees' dependents "
            "are covered. The policy does not specify whether "
            "family members are included."
        )

        result = self.evaluate("family_coverage", answer)

        self.assertTrue(result["quality_checks_passed"])
        self.assertEqual(result["classification"], "REFUSED")

    def test_dependent_coverage_unsupported_claim_fails(self):
        answer = (
            "I couldn't find enough information in the available "
            "documents. However, dependents are definitely covered."
        )

        result = self.evaluate("family_coverage", answer)

        self.assertFalse(result["quality_checks_passed"])
        self.assertTrue(result["matched_forbidden_phrases"])

    def test_unknown_leave_carryover_is_refused(self):
        answer = (
            "I couldn't find enough information in the available "
            "documents to determine whether unused annual leave "
            "can be carried over."
        )

        result = self.evaluate("leave_carryover", answer)

        self.assertTrue(result["quality_checks_passed"])
        self.assertEqual(result["classification"], "REFUSED")

    def test_international_remote_work_requires_hr_approval(self):
        answer = (
            "International remote work requires explicit written "
            "approval from the HR department."
        )

        result = self.evaluate("international_remote_work", answer)

        self.assertTrue(result["quality_checks_passed"])
        self.assertEqual(result["classification"], "ANSWERED")

    def test_international_approval_answer_is_not_a_refusal(self):
        answer = (
            "International remote work requires explicit written "
            "approval from HR. I couldn't find enough information "
            "to answer that confidently."
        )

        result = self.evaluate(
            "international_remote_work_approval",
            answer,
        )

        self.assertFalse(result["quality_checks_passed"])
        self.assertEqual(
            result["classification"],
            "PARTIAL_REFUSAL",
        )


if __name__ == "__main__":
    unittest.main()