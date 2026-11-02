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


class TestDay30QualityEvaluation(unittest.TestCase):

    def test_correct_remote_work_answer(self):
        case = evaluator.TEST_CASES[0]

        answer = (
            "Employees may work remotely up to two days per week, "
            "subject to approval by their manager."
        )

        result = evaluator.evaluate_answer_quality(case, answer)

        self.assertTrue(result["quality_checks_passed"])
        self.assertEqual(result["classification"], "ANSWERED")

    def test_remote_work_answer_missing_manager(self):
        case = evaluator.TEST_CASES[0]

        answer = "Employees may work remotely up to two days per week."

        result = evaluator.evaluate_answer_quality(case, answer)

        self.assertFalse(result["quality_checks_passed"])
        self.assertIn(
            "manager",
            result["missing_required_all"],
        )

    def test_unsupported_remote_work_claim(self):
        case = evaluator.TEST_CASES[0]

        answer = "Employees have unlimited remote work."

        result = evaluator.evaluate_answer_quality(case, answer)

        self.assertFalse(result["quality_checks_passed"])
        self.assertTrue(
            result["matched_forbidden_phrases"]
        )

    def test_correct_international_approval_answer(self):
        case = evaluator.TEST_CASES[5]

        answer = (
            "International remote work requires explicit written "
            "approval from the HR department."
        )

        result = evaluator.evaluate_answer_quality(case, answer)

        self.assertTrue(result["quality_checks_passed"])

    def test_refusal_for_unknown_family_coverage(self):
        case = evaluator.TEST_CASES[3]

        answer = (
            "I couldn't find enough information in the available "
            "documents to answer that confidently."
        )

        result = evaluator.evaluate_answer_quality(case, answer)

        self.assertTrue(result["quality_checks_passed"])
        self.assertEqual(result["classification"], "REFUSED")

    def test_refusal_followed_by_unsupported_claim_fails(self):
      case = evaluator.TEST_CASES[3]

      answer = (
          "I couldn't find enough information in the available "
          "documents to answer that confidently. "
          "However, dependents are definitely covered."
      )

      result = evaluator.evaluate_answer_quality(case, answer)

      self.assertFalse(result["quality_checks_passed"])
      self.assertTrue(
          result["matched_forbidden_phrases"]
      )

    def test_partial_refusal_is_not_a_clean_answer(self):
        case = evaluator.TEST_CASES[5]

        answer = (
            "International remote work requires written approval "
            "from HR. I couldn't find enough information to answer "
            "that confidently."
        )

        result = evaluator.evaluate_answer_quality(case, answer)

        self.assertFalse(result["quality_checks_passed"])
        self.assertEqual(
            result["classification"],
            "PARTIAL_REFUSAL",
        )


if __name__ == "__main__":
    unittest.main()