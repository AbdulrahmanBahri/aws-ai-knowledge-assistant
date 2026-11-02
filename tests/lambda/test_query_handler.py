import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from botocore.exceptions import ClientError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAMBDA_PACKAGE = PROJECT_ROOT / "lambda_package"

sys.path.insert(0, str(LAMBDA_PACKAGE))

# The Lambda module reads these environment variables during import.
os.environ.setdefault("VECTOR_BUCKET_NAME", "test-vector-bucket")
os.environ.setdefault("VECTOR_INDEX_NAME", "test-index")
os.environ.setdefault("AWS_REGION", "eu-central-1")

import query_handler


class TestQueryHandler(unittest.TestCase):

    def setUp(self):
        # Reset mocks for each test.
        self.retrieve_patcher = patch.object(
            query_handler,
            "retrieve_chunks",
        )
        self.generate_patcher = patch.object(
            query_handler,
            "generate_answer",
        )

        self.mock_retrieve = self.retrieve_patcher.start()
        self.mock_generate = self.generate_patcher.start()

        self.addCleanup(self.retrieve_patcher.stop)
        self.addCleanup(self.generate_patcher.stop)

    def invoke(self, event):
        """Invoke the Lambda handler and decode its response."""
        result = query_handler.lambda_handler(event, None)

        self.assertIn("statusCode", result)
        self.assertIn("body", result)

        body = json.loads(result["body"])

        return result["statusCode"], body

    def test_rejects_missing_question(self):
        status, body = self.invoke({})

        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.mock_retrieve.assert_not_called()

    def test_rejects_empty_question(self):
        status, body = self.invoke({"question": "   "})

        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.mock_retrieve.assert_not_called()

    def test_rejects_non_string_question(self):
        status, body = self.invoke({"question": 123})

        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.mock_retrieve.assert_not_called()

    def test_rejects_question_over_2000_characters(self):
        status, body = self.invoke({
            "question": "a" * 2001,
        })

        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.mock_retrieve.assert_not_called()

    def test_accepts_question_at_2000_character_limit(self):
        self.mock_retrieve.return_value = []

        status, body = self.invoke({
            "question": "a" * 2000,
        })

        self.assertEqual(status, 200)
        self.assertFalse(body["grounded"])
        self.mock_retrieve.assert_called_once()

    def test_accepts_api_gateway_proxy_event(self):
        self.mock_retrieve.return_value = []

        status, body = self.invoke({
            "body": json.dumps({
                "question": "What is the remote work policy?",
            })
        })

        self.assertEqual(status, 200)
        self.assertIn("answer", body)
        self.assertFalse(body["grounded"])

    def test_rejects_invalid_json_body(self):
        status, body = self.invoke({
            "body": "{invalid-json",
        })

        self.assertEqual(status, 400)
        self.assertIn("error", body)
        self.mock_retrieve.assert_not_called()

    def test_refuses_when_no_chunks_are_retrieved(self):
        self.mock_retrieve.return_value = []

        status, body = self.invoke({
            "question": "What is the remote work policy?",
        })

        self.assertEqual(status, 200)
        self.assertFalse(body["grounded"])
        self.assertEqual(body["sources"], [])
        self.assertIn("enough information", body["answer"])
        self.mock_generate.assert_not_called()

    def test_refuses_when_all_chunks_are_too_distant(self):
        self.mock_retrieve.return_value = [
            {
                "chunk_id": "chunk-1",
                "distance": 0.9,
                "text": "Unrelated document text.",
                "source": "unrelated.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            }
        ]

        status, body = self.invoke({
            "question": "What is the remote work policy?",
        })

        self.assertEqual(status, 200)
        self.assertFalse(body["grounded"])
        self.assertEqual(body["sources"], [])
        self.mock_generate.assert_not_called()

    def test_returns_answer_and_source_for_relevant_evidence(self):
        self.mock_retrieve.return_value = [
            {
                "chunk_id": "chunk-1",
                "distance": 0.1,
                "text": "Employees may work remotely two days per week.",
                "source": "remote_work_policy.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            }
        ]

        self.mock_generate.return_value = (
            "Employees may work remotely two days per week."
        )

        status, body = self.invoke({
            "question": "How many days can employees work remotely?",
        })

        self.assertEqual(status, 200)
        self.assertTrue(body["grounded"])
        self.assertIn("two days per week", body["answer"])
        self.assertEqual(
            body["sources"][0]["document"],
            "remote_work_policy.txt",
        )

    def test_returns_500_when_aws_service_fails(self):
        self.mock_retrieve.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ServiceUnavailableException",
                    "Message": "Simulated service failure",
                }
            },
            "QueryVectors",
        )

        status, body = self.invoke({
            "question": "What is the remote work policy?",
        })

        self.assertEqual(status, 500)
        self.assertIn("error", body)
        self.assertNotIn("Simulated service failure", body["error"])


if __name__ == "__main__":
    unittest.main()