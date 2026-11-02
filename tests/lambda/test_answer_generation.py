import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAMBDA_PACKAGE = PROJECT_ROOT / "lambda_package"

sys.path.insert(0, str(LAMBDA_PACKAGE))

# Set required environment variables before importing query_handler.
os.environ.setdefault("VECTOR_BUCKET_NAME", "test-vector-bucket")
os.environ.setdefault("VECTOR_INDEX_NAME", "test-index")
os.environ.setdefault("AWS_REGION", "eu-central-1")

import query_handler


class TestAnswerGeneration(unittest.TestCase):

    def setUp(self):
        self.question = "How many days can employees work remotely?"

        self.chunks = [
            {
                "chunk_id": "chunk-1",
                "distance": 0.1,
                "text": (
                    "Employees may work remotely up to two days per week "
                    "with manager approval."
                ),
                "source": "remote_work_policy.txt",
                "document_id": "doc-1",
                "chunk_index": 0,
            }
        ]

    @patch.object(query_handler.bedrock, "converse")
    def test_generates_answer_from_bedrock_response(self, mock_converse):
        mock_converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": (
                                "Employees may work remotely up to "
                                "two days per week with manager approval."
                            )
                        }
                    ]
                }
            }
        }

        answer = query_handler.generate_answer(
            self.question,
            self.chunks,
        )

        self.assertIn("two days per week", answer)
        self.assertIn("manager approval", answer)
        mock_converse.assert_called_once()

    @patch.object(query_handler.bedrock, "converse")
    def test_prompt_contains_question_and_retrieved_evidence(
        self,
        mock_converse,
    ):
        mock_converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Two days per week."}]
                }
            }
        }

        query_handler.generate_answer(self.question, self.chunks)

        kwargs = mock_converse.call_args.kwargs

        system_prompt = " ".join(
            item["text"] for item in kwargs["system"]
        )

        user_prompt = " ".join(
            item["text"]
            for message in kwargs["messages"]
            for item in message["content"]
        )

        self.assertIn(self.question, user_prompt)
        self.assertIn(
            "Employees may work remotely up to two days per week",
            user_prompt,
        )
        self.assertIn("remote_work_policy.txt", user_prompt)
        self.assertIn(
            "using only the supplied document excerpts",
            system_prompt,
        )

    @patch.object(query_handler.bedrock, "converse")
    def test_uses_expected_model_and_low_temperature(self, mock_converse):
        mock_converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "Two days per week."}]
                }
            }
        }

        query_handler.generate_answer(self.question, self.chunks)

        kwargs = mock_converse.call_args.kwargs

        self.assertEqual(kwargs["modelId"], query_handler.MODEL_ID)
        self.assertEqual(kwargs["inferenceConfig"]["temperature"], 0.1)
        self.assertEqual(kwargs["inferenceConfig"]["maxTokens"], 400)

    @patch.object(query_handler.bedrock, "converse")
    def test_combines_multiple_retrieved_chunks(self, mock_converse):
        mock_converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "See the policy requirements."}]
                }
            }
        }

        second_chunk = {
            "chunk_id": "chunk-2",
            "distance": 0.2,
            "text": "Employees must obtain manager approval.",
            "source": "approval_policy.txt",
            "document_id": "doc-2",
            "chunk_index": 0,
        }

        query_handler.generate_answer(
            self.question,
            self.chunks + [second_chunk],
        )

        user_prompt = " ".join(
            item["text"]
            for message in mock_converse.call_args.kwargs["messages"]
            for item in message["content"]
        )

        self.assertIn("remote_work_policy.txt", user_prompt)
        self.assertIn("approval_policy.txt", user_prompt)
        self.assertIn(
            "Employees must obtain manager approval",
            user_prompt,
        )

    @patch.object(query_handler.bedrock, "converse")
    def test_raises_error_when_model_returns_no_text(self, mock_converse):
        mock_converse.return_value = {
            "output": {
                "message": {
                    "content": [{"type": "toolUse"}]
                }
            }
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "The model returned no text answer",
        ):
            query_handler.generate_answer(self.question, self.chunks)

    @patch.object(query_handler.bedrock, "converse")
    def test_returns_trimmed_answer(self, mock_converse):
        mock_converse.return_value = {
            "output": {
                "message": {
                    "content": [{"text": "  Two days per week.  "}]
                }
            }
        }

        answer = query_handler.generate_answer(
            self.question,
            self.chunks,
        )

        self.assertEqual(answer, "Two days per week.")


if __name__ == "__main__":
    unittest.main()