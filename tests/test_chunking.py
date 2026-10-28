
import unittest

from application.ingestion.chunking import (
    build_chunk_records,
    chunk_text,
)


class TestChunking(unittest.TestCase):

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text(""), [])

    def test_short_text_returns_one_chunk(self):
        text = "Remote work requires manager approval."
        self.assertEqual(chunk_text(text), [text])

    def test_chunk_records_have_traceable_ids(self):
        text = "Policy sentence. " * 100

        records = build_chunk_records(
            text=text,
            document_id="doc-123",
            version=1,
            chunk_size=100,
            overlap=20,
        )

        self.assertGreater(len(records), 1)
        self.assertEqual(
            records[0]["chunkId"],
            "doc-123:v1:chunk-0001",
        )
        self.assertEqual(records[0]["documentId"], "doc-123")
        self.assertEqual(records[0]["documentVersion"], 1)
        self.assertEqual(records[0]["chunkIndex"], 1)
        self.assertTrue(records[0]["text"])

    def test_chunk_ids_are_unique(self):
        text = "Policy sentence. " * 100
        records = build_chunk_records(
            text=text,
            document_id="doc-123",
            chunk_size=100,
            overlap=20,
        )

        ids = [record["chunkId"] for record in records]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
