
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INGESTION_DIR = PROJECT_ROOT / "application" / "ingestion"

sys.path.insert(0, str(INGESTION_DIR))

from chunking import build_chunk_records
from embeddings import generate_embedding


def main():
    document_text = """
    Employees may work remotely up to two days per week
    with manager approval.

    Employees receive health insurance through the company.
    """

    chunks = build_chunk_records(
        text=document_text,
        document_id="day12-test-document",
        version=1,
        chunk_size=100,
        overlap=20,
    )

    print(f"Number of chunks: {len(chunks)}")

    for chunk in chunks:
        vector = generate_embedding(chunk["text"])

        print(f"\nChunk ID: {chunk['chunkId']}")
        print(f"Chunk text: {chunk['text']!r}")
        print(f"Vector dimensions: {len(vector)}")

        # Keep the vector in memory for this experiment.
        chunk["embedding"] = vector

    print("\nAll chunks were embedded successfully.")


if __name__ == "__main__":
    main()
