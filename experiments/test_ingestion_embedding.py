
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "application" / "ingestion"),
)

from embeddings import generate_embedding


def main():
    text = (
        "Employees may work remotely up to two days per week "
        "with manager approval."
    )

    vector = generate_embedding(text)

    print(f"Dimensions: {len(vector)}")
    print(f"First five values: {vector[:5]}")
    print(f"All values finite: {all(map(is_finite, vector))}")


def is_finite(value):
    import math
    return math.isfinite(value)


if __name__ == "__main__":
    main()
