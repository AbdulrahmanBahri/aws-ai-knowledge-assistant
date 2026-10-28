from day13_retrieval import retrieve_chunks


QUESTION = "What approval is required for international remote work?"


def main():
    for top_k in (1, 3, 5):
        print("\n" + "=" * 72)
        print(f"TOP_K = {top_k}")
        print(f"Question: {QUESTION}")
        print("=" * 72)

        matches = retrieve_chunks(QUESTION, top_k=top_k)

        if not matches:
            print("No results returned.")
            continue

        for rank, match in enumerate(matches, start=1):
            print(f"\n--- Rank {rank} ---")
            print(f"Source: {match.get('source')}")
            print(f"Distance: {match.get('distance')}")
            print(f"Chunk ID: {match.get('chunk_id')}")
            print(f"Text:\n{match.get('text')}")

        updated_document_found = any(
            match.get("source") == "day21-lifecycle-test-new.txt"
            and "explicit written approval" in (
                match.get("text") or ""
            ).lower()
            for match in matches
        )

        print(
            "\nUpdated approval evidence found: "
            f"{updated_document_found}"
        )


if __name__ == "__main__":
    main()