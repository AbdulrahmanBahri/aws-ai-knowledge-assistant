
from day13_retrieval import retrieve_chunks
from day14_generation import generate_answer

TOP_K = 3


def answer_question(question: str) -> dict:
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    # Step 1: Retrieve relevant document chunks.
    matches = retrieve_chunks(question, top_k=TOP_K)

    # Step 2: Handle an empty retrieval response.
    if not matches:
        return {
            "question": question,
            "answer": (
                "I couldn't find any document chunks to "
                "answer your question."
            ),
            "sources": [],
        }

    # Step 3: Build evidence for the generation model.
    evidence_sections = []

    for match in matches:
        text = match.get("text")
        if not text or not text.strip():
            continue

        source = match.get("source") or "Unknown source"

        evidence_sections.append(
            f"Source: {source}\n"
            f"Document version: "
            f"{match.get('document_version', 'Unknown')}\n"
            f"Evidence:\n{text}"
        )

    if not evidence_sections:
        return {
            "question": question,
            "answer": (
                "The retrieved results did not contain "
                "usable text evidence."
            ),
            "sources": [],
        }

    evidence = "\n\n---\n\n".join(evidence_sections)

    # Step 4: Generate an answer grounded in the evidence.
    answer = generate_answer(question, evidence)

    # Step 5: Preserve source attribution.
    sources = list(dict.fromkeys(
        match.get("source")
        for match in matches
        if match.get("text")
        and match.get("source")
    ))

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    question = (
        "Who must approve international remote work, according to the updated document?"
    )

    result = answer_question(question)

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    if result["sources"]:
        for source in result["sources"]:
            print(f"- {source}")
    else:
        print("- No sources available")
