
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping chunks, preferring word boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            # Prefer ending at whitespace, rather than
            # splitting a word in the middle.
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Keep part of the previous chunk as context.
        next_start = end - overlap

        # Avoid getting stuck if whitespace fallback behaves oddly.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

    
def build_chunk_records(
    text: str,
    document_id: str,
    version: int = 1,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Create chunks with deterministic, traceable IDs."""

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return [
        {
            "chunkId": (
                f"{document_id}:v{version}:chunk-{index:04d}"
            ),
            "documentId": document_id,
            "documentVersion": version,
            "chunkIndex": index,
            "text": chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]

