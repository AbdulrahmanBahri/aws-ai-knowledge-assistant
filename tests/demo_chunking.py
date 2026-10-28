
from application.ingestion.chunking import build_chunk_records

sample_text = """
Remote Work Policy

Employees may work remotely up to two days per week
with manager approval.

All employees must follow the company's data security
policy when working remotely.

Requests for exceptions must be submitted to Human Resources.
""".strip()

records = build_chunk_records(
    text=sample_text,
    document_id="remote-work-policy",
    version=1,
    chunk_size=100,
    overlap=20,
)

for record in records:
    print(f"\nID: {record['chunkId']}")
    print(f"Characters: {len(record['text'])}")
    print(record["text"])
