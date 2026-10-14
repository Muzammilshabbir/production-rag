from pathlib import Path

from app.services.rag_service import (
    ingest_document,
)


file_path = Path(
    "data/documents/security.txt"
)


text = file_path.read_text(
    encoding="utf-8"
)


ingest_document(
    document_id="security",
    text=text,
)


print("Document ingested successfully.")