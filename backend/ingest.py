from __future__ import annotations

import hashlib
import io
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import UploadFile
from pypdf import PdfReader

from backend import db
from utils.chunking import TextChunk, chunk_page_text
from utils.embedding import embed_texts


logger = logging.getLogger(__name__)


def extract_pages_from_pdf(file_bytes: bytes) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page_number": idx, "text": text})
    return pages


def build_chunks(pages: list[dict[str, Any]]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page in pages:
        chunks.extend(chunk_page_text(page["text"], page["page_number"]))
    return chunks


def find_duplicate_document(filename: str, content_hash: str) -> dict[str, Any] | None:
    documents = db.list_documents()
    for document in documents:
        metadata = document.get("metadata") or {}
        if metadata.get("content_hash") == content_hash:
            return document
        if document.get("filename") == filename and document.get("status") == "ready":
            return document
    return None


async def ingest_pdf(upload_file: UploadFile) -> dict[str, Any]:
    filename = upload_file.filename or "uploaded.pdf"
    file_bytes = await upload_file.read()
    if not file_bytes:
        raise ValueError("Uploaded PDF is empty.")

    content_hash = hashlib.sha256(file_bytes).hexdigest()
    duplicate = find_duplicate_document(filename, content_hash)
    if duplicate:
        return {
            "document_id": duplicate["id"],
            "filename": duplicate["filename"],
            "status": "already_uploaded",
            "page_count": duplicate.get("page_count", 0),
            "chunk_count": duplicate.get("chunk_count", 0),
            "message": f"{duplicate['filename']} is already uploaded.",
        }

    document = db.create_document(
        filename,
        metadata={
            "content_type": upload_file.content_type,
            "content_hash": content_hash,
        },
    )
    document_id = document["id"]

    try:
        pages = extract_pages_from_pdf(file_bytes)
        chunks = build_chunks(pages)
        if not chunks:
            raise ValueError("No extractable text was found in the PDF.")

        embeddings = embed_texts(chunk.content for chunk in chunks)
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            rows.append(
                {
                    "document_id": document_id,
                    "content": chunk.content,
                    "embedding": embedding,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "source_label": chunk.source_label,
                }
            )
        db.insert_document_chunks(rows)

        updated = db.update_document(
            document_id,
            {
                "status": "ready",
                "page_count": len(pages),
                "chunk_count": len(chunks),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return {
            "document_id": updated["id"],
            "filename": updated["filename"],
            "status": updated["status"],
            "page_count": updated["page_count"],
            "chunk_count": updated["chunk_count"],
            "message": f"{updated['filename']} processed successfully.",
        }
    except Exception as exc:
        logger.exception("Failed to ingest PDF %s", filename)
        db.update_document(
            document_id,
            {
                "status": "failed",
                "metadata": {
                    "content_type": upload_file.content_type,
                    "content_hash": content_hash,
                    "error": str(exc),
                },
            },
        )
        raise
