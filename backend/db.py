from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from supabase import Client, create_client

from backend.config import settings


logger = logging.getLogger(__name__)


def get_supabase_client() -> Client:
    settings.validate_required()
    return create_client(settings.supabase_url, settings.supabase_key)


def create_document(filename: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    client = get_supabase_client()
    payload = {
        "filename": filename,
        "status": "processing",
        "metadata": metadata or {},
    }
    response = client.table("documents").insert(payload).execute()
    return response.data[0]


def update_document(document_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    client = get_supabase_client()
    response = client.table("documents").update(updates).eq("id", document_id).execute()
    return response.data[0]


def list_documents() -> list[dict[str, Any]]:
    client = get_supabase_client()
    response = (
        client.table("documents")
        .select("*")
        .order("uploaded_at", desc=True)
        .execute()
    )
    return response.data or []


def delete_document(document_id: str) -> None:
    client = get_supabase_client()
    client.table("documents").delete().eq("id", document_id).execute()


def insert_document_chunks(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    client = get_supabase_client()
    batch_size = 100
    for offset in range(0, len(rows), batch_size):
        batch = list(rows[offset : offset + batch_size])
        client.table("document_chunks").insert(batch).execute()


def hybrid_search_chunks(
    query_embedding: list[float],
    question: str,
    document_ids: list[str] | None = None,
    match_count: int | None = None,
) -> list[dict[str, Any]]:
    client = get_supabase_client()
    params = {
        "query_text": question,
        "query_embedding": query_embedding,
        "match_count": match_count or settings.hybrid_match_count,
        "document_ids": document_ids,
    }
    response = client.rpc("hybrid_search_chunks", params).execute()
    return response.data or []
