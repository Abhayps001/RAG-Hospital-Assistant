from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict, defaultdict
from typing import Any

from groq import Groq

from backend import db
from backend.config import settings
from utils.embedding import embed_query


FALLBACK_ANSWER = "I don't have that information in the provided document."
SOURCE_PATTERN = re.compile(r"\s*\((?:Source|source)[^)]*\)\s*")
INDEX_PATTERN = re.compile(r"\d+")


def make_display_name(filename: str) -> str:
    stem, _ = os.path.splitext(filename)
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    words = [word.capitalize() for word in cleaned.split()]
    return " ".join(words) or filename


def deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for chunk in chunks:
        fingerprint = hashlib.sha1(
            f"{chunk.get('document_id')}|{chunk.get('page_number')}|{chunk.get('content')}".encode("utf-8")
        ).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(chunk)
    return deduped


def get_ready_documents() -> list[dict[str, Any]]:
    return [doc for doc in db.list_documents() if doc.get("status") == "ready"]


def resolve_scope_documents(document_ids: list[str] | None) -> list[dict[str, Any]]:
    ready_documents = get_ready_documents()
    if document_ids:
        allowed = set(document_ids)
        return [doc for doc in ready_documents if doc.get("id") in allowed]
    return ready_documents


def select_context_chunks(chunks: list[dict[str, Any]], scoped_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = deduplicate_chunks(chunks)
    if len(scoped_documents) <= 1:
        return deduped[: settings.final_context_chunks]

    max_context_chunks = max(settings.final_context_chunks, min(len(scoped_documents) * 2, 8))
    per_doc_quota = 2 if len(scoped_documents) <= 3 else 1
    selected: list[dict[str, Any]] = []
    selected_fingerprints: set[str] = set()
    per_doc_counts: dict[str, int] = defaultdict(int)

    for chunk in deduped:
        document_id = str(chunk.get("document_id"))
        fingerprint = hashlib.sha1(
            f"{document_id}|{chunk.get('page_number')}|{chunk.get('content')}".encode("utf-8")
        ).hexdigest()
        if fingerprint in selected_fingerprints:
            continue
        if per_doc_counts[document_id] >= per_doc_quota:
            continue
        selected.append(chunk)
        selected_fingerprints.add(fingerprint)
        per_doc_counts[document_id] += 1
        if len(selected) >= max_context_chunks:
            return selected

    for chunk in deduped:
        document_id = str(chunk.get("document_id"))
        fingerprint = hashlib.sha1(
            f"{document_id}|{chunk.get('page_number')}|{chunk.get('content')}".encode("utf-8")
        ).hexdigest()
        if fingerprint in selected_fingerprints:
            continue
        selected.append(chunk)
        selected_fingerprints.add(fingerprint)
        if len(selected) >= max_context_chunks:
            break
    return selected


def build_context(chunks: list[dict[str, Any]]) -> str:
    sections = []
    for idx, chunk in enumerate(chunks, start=1):
        filename = str(chunk.get("filename", "Unknown document"))
        display_name = make_display_name(filename)
        page = chunk.get("page_number", "?")
        content = str(chunk.get("content", "")).strip()
        sections.append(f"[Source {idx}] {display_name} ({filename}) - page {page}\n{content}")
    return "\n\n".join(sections)


def build_sources(chunks: list[dict[str, Any]]) -> list[str]:
    ordered = OrderedDict()
    for chunk in chunks:
        label = f"{chunk.get('filename', 'Unknown document')} - page {chunk.get('page_number', '?')}"
        ordered[label] = None
    return list(ordered.keys())


def format_source_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": chunk.get("document_id"),
            "filename": chunk.get("filename"),
            "display_name": make_display_name(str(chunk.get("filename", "Unknown document"))),
            "page_number": chunk.get("page_number"),
            "content": chunk.get("content"),
            "score": chunk.get("combined_score"),
        }
        for chunk in chunks
    ]


def build_answer_messages(
    question: str,
    context: str,
    chat_history: list[dict[str, str]] | None = None,
    multi_document: bool = False,
) -> list[dict[str, str]]:
    multi_doc_instruction = (
        "If more than one selected document contains the answer, write one short line per document.\n"
        "Use this exact format on separate lines: <document name>: <answer>.\n"
        "Use the human-readable document name from the context label, not the raw filename.\n"
        "Do not put two document answers on the same line.\n"
    ) if multi_document else ""

    system_prompt = (
        "You are a hospital assistant AI.\n"
        "Answer ONLY using the provided context.\n"
        f'If the answer is not present, reply exactly with: "{FALLBACK_ANSWER}"\n'
        "Do not use outside knowledge, do not guess, and do not merge conflicting facts across documents.\n"
        "Do not mention Source numbers or page numbers inside the answer body.\n"
        + multi_doc_instruction +
        "Keep the answer short and directly useful."
    )
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for message in chat_history[-6:]:
            role = message.get("role", "")
            content = message.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    messages.append({"role": "user", "content": user_prompt})
    return messages


def sanitize_answer(answer: str) -> str:
    cleaned = SOURCE_PATTERN.sub(" ", answer).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace(" .", ".")
    return cleaned or FALLBACK_ANSWER


def parse_source_indexes(raw_text: str, max_index: int) -> list[int]:
    candidates: list[int] = []
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            values = parsed.get("sources", [])
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, int) and 1 <= value <= max_index:
                        candidates.append(value)
    except json.JSONDecodeError:
        for token in INDEX_PATTERN.findall(raw_text):
            value = int(token)
            if 1 <= value <= max_index:
                candidates.append(value)

    ordered: list[int] = []
    seen: set[int] = set()
    for value in candidates:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def select_supporting_chunks_with_llm(client: Groq, question: str, answer: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_sources = []
    for idx, chunk in enumerate(chunks, start=1):
        compact_sources.append(
            f"Source {idx} | {chunk.get('filename')} | page {chunk.get('page_number')}\n{chunk.get('content', '')}"
        )

    prompt = (
        "Choose only the source numbers that directly support the final answer.\n"
        "Return strict JSON with this shape only: {\"sources\": [1, 2]}\n"
        "Do not include a source unless the answer is explicitly supported by that source text.\n"
        "Prefer the smallest correct set of sources.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Available sources:\n"
        + "\n\n".join(compact_sources)
    )

    completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": "You select supporting citations from provided evidence only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw_selection = completion.choices[0].message.content.strip()
    indexes = parse_source_indexes(raw_selection, len(chunks))
    if not indexes:
        return []
    return [chunks[index - 1] for index in indexes]


def get_groq_client() -> Groq:
    settings.validate_required()
    return Groq(api_key=settings.groq_api_key)


def answer_question(
    question: str,
    document_ids: list[str] | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    scoped_documents = resolve_scope_documents(document_ids)
    if not scoped_documents:
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "source_chunks": [],
            "matched_documents": [],
        }

    scoped_document_ids = [str(doc.get("id")) for doc in scoped_documents]
    match_count = max(settings.hybrid_match_count * 3, 18) if len(scoped_documents) > 1 else settings.hybrid_match_count
    query_embedding = embed_query(question)
    raw_chunks = db.hybrid_search_chunks(
        query_embedding,
        question,
        document_ids=scoped_document_ids,
        match_count=match_count,
    )
    context_chunks = select_context_chunks(raw_chunks, scoped_documents)

    if not context_chunks:
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "source_chunks": [],
            "matched_documents": [],
        }

    context = build_context(context_chunks)
    client = get_groq_client()
    answer_completion = client.chat.completions.create(
        model=settings.groq_model,
        messages=build_answer_messages(
            question=question,
            context=context,
            chat_history=chat_history,
            multi_document=len(scoped_documents) > 1,
        ),
        temperature=0,
    )
    answer = sanitize_answer(answer_completion.choices[0].message.content.strip())

    if answer == FALLBACK_ANSWER:
        return {
            "answer": answer,
            "sources": [],
            "source_chunks": [],
            "matched_documents": [],
        }

    supporting_chunks = select_supporting_chunks_with_llm(client, question, answer, context_chunks)
    if not supporting_chunks:
        supporting_chunks = context_chunks[:1]

    matched_documents = list(
        OrderedDict((chunk.get("document_id"), chunk.get("filename")) for chunk in supporting_chunks).items()
    )
    return {
        "answer": answer,
        "sources": build_sources(supporting_chunks),
        "source_chunks": format_source_chunks(supporting_chunks),
        "matched_documents": [
            {"document_id": document_id, "filename": filename}
            for document_id, filename in matched_documents
        ],
    }
