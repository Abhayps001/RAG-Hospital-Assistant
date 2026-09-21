from __future__ import annotations

from dataclasses import dataclass

from .embedding import get_tokenizer


@dataclass(slots=True)
class TextChunk:
    content: str
    page_number: int
    chunk_index: int
    source_label: str


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    collapsed = "\n".join(line for line in lines if line)
    return " ".join(collapsed.split())


def chunk_page_text(
    text: str,
    page_number: int,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[TextChunk]:
    normalized = clean_text(text)
    if not normalized:
        return []

    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(normalized, add_special_tokens=False)
    if not token_ids:
        return []

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(
                TextChunk(
                    content=chunk_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    source_label=f"page {page_number}",
                )
            )
            chunk_index += 1
        if end >= len(token_ids):
            break
        start = max(end - overlap, start + 1)
    return chunks
