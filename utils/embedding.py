from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from backend.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


@lru_cache(maxsize=1)
def get_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(settings.embedding_model_name)


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=16,
    )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def count_tokens(text: str) -> int:
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))
