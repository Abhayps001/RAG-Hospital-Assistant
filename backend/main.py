from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import db
from backend.config import settings
from backend.ingest import ingest_pdf
from backend.rag_pipeline import answer_question
from utils.embedding import get_embedding_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="Multi-Document Hospital RAG Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    document_ids: list[str] | None = None
    chat_history: list[ChatMessage] | None = None


def build_strict_query_response(result: dict[str, object]) -> dict[str, object]:
    answer = str(result.get("answer", ""))
    if answer == "I don't have that information in the provided document.":
        return {"answer": answer, "sources": []}

    page_numbers: set[int] = set()
    for chunk in result.get("source_chunks", []):
        page_number = chunk.get("page_number")
        if isinstance(page_number, int):
            page_numbers.add(page_number)

    ordered_sources = [f"page {page_number}" for page_number in sorted(page_numbers)]
    return {
        "answer": answer,
        "sources": ordered_sources,
    }


@app.get("/health")
def health() -> dict[str, object]:
    dependencies = {
        "supabase_configured": bool(settings.supabase_url and settings.supabase_key),
        "groq_configured": bool(settings.groq_api_key),
        "embedding_model": settings.embedding_model_name,
    }
    try:
        get_embedding_model()
        dependencies["embedding_loaded"] = True
    except Exception:
        dependencies["embedding_loaded"] = False
    return {"status": "ok", "dependencies": dependencies}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        return await ingest_pdf(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {exc}") from exc


@app.get("/documents")
def get_documents() -> list[dict[str, object]]:
    try:
        return db.list_documents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {exc}") from exc


@app.delete("/documents/{document_id}")
def remove_document(document_id: str) -> dict[str, str]:
    try:
        db.delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {exc}") from exc


@app.post("/query")
def query_documents(request: QueryRequest) -> dict[str, object]:
    try:
        result = answer_question(
            question=request.question,
            document_ids=request.document_ids,
            chat_history=[message.model_dump() for message in request.chat_history or []],
        )
        return build_strict_query_response(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc


@app.post("/query/details")
def query_documents_with_details(request: QueryRequest) -> dict[str, object]:
    try:
        return answer_question(
            question=request.question,
            document_ids=request.document_ids,
            chat_history=[message.model_dump() for message in request.chat_history or []],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc
