from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import streamlit as st
from requests import HTTPError

from backend.config import settings


st.set_page_config(page_title="Hospital Patient Query Assistant", page_icon="H", layout="wide")

API_BASE_URL = settings.frontend_api_base_url.rstrip("/")
DOCUMENT_SCOPE_KEY = "document_scope_widget"
LAST_SCOPE_KEY = "last_document_scope"


def init_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault(DOCUMENT_SCOPE_KEY, [])
    st.session_state.setdefault(LAST_SCOPE_KEY, [])
    st.session_state.setdefault("reset_document_scope", False)
    st.session_state.setdefault("knowledge_base_notice", "")


def api_get(path: str) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=120)
    response.raise_for_status()
    return response.json()


def api_post(path: str, *, json: dict[str, Any] | None = None, files: dict[str, Any] | None = None) -> Any:
    response = requests.post(f"{API_BASE_URL}{path}", json=json, files=files, timeout=300)
    response.raise_for_status()
    return response.json()


def api_delete(path: str) -> Any:
    response = requests.delete(f"{API_BASE_URL}{path}", timeout=120)
    response.raise_for_status()
    return response.json()


def reset_chat_for_knowledge_base_change(message: str) -> None:
    st.session_state["chat_history"] = []
    st.session_state["knowledge_base_notice"] = message


def schedule_scope_reset() -> None:
    st.session_state["reset_document_scope"] = True


def refresh_documents() -> list[dict[str, Any]]:
    documents = api_get("/documents")
    st.session_state["documents"] = documents
    return documents


def render_header() -> None:
    st.markdown(
        """
        <div class="hero fade-up">
            <div>
                <p class="eyebrow">Multi-Document Retrieval-Augmented Generation</p>
                <h1>Hospital Patient Query Assistant</h1>
                <p class="subtitle">
                    Upload one or more hospital PDFs, process them into searchable knowledge, and answer patient
                    questions using only grounded evidence from the indexed documents.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_text(text: str) -> None:
    safe_text = html.escape(text).replace("\n", "<br>")
    st.markdown(f"<div class='answer-text'>{safe_text}</div>", unsafe_allow_html=True)


def current_scope_labels() -> list[str]:
    return list(st.session_state.get(DOCUMENT_SCOPE_KEY, []))


def handle_scope_change() -> None:
    current_scope = current_scope_labels()
    previous_scope = st.session_state.get(LAST_SCOPE_KEY, [])
    if current_scope != previous_scope:
        st.session_state[LAST_SCOPE_KEY] = list(current_scope)
        reset_chat_for_knowledge_base_change("Search scope changed. Ask your question again for results from the new document set.")


def render_sidebar() -> list[str]:
    with st.sidebar:
        st.markdown("## Upload & Index")
        uploads = st.file_uploader(
            "Add hospital PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files. Each document is indexed independently.",
        )
        if st.button("Process Document(s)", use_container_width=True, disabled=not uploads):
            status_messages = []
            for upload in uploads or []:
                with st.spinner(f"Processing {upload.name}..."):
                    result = api_post("/upload", files={"file": (upload.name, upload.getvalue(), "application/pdf")})
                    status_messages.append((result.get("status"), result.get("message", "")))
            refresh_documents()
            reset_chat_for_knowledge_base_change("Knowledge base updated. Ask your question again.")
            for status, message in status_messages:
                if status == "already_uploaded":
                    st.info(message)
                else:
                    st.success(message or "Documents processed successfully.")

        st.markdown("## Search Scope")
        documents = refresh_documents()
        if not documents:
            st.info("No documents indexed yet.")
            return []

        options = {f"{doc['filename']} ({doc['status']})": doc["id"] for doc in documents}
        if st.session_state.get("reset_document_scope"):
            st.session_state[DOCUMENT_SCOPE_KEY] = []
            st.session_state[LAST_SCOPE_KEY] = []
            st.session_state["reset_document_scope"] = False
        valid_labels = [label for label in current_scope_labels() if label in options]
        st.session_state[DOCUMENT_SCOPE_KEY] = valid_labels
        st.multiselect(
            "Choose document scope",
            options=list(options.keys()),
            key=DOCUMENT_SCOPE_KEY,
            help="Leave empty to search across all indexed documents.",
            on_change=handle_scope_change,
        )

        st.markdown("## Indexed Files")
        for doc in documents:
            container = st.container(border=True)
            with container:
                st.markdown(f"**{doc['filename']}**")
                st.caption(
                    f"Status: {doc.get('status', 'unknown')} | Pages: {doc.get('page_count', 0)} | "
                    f"Chunks: {doc.get('chunk_count', 0)}"
                )
                if st.button("Remove", key=f"delete-{doc['id']}", use_container_width=True):
                    api_delete(f"/documents/{doc['id']}")
                    refresh_documents()
                    schedule_scope_reset()
                    reset_chat_for_knowledge_base_change(
                        f"Removed {doc['filename']}. Previous answers were cleared because the source set changed."
                    )
                    st.rerun()
        return [options[label] for label in current_scope_labels()]


def render_sources(response: dict[str, Any], selected_document_ids: list[str]) -> None:
    source_chunks = response.get("source_chunks", []) if isinstance(response, dict) else []
    if not source_chunks:
        return

    grouped: dict[str, list[int]] = {}
    for chunk in source_chunks:
        filename = str(chunk.get("filename", "Document"))
        page_number = chunk.get("page_number")
        if not isinstance(page_number, int):
            continue
        grouped.setdefault(filename, [])
        if page_number not in grouped[filename]:
            grouped[filename].append(page_number)

    if len(grouped) > 1 or len(selected_document_ids) > 1:
        labels = [f"{filename}: " + ", ".join(f"page {page}" for page in sorted(pages)) for filename, pages in grouped.items()]
        st.caption("Sources: " + " | ".join(labels))
        return

    pages = []
    for values in grouped.values():
        pages.extend(values)
    ordered_pages = sorted(set(pages))
    st.caption("Sources: " + ", ".join(f"page {page}" for page in ordered_pages))


def render_chat(selected_document_ids: list[str]) -> None:
    st.markdown("## Ask Questions")
    notice = st.session_state.get("knowledge_base_notice")
    if notice:
        st.info(notice)
        st.session_state["knowledge_base_notice"] = ""

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            render_answer_text(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("response", {}), message.get("selected_document_ids", []))

    prompt = st.chat_input("Ask a question about the uploaded hospital documents...")
    if not prompt:
        return

    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        render_answer_text(prompt)

    payload = {
        "question": prompt,
        "document_ids": selected_document_ids or None,
        "chat_history": st.session_state["chat_history"][-8:],
    }
    with st.chat_message("assistant"):
        with st.spinner("Retrieving answer..."):
            response = api_post("/query/details", json=payload)
        render_answer_text(response["answer"])
        render_sources(response, selected_document_ids)
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": response["answer"],
                "response": response,
                "selected_document_ids": list(selected_document_ids),
            }
        )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(30, 64, 175, 0.18), transparent 28%),
                    radial-gradient(circle at top right, rgba(8, 145, 178, 0.18), transparent 24%),
                    linear-gradient(180deg, #07111f 0%, #0b1728 100%);
                color: #ecf2ff;
            }
            .fade-up {
                animation: fadeUp .45s ease-out both;
            }
            @keyframes fadeUp {
                from { opacity: 0; transform: translateY(12px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .hero {
                padding: 1.5rem 1.75rem;
                border: 1px solid rgba(148, 163, 184, 0.18);
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(15, 118, 110, 0.26));
                border-radius: 24px;
                margin-bottom: 1.25rem;
                box-shadow: 0 20px 50px rgba(8, 15, 30, 0.35);
            }
            .eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.18em;
                color: #7dd3fc;
                font-size: 0.74rem;
                margin-bottom: 0.35rem;
            }
            .hero h1 {
                font-size: 2.5rem;
                margin: 0 0 0.55rem 0;
                color: #f8fafc;
            }
            .subtitle {
                color: #dbe7fb;
                max-width: 58rem;
                margin: 0;
                line-height: 1.7;
            }
            .answer-text {
                color: #f8fafc;
                line-height: 1.7;
                white-space: normal;
                word-break: break-word;
                font-size: 1rem;
            }
            [data-testid="stSidebar"] {
                background: rgba(7, 17, 31, 0.96);
                border-right: 1px solid rgba(148, 163, 184, 0.12);
            }
            [data-testid="stChatMessage"] {
                border-radius: 18px;
                border: 1px solid rgba(148, 163, 184, 0.12);
                background: rgba(15, 23, 42, 0.84);
                padding: 0.5rem 0.85rem;
                box-shadow: 0 12px 32px rgba(8, 15, 30, 0.28);
            }
            [data-testid="stFileUploaderDropzone"] {
                background: rgba(15, 23, 42, 0.92);
                border: 1px solid rgba(148, 163, 184, 0.16);
            }
            [data-testid="stFileUploaderDropzone"] * {
                color: #e5efff !important;
            }
            label, .stMarkdown, .stCaption, .stText {
                color: #e5efff !important;
            }
            [data-baseweb="select"] > div {
                background: rgba(15, 23, 42, 0.96) !important;
                color: #f8fafc !important;
                border: 1px solid rgba(148, 163, 184, 0.16) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    inject_styles()
    render_header()
    try:
        selected_document_ids = render_sidebar()
        render_chat(selected_document_ids)
    except HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Request failed: {detail}")
    except requests.RequestException as exc:
        st.error(f"Unable to reach backend at {API_BASE_URL}: {exc}")


if __name__ == "__main__":
    main()
