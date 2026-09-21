from backend.rag_pipeline import FALLBACK_ANSWER, build_context, build_sources, deduplicate_chunks, format_source_chunks


def test_deduplicate_chunks_removes_exact_duplicates() -> None:
    chunks = [
        {"document_id": "doc-1", "page_number": 1, "content": "MRI costs $450"},
        {"document_id": "doc-1", "page_number": 1, "content": "MRI costs $450"},
        {"document_id": "doc-2", "page_number": 2, "content": "ICU costs $600/day"},
    ]
    deduped = deduplicate_chunks(chunks)
    assert len(deduped) == 2


def test_build_sources_uses_filename_and_page() -> None:
    chunks = [
        {"filename": "hospital-a.pdf", "page_number": 1},
        {"filename": "hospital-b.pdf", "page_number": 3},
    ]
    assert build_sources(chunks) == ["hospital-a.pdf - page 1", "hospital-b.pdf - page 3"]


def test_build_context_contains_source_headers() -> None:
    chunks = [{"filename": "hospital.pdf", "page_number": 4, "content": "Emergency number is 1066"}]
    context = build_context(chunks)
    assert "[Source 1] hospital.pdf - page 4" in context
    assert "Emergency number is 1066" in context


def test_format_source_chunks_includes_score() -> None:
    formatted = format_source_chunks(
        [
            {
                "document_id": "doc-1",
                "filename": "hospital.pdf",
                "page_number": 5,
                "content": "OPD timings are 9 AM to 5 PM",
                "combined_score": 0.88,
            }
        ]
    )
    assert formatted[0]["score"] == 0.88


def test_fallback_answer_is_exact() -> None:
    assert FALLBACK_ANSWER == "I don't have that information in the provided document."
