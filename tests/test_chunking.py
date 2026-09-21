from utils.chunking import chunk_page_text, clean_text


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("Hello \n\n World   ") == "Hello World"


def test_chunk_page_text_preserves_page_metadata() -> None:
    text = " ".join(["token"] * 1500)
    chunks = chunk_page_text(text, page_number=2, chunk_size=300, overlap=50)
    assert len(chunks) >= 2
    assert all(chunk.page_number == 2 for chunk in chunks)
    assert chunks[0].source_label == "page 2"
