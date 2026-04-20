from src.ingestion.chunking import chunk_text


def test_chunk_overlap():
    text = "a " * 500
    chunks = chunk_text(text, "doc1", chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert all(c.doc_id == "doc1" for c in chunks)


def test_empty_chunk():
    assert chunk_text("", "d") == []
