"""Edge cases for `chunk_text` / `chunk_pages`."""

from __future__ import annotations

import pytest

from src.ingestion.chunking import chunk_pages, chunk_text


def test_chunk_text_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", "doc", chunk_size=0)


def test_chunk_text_overlap_gte_size_is_reduced() -> None:
    chunks = chunk_text("a" * 40, "doc", chunk_size=10, overlap=10)
    # Should still progress; not infinite loop.
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)


def test_chunk_text_whitespace_only_returns_empty() -> None:
    assert chunk_text("   \n\t  ", "d") == []


def test_chunk_pages_global_index_is_monotonic() -> None:
    pages = [(1, "aaaa " * 200), (2, "bbbb " * 200)]
    chunks = chunk_pages(pages, doc_id="d", source_path="s.pdf")
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)


def test_chunk_pages_carries_page_metadata() -> None:
    pages = [(7, "hello " * 200)]
    chunks = chunk_pages(pages, doc_id="d")
    assert all(c.page == 7 for c in chunks)


def test_chunk_stable_id_includes_page() -> None:
    pages = [(3, "x " * 200)]
    chunks = chunk_pages(pages, doc_id="D")
    sid = chunks[0].stable_id()
    assert sid.startswith("D:")
    assert ":p3" in sid
