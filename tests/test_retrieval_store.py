"""Vector store query behavior (deterministic embeddings via conftest)."""

from pathlib import Path

import pytest

from src.ingestion.models import Chunk
from src.retrieval.store import RetrievedChunk, VectorStore, dedupe_retrieved_by_text


def test_dedupe_retrieved_by_text_drops_duplicate_bodies() -> None:
    hits = [
        RetrievedChunk("a", "same text", 0.9, {"doc_id": "x"}),
        RetrievedChunk("b", "same text", 0.8, {"doc_id": "y"}),
        RetrievedChunk("c", "other", 0.7, {"doc_id": "z"}),
    ]
    out = dedupe_retrieved_by_text(hits)
    assert len(out) == 2
    assert {h.chunk_id for h in out} == {"a", "c"}


def test_query_empty_collection_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic_embeddings: None
) -> None:
    monkeypatch.setattr("src.config.settings.chroma_path", tmp_path / "empty_chroma")
    store = VectorStore(str(tmp_path / "empty_chroma"))
    assert store.query("anything", n_results=5) == []


def test_upsert_and_query_returns_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic_embeddings: None
) -> None:
    monkeypatch.setattr("src.config.settings.chroma_path", tmp_path / "ix")
    store = VectorStore(str(tmp_path / "ix"))
    chunks = [
        Chunk(
            text="The earnest money is held in escrow until closing.",
            doc_id="doc-a",
            chunk_index=0,
            page=1,
            source_path="memo.pdf",
        )
    ]
    assert store.upsert_chunks(chunks) == 1
    hits = store.query("escrow earnest money", n_results=3)
    assert len(hits) >= 1
    assert "escrow" in hits[0].text.lower()
