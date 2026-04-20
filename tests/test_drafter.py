"""Drafter behavior without calling Ollama."""

from pathlib import Path

import pytest

from src.generation.drafter import Drafter
from src.retrieval.store import VectorStore


def test_draft_skips_llm_when_no_indexed_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "chroma_test"
    monkeypatch.setattr("src.config.settings.chroma_path", db)
    store = VectorStore(str(db))
    d = Drafter(store, learner=None)
    out = d.draft("internal_memo", query="test")
    assert "No indexed document chunks" in out.text
    assert out.evidence == []
    assert out.citations_valid == []


def test_draft_doc_id_filter_no_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Filter excludes everything in an empty store → same as no evidence."""
    db = tmp_path / "chroma_f"
    monkeypatch.setattr("src.config.settings.chroma_path", db)
    store = VectorStore(str(db))
    d = Drafter(store, None)
    out = d.draft("case_fact_summary", doc_id_filter="nonexistent-doc-id")
    assert "No indexed document chunks" in out.text
