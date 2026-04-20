"""
End-to-end API test with indexed PDF chunks and mocked Ollama (no local LLM required).
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client_with_indexed_pdf(tmp_path, monkeypatch, deterministic_embeddings):
    """Fresh Chroma + feedback DB + sample PDF indexed."""
    monkeypatch.setattr("src.config.settings.chroma_path", tmp_path / "chroma_ix")
    monkeypatch.setattr("src.config.settings.feedback_db_path", tmp_path / "feedback_ix.db")
    import src.api.main as api_main

    api_main._store = None
    api_main._repo = None
    api_main._learner = None
    api_main._drafter = None

    from src.ingestion.pipeline import ingest_pdf

    pdf = Path(__file__).resolve().parents[1] / "examples" / "sample_memo.pdf"
    assert pdf.is_file(), f"Missing {pdf}; run scripts/create_sample_pdf.py"
    chunks = ingest_pdf(pdf)
    assert len(chunks) >= 1
    n = api_main.get_store().upsert_chunks(chunks)
    assert n >= 1

    with TestClient(api_main.app) as c:
        yield c


def test_draft_after_ingest_mock_llm(client_with_indexed_pdf: TestClient) -> None:
    with patch(
        "src.generation.drafter.generate_chat",
        return_value="## Summary\nClosing by April 15 per agreement [1].",
    ):
        r = client_with_indexed_pdf.post(
            "/draft",
            json={"task": "internal_memo", "query": "dates"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["draft_id"]
    assert "[1]" in body["text"] or "April" in body["text"]
    assert len(body["evidence"]) >= 1
    assert body["citations_valid"]
    assert body.get("citations_all_valid") is True


def test_feedback_round_trip_mock_llm(client_with_indexed_pdf: TestClient) -> None:
    with patch(
        "src.generation.drafter.generate_chat",
        return_value="Draft line [1].",
    ):
        r1 = client_with_indexed_pdf.post("/draft", json={"task": "internal_memo"})
    assert r1.status_code == 200
    did = r1.json()["draft_id"]

    r2 = client_with_indexed_pdf.post(
        "/feedback",
        json={"draft_id": did, "edited_text": "Edited draft line [1]."},
    )
    assert r2.status_code == 200
    assert r2.json()["ok"] is True


def test_draft_returns_503_when_llm_backend_fails(
    client_with_indexed_pdf: TestClient,
) -> None:
    with patch(
        "src.generation.drafter.generate_chat",
        side_effect=RuntimeError("Ollama unreachable"),
    ):
        r = client_with_indexed_pdf.post("/draft", json={"task": "internal_memo", "query": "x"})
    assert r.status_code == 503
    assert "detail" in r.json()
