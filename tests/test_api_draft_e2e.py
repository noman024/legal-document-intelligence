"""API /draft + /feedback with indexed chunks and mocked LLM: full pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def api_with_indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic_embeddings):
    monkeypatch.setattr("src.config.settings.chroma_path", tmp_path / "ch")
    monkeypatch.setattr("src.config.settings.feedback_db_path", tmp_path / "fb.db")
    import src.api.main as api_main

    api_main._store = None
    api_main._repo = None
    api_main._learner = None
    api_main._drafter = None

    pdf = tmp_path / "memo.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Purchase Agreement. Closing April 15, 2024. Escrow $125,000. "
        "Title objections within ten business days.",
        fontsize=10,
    )
    doc.save(pdf)
    doc.close()

    client = TestClient(api_main.app)
    r = client.post(
        "/ingest",
        files={"file": ("memo.pdf", pdf.read_bytes(), "application/pdf")},
    )
    assert r.status_code == 200
    return client


def test_draft_returns_evidence_with_citations(api_with_indexed: TestClient) -> None:
    with patch(
        "src.generation.drafter.generate_chat",
        return_value="Key date: April 15, 2024 [1]. Escrow held [1].",
    ):
        r = api_with_indexed.post(
            "/draft",
            json={"task": "internal_memo", "query": "closing date"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["citations_valid"] == [1]
    assert not body["citations_invalid"]
    assert body.get("citations_all_valid") is True
    assert body["evidence"] and body["evidence"][0]["index"] == 1
    assert "page" in body["evidence"][0]
    assert body["evidence"][0]["text"]


def test_draft_reports_invalid_citations(api_with_indexed: TestClient) -> None:
    """Model hallucinated evidence index → surfaced to caller."""
    with patch(
        "src.generation.drafter.generate_chat",
        return_value="Claim [99] with no basis.",
    ):
        r = api_with_indexed.post("/draft", json={"task": "case_fact_summary"})
    assert r.status_code == 200
    body = r.json()
    assert 99 in body["citations_invalid"]
    assert body.get("citations_all_valid") is False


def test_strict_citations_returns_422_without_saving_draft(
    api_with_indexed: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.api.main as api_main

    monkeypatch.setattr(api_main.settings, "strict_citations", True)
    with patch(
        "src.generation.drafter.generate_chat",
        return_value="Unsupported claim [99].",
    ):
        r = api_with_indexed.post("/draft", json={"task": "case_fact_summary"})
    assert r.status_code == 422
    detail = r.json().get("detail")
    assert isinstance(detail, dict)
    assert 99 in detail.get("citations_invalid", [])

    with patch("src.generation.drafter.generate_chat", return_value="Only [1]."):
        r2 = api_with_indexed.post("/draft", json={"task": "case_fact_summary"})
    assert r2.status_code == 200
    assert r2.json().get("citations_all_valid") is True


def test_feedback_updates_learning_and_improves_second_draft(api_with_indexed: TestClient) -> None:
    """Submit edit → next draft sees few-shot augmentation in system prompt."""
    with patch("src.generation.drafter.generate_chat", return_value="plain [1]"):
        r1 = api_with_indexed.post("/draft", json={"task": "internal_memo"})
    assert r1.status_code == 200
    did = r1.json()["draft_id"]

    r2 = api_with_indexed.post(
        "/feedback",
        json={"draft_id": did, "edited_text": "plain [1]\nOperator prefers bullet points."},
    )
    assert r2.status_code == 200
    assert r2.json()["correction_bullets"]

    captured: dict = {}

    def _capture(messages, **_kw):
        captured["system"] = messages[0]["content"]
        return "second [1]"

    with patch("src.generation.drafter.generate_chat", side_effect=_capture):
        r3 = api_with_indexed.post("/draft", json={"task": "internal_memo"})
    assert r3.status_code == 200
    assert "Operator preferences" in captured["system"]


def test_draft_with_unknown_doc_id_filter_returns_guidance(api_with_indexed: TestClient) -> None:
    r = api_with_indexed.post(
        "/draft",
        json={"task": "internal_memo", "doc_id_filter": "does-not-exist"},
    )
    assert r.status_code == 200
    assert "No indexed document chunks" in r.json()["text"]


def test_draft_every_supported_task(api_with_indexed: TestClient) -> None:
    """Smoke: each supported task type routes through Drafter without error."""
    tasks = [
        "case_fact_summary",
        "document_summary",
        "title_review_summary",
        "internal_memo",
        "notice_summary",
        "document_checklist",
    ]
    for t in tasks:
        with patch("src.generation.drafter.generate_chat", return_value=f"{t} [1]"):
            r = api_with_indexed.post("/draft", json={"task": t})
        assert r.status_code == 200, (t, r.text)
