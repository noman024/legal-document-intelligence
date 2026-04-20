"""HTTP API smoke and error paths (no Ollama for empty-index draft)."""

from pathlib import Path

import fitz
import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deterministic_embeddings: None,
) -> TestClient:
    monkeypatch.setattr("src.config.settings.chroma_path", tmp_path / "chroma_data")
    monkeypatch.setattr("src.config.settings.feedback_db_path", tmp_path / "feedback.db")
    import src.api.main as api_main

    api_main._store = None
    api_main._repo = None
    api_main._learner = None
    api_main._drafter = None
    return TestClient(api_main.app)


def test_favicon_redirects_to_ui_svg(client: TestClient) -> None:
    r = client.get("/favicon.ico", follow_redirects=False)
    assert r.status_code in (302, 307)
    loc = r.headers.get("location") or ""
    assert "favicon.svg" in loc


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data.get("service") == "legal-doc-intelligence"


def test_health_ready(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": [{"name": "llama3.2:latest"}]}

    monkeypatch.setattr("src.api.main.httpx.get", lambda *a, **k: _Resp())
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["ollama"] == "reachable"
    assert body.get("model_available") is True


def test_ingest_rejects_non_pdf(client: TestClient) -> None:
    r = client.post("/ingest", files={"file": ("note.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_ingest_rejects_corrupt_pdf(client: TestClient) -> None:
    r = client.post("/ingest", files={"file": ("bad.pdf", b"not a valid pdf", "application/pdf")})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_ingest_empty_file(client: TestClient) -> None:
    r = client.post("/ingest", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert r.status_code == 400


def test_ingest_returns_structured(client: TestClient, tmp_path: Path) -> None:
    p = tmp_path / "memo.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Closing April 15, 2024. Price $2,500,000.", fontsize=12)
    doc.save(p)
    doc.close()
    data = p.read_bytes()
    r = client.post("/ingest", files={"file": ("memo.pdf", data, "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    st = body.get("structured")
    assert isinstance(st, dict)
    assert st.get("page_count", 0) >= 1
    assert st.get("dates_found")


def test_draft_empty_index_no_ollama(client: TestClient) -> None:
    """With no chunks indexed, draft returns guidance text without calling Ollama."""
    r = client.post(
        "/draft",
        json={"task": "internal_memo", "query": "anything"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "No indexed document chunks" in body["text"]
    assert body["evidence"] == []


def test_draft_validation_missing_task(client: TestClient) -> None:
    r = client.post("/draft", json={"query": "only query"})
    assert r.status_code == 422


def test_feedback_unknown_draft(client: TestClient) -> None:
    r = client.post(
        "/feedback",
        json={"draft_id": "00000000-0000-0000-0000-000000000099", "edited_text": "x"},
    )
    assert r.status_code == 404


def test_endpoints_do_not_require_authorization(client: TestClient) -> None:
    """Spec has no API auth; clients may send unused Authorization headers."""
    r = client.get("/health", headers={"Authorization": "Bearer not-used"})
    assert r.status_code == 200


def test_root_redirects_to_ui_or_docs(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers.get("location")


def test_health_ready_degraded_when_model_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": [{"name": "some-other-model:latest"}]}

    monkeypatch.setattr("src.api.main.httpx.get", lambda *a, **k: _Resp())
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json().get("model_available") is False


def test_draft_malformed_json_body(client: TestClient) -> None:
    r = client.post(
        "/draft",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


def test_ingest_multi_page_pdf(client: TestClient, tmp_path: Path) -> None:
    """Larger PDF: multiple pages with extractable text (coordinates match PyMuPDF text layer)."""
    p = tmp_path / "many.pdf"
    line = "Representations and warranties; closing April 15, 2024. Escrow $125,000. "
    doc = fitz.open()
    for i in range(12):
        page = doc.new_page()
        page.insert_text((72, 72 + i * 2), (line * 15).strip(), fontsize=9)
    doc.save(p)
    doc.close()
    data = p.read_bytes()
    r = client.post("/ingest", files={"file": ("many.pdf", data, "application/pdf")})
    assert r.status_code == 200
    assert r.json().get("indexed_chunks", 0) >= 1
