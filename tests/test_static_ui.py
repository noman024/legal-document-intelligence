"""Operator UI: static assets are served and contain expected structure."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deterministic_embeddings: None) -> TestClient:
    monkeypatch.setattr("src.config.settings.chroma_path", tmp_path / "chroma_ui")
    monkeypatch.setattr("src.config.settings.feedback_db_path", tmp_path / "fb_ui.db")
    import src.api.main as api_main

    api_main._store = None
    api_main._repo = None
    api_main._learner = None
    api_main._drafter = None
    return TestClient(api_main.app)


def test_ui_index_served(client: TestClient) -> None:
    r = client.get("/ui/")
    assert r.status_code == 200
    assert "Review workspace" in r.text
    assert "draft" in r.text.lower()


def test_ui_favicon_served(client: TestClient) -> None:
    r = client.get("/ui/favicon.svg")
    assert r.status_code == 200
    body = r.content.lower()
    assert b"<svg" in body or b"svg" in body


def test_static_file_matches_repo_copy(client: TestClient) -> None:
    root = Path(__file__).resolve().parents[1]
    disk = (root / "static" / "index.html").read_text(encoding="utf-8")
    assert client.get("/ui/").text == disk
