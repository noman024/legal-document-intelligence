"""Edge cases: corrupt PDFs, blank content, invalid inputs."""

from pathlib import Path

import fitz
import pytest

from src.ingestion.pipeline import ingest_pdf, ingest_pdf_bytes


def test_ingest_invalid_pdf_bytes():
    with pytest.raises(ValueError, match="Invalid|unreadable|Failed"):
        ingest_pdf_bytes(b"not a valid pdf", filename="bad.pdf")


def test_ingest_blank_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF with a page but no insert_text — may yield empty or minimal chunks."""
    monkeypatch.setattr("src.config.settings.ocr_min_chars_per_page", 0)
    p = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(p)
    doc.close()
    chunks = ingest_pdf(p)
    assert isinstance(chunks, list)


def test_ingest_whitespace_only_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.config.settings.ocr_min_chars_per_page", 0)
    p = tmp_path / "spaces.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "   \n\n   ", fontsize=12)
    doc.save(p)
    doc.close()
    chunks = ingest_pdf(p)
    assert isinstance(chunks, list)
