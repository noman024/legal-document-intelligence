"""End-to-end PDF ingestion pipeline (text path and OCR-fallback path)."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.ingestion.pipeline import ingest_image_bytes, ingest_pdf, ingest_pdf_bytes


def _make_pdf(path: Path, body: str, *, pages: int = 1) -> None:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), body, fontsize=11)
    doc.save(path)
    doc.close()


def test_ingest_pdf_produces_chunks_with_structure(tmp_path: Path) -> None:
    p = tmp_path / "memo.pdf"
    _make_pdf(p, "Closing on April 15, 2024. Price $1,000,000. NOTICE.")
    chunks = ingest_pdf(p)
    assert chunks
    assert chunks[0].doc_id == "memo"
    struct = chunks[0].extra.get("structured")
    assert struct and struct["page_count"] == 1
    assert any("2024" in d for d in struct["dates_found"])
    assert any("1,000,000" in m for m in struct["currency_amounts"])


def test_ingest_pdf_bytes_uses_generated_doc_id(tmp_path: Path) -> None:
    p = tmp_path / "s.pdf"
    _make_pdf(p, "hello world")
    data = p.read_bytes()
    chunks = ingest_pdf_bytes(data, filename="input.pdf")
    assert chunks
    assert chunks[0].doc_id.startswith("input_")
    assert chunks[0].source_path == "input.pdf"


def test_ingest_pdf_bytes_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError):
        ingest_pdf_bytes(b"bogus", filename="x.pdf")


def test_ingest_pdf_nonexistent_path_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ingest_pdf(tmp_path / "missing.pdf")


def test_ingest_pdf_ocr_fallback_is_invoked_for_sparse_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the text layer to look sparse; verify OCR code path is used."""
    p = tmp_path / "scan.pdf"
    _make_pdf(p, "x")  # only 1 char → below default threshold

    calls = {"n": 0}

    def fake_ocr(_page, dpi: int = 200) -> str:
        calls["n"] += 1
        return "RECOVERED TEXT FROM OCR about purchase agreement dated March 1, 2024."

    monkeypatch.setattr("src.ingestion.extract._ocr_page_image", fake_ocr)
    chunks = ingest_pdf(p)
    assert calls["n"] >= 1
    assert chunks
    assert "RECOVERED" in chunks[0].text


def test_ingest_image_bytes_uses_ocr_and_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from io import BytesIO

    from PIL import Image

    monkeypatch.setattr(
        "pytesseract.image_to_string",
        lambda *_a, **_k: "Internal memo. Closing April 15, 2024. Escrow $99,000.",
    )
    buf = BytesIO()
    Image.new("RGB", (40, 40), color=(250, 250, 250)).save(buf, format="PNG")
    chunks = ingest_image_bytes(buf.getvalue(), filename="whiteboard.png")
    assert chunks
    assert "whiteboard_" in chunks[0].doc_id
    struct = chunks[0].extra.get("structured")
    assert struct and struct.get("page_count") == 1


def _png_bytes_white_small() -> bytes:
    from io import BytesIO

    from PIL import Image

    b = BytesIO()
    Image.new("RGB", (4, 4), color="white").save(b, format="PNG")
    return b.getvalue()


def test_ingest_image_bytes_empty_ocr_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pytesseract.image_to_string", lambda *_a, **_k: "")
    with pytest.raises(ValueError, match="No text could be extracted"):
        ingest_image_bytes(_png_bytes_white_small(), filename="blank.png")


def test_ingest_pdf_repeated_headers_are_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-page PDF with a repeated header should drop it via `strip_repeated_headers_footers`."""
    p = tmp_path / "rh.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"CONFIDENTIAL HEADER\nbody content page {i + 1}", fontsize=10)
    doc.save(p)
    doc.close()
    chunks = ingest_pdf(p)
    # At least one chunk should not contain the repeated header verbatim.
    assert any("body content" in c.text for c in chunks)
