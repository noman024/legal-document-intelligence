"""Ingest filename rules shared by API, CLI, and UI."""

from src.ingestion.formats import (
    INGEST_ALLOWED_SUFFIXES,
    is_allowed_ingest_filename,
    is_image_ingest_filename,
    is_pdf_ingest_filename,
)


def test_pdf_and_image_suffixes_recognized() -> None:
    assert is_pdf_ingest_filename("deal.PDF")
    assert is_image_ingest_filename("notes.TIFF")
    assert is_image_ingest_filename("scan.jpeg")
    assert is_allowed_ingest_filename("x.webp")
    assert not is_allowed_ingest_filename("memo.docx")
    assert not is_allowed_ingest_filename("")
    assert ".pdf" in INGEST_ALLOWED_SUFFIXES
    assert ".png" in INGEST_ALLOWED_SUFFIXES
