"""Supported ingest file types (API, CLI, UI)."""

from __future__ import annotations

from pathlib import Path

INGEST_PDF_SUFFIXES: frozenset[str] = frozenset({".pdf"})
INGEST_IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"})
INGEST_ALLOWED_SUFFIXES: frozenset[str] = INGEST_PDF_SUFFIXES | INGEST_IMAGE_SUFFIXES


def ingest_suffix(filename: str) -> str:
    """Lowercase suffix including dot, e.g. `.PDF` -> `.pdf`."""
    return Path(filename).suffix.lower()


def is_allowed_ingest_filename(filename: str) -> bool:
    return bool(filename) and ingest_suffix(filename) in INGEST_ALLOWED_SUFFIXES


def is_pdf_ingest_filename(filename: str) -> bool:
    return ingest_suffix(filename) in INGEST_PDF_SUFFIXES


def is_image_ingest_filename(filename: str) -> bool:
    return ingest_suffix(filename) in INGEST_IMAGE_SUFFIXES
