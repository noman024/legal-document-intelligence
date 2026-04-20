"""PDF text extraction with OCR fallback for sparse pages."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import BinaryIO

import fitz  # PyMuPDF

from src.config import settings

logger = logging.getLogger(__name__)


def _ocr_page_image(page: fitz.Page, dpi: int = 200) -> str:
    """Rasterize PDF page and OCR with Tesseract."""
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(img)
    return text or ""


def extract_pages_from_pdf(
    file_path: Path | None = None,
    fileobj: BinaryIO | None = None,
    *,
    ocr_min_chars: int | None = None,
) -> list[tuple[int, str]]:
    """
    Returns list of (1-based page number, text) per page.
    Uses native text; if a page has very little text, runs OCR.
    """
    threshold = ocr_min_chars if ocr_min_chars is not None else settings.ocr_min_chars_per_page

    try:
        if file_path is not None:
            doc = fitz.open(file_path)
        elif fileobj is not None:
            data = fileobj.read()
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            raise ValueError("file_path or fileobj required")
    except fitz.FileDataError as e:
        raise ValueError(f"Invalid or unreadable PDF: {e}") from e

    try:
        pages: list[tuple[int, str]] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            raw = page.get_text("text") or ""
            text = raw.strip()
            if len(text) < threshold:
                try:
                    ocr = _ocr_page_image(page)
                    if len(ocr.strip()) > len(text):
                        text = ocr.strip()
                        logger.info("Page %s: used OCR (%s chars)", i + 1, len(text))
                except Exception as e:
                    logger.warning("OCR failed for page %s: %s", i + 1, e)
            pages.append((i + 1, text))
        return pages
    finally:
        doc.close()


def extract_pages_from_pdf_bytes(data: bytes) -> list[tuple[int, str]]:
    return extract_pages_from_pdf(fileobj=io.BytesIO(data))
