"""End-to-end PDF -> Chunk list."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from src.ingestion.chunking import chunk_pages
from src.ingestion.extract import extract_pages_from_pdf, extract_pages_from_pdf_bytes
from src.ingestion.models import Chunk
from src.ingestion.structure import compact_structure_json, extract_document_structure
from src.ingestion.text_clean import normalize_whitespace, strip_repeated_headers_footers

logger = logging.getLogger(__name__)


def _prepare_pages(raw_pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    cleaned = [(p, normalize_whitespace(t)) for p, t in raw_pages]
    texts = [t for _, t in cleaned]
    stripped = strip_repeated_headers_footers(texts)
    return [(cleaned[i][0], stripped[i]) for i in range(len(cleaned))]


def _attach_document_structure(chunks: list[Chunk], pages: list[tuple[int, str]]) -> list[Chunk]:
    full_text = "\n\n".join(t for _, t in pages)
    struct = extract_document_structure(full_text, page_count=len(pages))
    sj = compact_structure_json(struct)
    for c in chunks:
        c.extra["structured"] = struct
        c.extra["structured_json"] = sj
    return chunks


def ingest_pdf(path: Path, doc_id: str | None = None) -> list[Chunk]:
    """Read PDF from disk; return chunks with metadata."""
    did = doc_id or path.stem
    try:
        raw = extract_pages_from_pdf(file_path=path)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read PDF {path}: {e}") from e
    pages = _prepare_pages(raw)
    chunks = chunk_pages(pages, did, source_path=str(path.resolve()))
    if chunks:
        chunks = _attach_document_structure(chunks, pages)
    if not chunks:
        logger.warning("No text chunks produced for %s (empty or unreadable pages?)", path)
    return chunks


def ingest_pdf_bytes(
    data: bytes,
    *,
    filename: str = "upload.pdf",
    doc_id: str | None = None,
) -> list[Chunk]:
    """Ingest PDF from memory (e.g. upload)."""
    did = doc_id or f"{Path(filename).stem}_{uuid4().hex[:8]}"
    try:
        raw = extract_pages_from_pdf_bytes(data)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read PDF {filename!r}: {e}") from e
    pages = _prepare_pages(raw)
    chunks = chunk_pages(pages, did, source_path=filename)
    if chunks:
        chunks = _attach_document_structure(chunks, pages)
    if not chunks:
        logger.warning("No text chunks produced for upload %s", filename)
    return chunks
