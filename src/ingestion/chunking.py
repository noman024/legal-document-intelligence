"""Character-based chunking with overlap."""

from __future__ import annotations

from src.config import settings
from src.ingestion.models import Chunk


def chunk_text(
    text: str,
    doc_id: str,
    *,
    source_path: str | None = None,
    page: int | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = chunk_size if chunk_size is not None else settings.chunk_size
    ov = overlap if overlap is not None else settings.chunk_overlap
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if ov >= size:
        ov = max(0, size // 4)

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    text=piece,
                    doc_id=doc_id,
                    chunk_index=idx,
                    page=page,
                    source_path=source_path,
                    extra={"char_start": start, "char_end": end},
                )
            )
            idx += 1
        if end >= len(text):
            break
        start = end - ov
    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    doc_id: str,
    *,
    source_path: str | None = None,
) -> list[Chunk]:
    """Chunk each page's text; chunk_index is global across the document."""
    all_chunks: list[Chunk] = []
    offset = 0
    for page_num, page_text in pages:
        page_chunks = chunk_text(
            page_text,
            doc_id,
            source_path=source_path,
            page=page_num,
        )
        for c in page_chunks:
            c.chunk_index = offset
            offset += 1
            all_chunks.append(c)
    return all_chunks
