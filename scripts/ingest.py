"""CLI: index PDF files from disk into Chroma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.formats import INGEST_ALLOWED_SUFFIXES  # noqa: E402
from src.ingestion.pipeline import ingest_image_bytes, ingest_pdf  # noqa: E402
from src.retrieval.store import VectorStore  # noqa: E402


def _collect_paths(path: Path) -> list[Path]:
    if path.is_dir():
        found: list[Path] = []
        for dot_ext in sorted(INGEST_ALLOWED_SUFFIXES):
            ext = dot_ext.lstrip(".")
            found.extend(path.glob(f"*.{ext}"))
        return sorted({p.resolve() for p in found}, key=lambda p: p.name.lower())
    return [path]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Index PDF and/or image files (OCR for images) into Chroma.",
    )
    p.add_argument("path", type=Path, help="PDF/image file or directory of supported files")
    args = p.parse_args()
    path: Path = args.path

    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    store = VectorStore()
    paths = _collect_paths(path)
    if not paths:
        print("No supported files found.", file=sys.stderr)
        sys.exit(1)
    total = 0
    for item in paths:
        suf = item.suffix.lower()
        if suf not in INGEST_ALLOWED_SUFFIXES:
            print(f"Skip (unsupported): {item.name}", file=sys.stderr)
            continue
        try:
            if suf == ".pdf":
                chunks = ingest_pdf(item)
            else:
                data = item.read_bytes()
                chunks = ingest_image_bytes(data, filename=item.name)
        except ValueError as e:
            print(f"{item.name}: error: {e}", file=sys.stderr)
            continue
        n = store.upsert_chunks(chunks)
        total += n
        did = chunks[0].doc_id if chunks else "-"
        kind = "pdf" if suf == ".pdf" else "image"
        print(f"{item.name} ({kind}): indexed {n} chunks (doc_id={did})")
    print(f"Done. Total chunks upserted: {total}")
    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
