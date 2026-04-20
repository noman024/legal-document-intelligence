"""CLI: index PDF files from disk into Chroma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.pipeline import ingest_pdf  # noqa: E402
from src.retrieval.store import VectorStore  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Index PDF file(s) into Chroma.")
    p.add_argument("path", type=Path, help="PDF file or directory of PDFs")
    args = p.parse_args()
    path: Path = args.path

    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    store = VectorStore()
    paths: list[Path]
    if path.is_dir():
        paths = sorted(path.glob("*.pdf"))
    else:
        paths = [path]
    if not paths:
        print("No PDF files found.", file=sys.stderr)
        sys.exit(1)
    total = 0
    for pdf in paths:
        if pdf.suffix.lower() != ".pdf":
            continue
        chunks = ingest_pdf(pdf)
        n = store.upsert_chunks(chunks)
        total += n
        did = chunks[0].doc_id if chunks else "-"
        print(f"{pdf.name}: indexed {n} chunks (doc_id={did})")
    print(f"Done. Total chunks upserted: {total}")


if __name__ == "__main__":
    main()
