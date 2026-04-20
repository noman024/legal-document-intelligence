#!/usr/bin/env python3
"""
Evaluation metrics for documentation (no Ollama; may download embedding weights on first run).

Usage (repo root, venv active):
  python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    from src.generation.citations import validate_citations
    from src.ingestion.pipeline import ingest_pdf
    from src.retrieval.store import RetrievedChunk, VectorStore, dedupe_retrieved_by_text

    pdf = _ROOT / "examples" / "sample_memo.pdf"
    if not pdf.is_file():
        print("Missing examples/sample_memo.pdf — run: python scripts/create_sample_pdf.py", file=sys.stderr)
        sys.exit(1)

    chunks = ingest_pdf(pdf)
    struct = (chunks[0].extra or {}).get("structured") if chunks else {}
    dates_n = len(struct.get("dates_found", []))
    money_n = len(struct.get("currency_amounts", []))
    markers_n = len(struct.get("legal_markers", []))

    ok, bad = validate_citations(
        "Summary [1]. Same cite [1][2]. Bad [99].",
        max_evidence=3,
    )
    assert ok == [1, 2] and bad == [99]

    dup = [
        RetrievedChunk("a", "same text", 0.9, {"doc_id": "x"}),
        RetrievedChunk("b", "same text", 0.8, {"doc_id": "y"}),
        RetrievedChunk("c", "other", 0.7, {"doc_id": "z"}),
    ]
    deduped = dedupe_retrieved_by_text(dup)
    assert len(deduped) == 2

    # Lightweight retrieval check: index temp store and query once.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        monkey_chroma = Path(td) / "ev_chroma"
        store = VectorStore(str(monkey_chroma))
        store.upsert_chunks(chunks[:3])
        hits = store.query("environmental risk purchase agreement", n_results=4)
        hit_ok = any(
            "environmental" in (h.text or "").lower() or "risk" in (h.text or "").lower() for h in hits
        )

    from src.ingestion.formats import INGEST_ALLOWED_SUFFIXES

    print("=== Evaluation (repeatable) ===")
    print(f"ingest.supported_suffixes: {sorted(INGEST_ALLOWED_SUFFIXES)}")
    print(f"structured_extraction.dates_found: {dates_n}")
    print(f"structured_extraction.currency_amounts: {money_n}")
    print(f"structured_extraction.legal_markers: {markers_n}")
    print("citation_validation: invalid [99] detected, valid [1,2]")
    print(f"retrieval_dedupe.unit: {len(deduped)} unique from {len(dup)} hits")
    print(f"retrieval_sample_memo.relevant_hit: {hit_ok}")


if __name__ == "__main__":
    main()
