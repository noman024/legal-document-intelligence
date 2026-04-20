"""Lightweight structured fields from legal-style text (regex + heuristics, no LLM)."""

from __future__ import annotations

import re
from typing import Any


_MONEY = re.compile(r"\$[\d,]+(?:\.\d{2})?\b")
_DATE_MDY = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_DATE_SLASH = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_DATE_ISOISH = re.compile(r"\b(?:20|19)\d{2}-\d{2}-\d{2}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_LEGAL_MARKERS = (
    "MEMORANDUM",
    "AGREEMENT",
    "PURCHASE",
    "LEASE",
    "NOTICE",
    "PLAINTIFF",
    "DEFENDANT",
    "PETITION",
    "ORDER",
)


def _uniq_preserve(seq: list[str], cap: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= cap:
            break
    return out


def extract_document_structure(full_text: str, *, page_count: int) -> dict[str, Any]:
    """
    Derive a small JSON-serializable structure for downstream retrieval and drafting.
    Not legal advice; heuristic extraction only.
    """
    text = full_text[:400_000]

    dates: list[str] = []
    for rx in (_DATE_MDY, _DATE_SLASH, _DATE_ISOISH):
        dates.extend(m.group(0) for m in rx.finditer(text))
    dates = _uniq_preserve(dates, 40)

    money = _uniq_preserve([m.group(0) for m in _MONEY.finditer(text)], 40)
    emails = _uniq_preserve([m.group(0) for m in _EMAIL.finditer(text)], 20)

    upper_blob = text.upper()
    markers_found = [m for m in _LEGAL_MARKERS if m in upper_blob][:20]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    heading_like = [
        ln[:200]
        for ln in lines
        if 3 < len(ln) < 200
        and (
            (ln.isupper() and len(ln.split()) <= 14)
            or re.match(r"^(?:Re:|RE:|In the Matter of|In re:)\s+", ln, re.I)
        )
    ][:15]

    return {
        "page_count": page_count,
        "approx_char_count": len(full_text),
        "dates_found": dates,
        "currency_amounts": money,
        "email_addresses": emails,
        "legal_markers": markers_found,
        "candidate_headings": heading_like,
    }


def compact_structure_json(struct: dict[str, Any], max_len: int = 3500) -> str:
    """Serialize for Chroma metadata (string values only)."""
    import json

    def dumps(d: dict[str, Any]) -> str:
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))

    cur = dict(struct)
    for _ in range(12):
        s = dumps(cur)
        if len(s) <= max_len:
            return s
        for key in ("candidate_headings", "dates_found", "currency_amounts", "email_addresses", "legal_markers"):
            if isinstance(cur.get(key), list) and len(cur[key]) > 1:
                cur[key] = cur[key][: max(1, len(cur[key]) // 2)]
                break
        else:
            return dumps({"page_count": cur.get("page_count"), "truncated": True})
    return dumps({"page_count": struct.get("page_count"), "truncated": True})
