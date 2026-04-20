"""Edge cases for `extract_document_structure` and `compact_structure_json`."""

from __future__ import annotations

import json

from src.ingestion.structure import compact_structure_json, extract_document_structure


def test_empty_text_returns_empty_lists_with_page_count() -> None:
    s = extract_document_structure("", page_count=0)
    assert s["page_count"] == 0
    assert s["dates_found"] == []
    assert s["currency_amounts"] == []
    assert s["email_addresses"] == []


def test_date_formats_are_all_recognized() -> None:
    t = "On January 3, 2024 or 01/03/2024 or 2024-01-03 something happened."
    s = extract_document_structure(t, page_count=1)
    assert any("2024-01-03" in d for d in s["dates_found"])
    assert any("01/03/2024" in d for d in s["dates_found"])


def test_dedupe_preserves_first_occurrence() -> None:
    t = "Closing April 15, 2024. April 15, 2024 again. Price $100."
    s = extract_document_structure(t, page_count=1)
    # Date should appear only once even though text repeats.
    dates = s["dates_found"]
    assert len([d for d in dates if "April 15, 2024" in d]) == 1


def test_compact_structure_json_roundtrips_when_small() -> None:
    s = extract_document_structure("NOTICE: Hello.", page_count=1)
    blob = compact_structure_json(s, max_len=10_000)
    parsed = json.loads(blob)
    assert parsed["page_count"] == 1


def test_compact_structure_json_truncates_when_large() -> None:
    # Construct a very large structure; compaction must still fit under max_len.
    big = {
        "page_count": 1,
        "approx_char_count": 100,
        "dates_found": [f"d{i}" for i in range(500)],
        "currency_amounts": [f"$ {i}" for i in range(500)],
        "email_addresses": [f"a{i}@b.com" for i in range(100)],
        "legal_markers": ["MEMORANDUM"] * 20,
        "candidate_headings": ["HEADING " * 10 for _ in range(200)],
    }
    blob = compact_structure_json(big, max_len=400)
    assert len(blob) <= 400
    parsed = json.loads(blob)
    assert parsed.get("page_count") == 1
