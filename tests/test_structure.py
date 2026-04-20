"""Structured field extraction (no PDF required)."""

from src.ingestion.structure import extract_document_structure


def test_extract_finds_dates_and_money() -> None:
    text = """
    INTERNAL MEMORANDUM
    Re: Purchase Agreement — 100 Main Street
    Effective March 1, 2024. Closing on or before April 15, 2024.
    Purchase price $2,500,000 and earnest $125,000.
    """
    s = extract_document_structure(text, page_count=1)
    assert s["page_count"] == 1
    assert any("2024" in d for d in s["dates_found"])
    assert any("2,500,000" in m for m in s["currency_amounts"])
    assert "MEMORANDUM" in s["legal_markers"] or "AGREEMENT" in s["legal_markers"]


def test_extract_emails() -> None:
    s = extract_document_structure("Contact counsel@firm.com and copy a@b.co", page_count=1)
    assert any("counsel@firm.com" in e for e in s["email_addresses"])
