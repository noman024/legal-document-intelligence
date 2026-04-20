"""Create a small synthetic legal-style PDF for demos."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import fitz  # noqa: E402


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "examples" / "sample_memo.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    text = """
    INTERNAL MEMORANDUM — MATTER 2024-042
    Re: Purchase Agreement — 100 Main Street

    Parties: Buyer ABC LLC; Seller XYZ Corp.
    Effective date: March 1, 2024.

    Key terms: Purchase price $2,500,000; earnest money $125,000 held in escrow
    with First National Title. Closing on or before April 15, 2024.

    Title: Seller shall convey fee simple title subject only to (1) standard
    utility easements, (2) covenants recorded at Book 900 Page 12, and (3)
    municipal zoning as of the date of closing.

    Notice: Any objection to title must be delivered in writing within ten (10)
    business days of receipt of the commitment.

    Risk: Environmental Phase I noted a REC concerning former underground storage;
    further sampling recommended prior to closing.
    """
    page.insert_text((50, 72), text.strip(), fontsize=11)
    doc.save(out)
    doc.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
