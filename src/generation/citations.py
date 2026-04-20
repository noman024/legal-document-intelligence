"""Parse [n] style citations from draft text."""

from __future__ import annotations

import re


_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_cited_indices(text: str) -> list[int]:
    found = sorted({int(m.group(1)) for m in _CITATION_RE.finditer(text)})
    return found


def validate_citations(text: str, max_evidence: int) -> tuple[list[int], list[int]]:
    """
    Returns (valid_indices, out_of_range_indices).
    """
    cited = extract_cited_indices(text)
    valid = [i for i in cited if 1 <= i <= max_evidence]
    bad = [i for i in cited if i < 1 or i > max_evidence]
    return valid, bad
