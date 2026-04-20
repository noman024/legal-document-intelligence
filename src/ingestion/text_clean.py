"""Normalize extracted text for downstream chunking."""

import re


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_repeated_headers_footers(pages: list[str], min_runs: int = 2) -> list[str]:
    """
    Heuristic: if the same first/last line appears on many pages, drop it.
    Best-effort for legal PDFs with running headers.
    """
    if len(pages) < 2:
        return pages

    first_lines = [p.split("\n", 1)[0].strip() for p in pages if p.strip()]
    last_lines = []
    for p in pages:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if lines:
            last_lines.append(lines[-1])

    from collections import Counter

    fc = Counter(first_lines)
    lc = Counter(last_lines)
    common_first = {line for line, c in fc.items() if c >= min_runs and len(line) < 120}
    common_last = {line for line, c in lc.items() if c >= min_runs and len(line) < 120}

    out: list[str] = []
    for p in pages:
        lines = p.split("\n")
        if lines and lines[0].strip() in common_first:
            lines = lines[1:]
        if lines and lines[-1].strip() in common_last:
            lines = lines[:-1]
        out.append("\n".join(lines))
    return out
