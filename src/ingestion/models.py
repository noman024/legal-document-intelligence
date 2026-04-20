from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A text segment indexed for retrieval."""

    text: str
    doc_id: str
    chunk_index: int
    page: int | None = None
    source_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def stable_id(self) -> str:
        base = f"{self.doc_id}:{self.chunk_index}"
        if self.page is not None:
            return f"{base}:p{self.page}"
        return base
