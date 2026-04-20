import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def deterministic_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """384-dim fake vectors (MiniLM size) so Chroma upsert/query works offline."""
    dim = 384

    def embed_texts(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i, t in enumerate(texts):
            v = [0.0] * dim
            h = hash(t) & 0xFFFFFFFF
            for j in range(dim):
                v[j] = float((h + i * 17 + j * 31) % 997) / 997.0
            out.append(v)
        return out

    def embed_query(text: str) -> list[float]:
        return embed_texts([text])[0]

    # Patch both modules: `store` binds imports at load time.
    for mod in ("src.retrieval.embedder", "src.retrieval.store"):
        monkeypatch.setattr(f"{mod}.embed_texts", embed_texts)
        monkeypatch.setattr(f"{mod}.embed_query", embed_query)
