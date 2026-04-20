"""Chroma-backed vector store for legal chunks."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.ingestion.models import Chunk
from src.retrieval.embedder import embed_query, embed_texts

logger = logging.getLogger(__name__)


def dedupe_retrieved_by_text(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop near-duplicate retrievals (same body text from repeated ingests)."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for ch in chunks:
        key = hashlib.sha256(ch.text.strip().encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(ch)
    return out


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float | None
    metadata: dict[str, Any]


class VectorStore:
    def __init__(self, persist_path: str | None = None) -> None:
        path = str(persist_path or settings.chroma_path)
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        ids = [c.stable_id() for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        metadatas = []
        for c in chunks:
            meta = {
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "source_path": c.source_path or "",
            }
            if c.page is not None:
                meta["page"] = c.page
            sj = (c.extra or {}).get("structured_json")
            if sj:
                meta["structured_json"] = sj[:4000]
            metadatas.append(meta)
        self._collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info("Indexed %s chunks", len(chunks))
        return len(chunks)

    def query(
        self,
        query_text: str,
        n_results: int | None = None,
        doc_id_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        k = n_results if n_results is not None else settings.retrieve_top_k
        try:
            if self._collection.count() == 0:
                return []
        except Exception as e:
            logger.warning("Could not read collection size: %s", e)
        emb = embed_query(query_text)
        where: dict[str, Any] | None = None
        if doc_id_filter:
            where = {"doc_id": doc_id_filter}

        mult = max(1, int(settings.retrieve_dedup_multiplier))
        fetch_k = min(k * mult, 64)

        res = self._collection.query(
            query_embeddings=[emb],
            n_results=fetch_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        dists = res.get("distances") or []
        if not ids or not ids[0]:
            return out
        for i, cid in enumerate(ids[0]):
            text = (docs[0][i] if docs and docs[0] else "") or ""
            meta = dict((metas[0][i] if metas and metas[0] else {}) or {})
            dist = None
            if dists and dists[0] and i < len(dists[0]):
                dist = float(dists[0][i])
            score = 1.0 - dist if dist is not None else None
            out.append(RetrievedChunk(chunk_id=cid, text=text, score=score, metadata=meta))
        out = dedupe_retrieved_by_text(out)
        return out[:k]
