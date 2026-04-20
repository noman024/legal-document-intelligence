"""Retrieve → prompt → Ollama → structured result."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.config import settings
from src.feedback.learn import FeedbackLearner
from src.generation.citations import validate_citations
from src.generation.ollama_client import generate_chat
from src.generation.prompts import build_system_prompt, build_user_prompt, format_evidence_blocks
from src.retrieval.store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


def _structured_from_metadata(meta: dict[str, Any]) -> dict[str, Any] | None:
    raw = meta.get("structured_json")
    if not raw:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError:
        return None


@dataclass
class DraftResult:
    text: str
    task: str
    query: str | None
    evidence: list[dict[str, Any]]
    citations_valid: list[int]
    citations_invalid: list[int]


class Drafter:
    def __init__(
        self,
        store: VectorStore,
        learner: FeedbackLearner | None = None,
    ) -> None:
        self._store = store
        self._learner = learner

    def draft(
        self,
        task: str,
        query: str | None = None,
        *,
        doc_id_filter: str | None = None,
        top_k: int | None = None,
    ) -> DraftResult:
        k = top_k if top_k is not None else settings.retrieve_top_k
        retrieval_query = " ".join(x for x in [task, query or ""] if x).strip()
        chunks = self._store.query(retrieval_query, n_results=k, doc_id_filter=doc_id_filter)

        if not chunks:
            msg = (
                "No indexed document chunks matched this request. "
                "Ingest at least one PDF (POST /ingest or the ingest CLI) and try again. "
                "If you used doc_id_filter, confirm that document was indexed."
            )
            logger.warning("Draft skipped LLM: empty retrieval (task=%s, filter=%s)", task, doc_id_filter)
            return DraftResult(
                text=msg,
                task=task,
                query=query,
                evidence=[],
                citations_valid=[],
                citations_invalid=[],
            )

        evidence_block = format_evidence_blocks(chunks)
        few_shot = None
        if self._learner:
            few_shot = self._learner.get_augmentation_block(task, query or "")

        system = build_system_prompt(task, few_shot_block=few_shot)
        user = build_user_prompt(query, evidence_block)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = generate_chat(messages)

        valid, invalid = validate_citations(text, max_evidence=len(chunks))
        evidence_serial = [
            {
                "index": i + 1,
                "chunk_id": ch.chunk_id,
                "doc_id": ch.metadata.get("doc_id"),
                "page": ch.metadata.get("page"),
                "score": ch.score,
                "text": ch.text,
                "structured": _structured_from_metadata(ch.metadata),
            }
            for i, ch in enumerate(chunks)
        ]
        return DraftResult(
            text=text,
            task=task,
            query=query,
            evidence=evidence_serial,
            citations_valid=valid,
            citations_invalid=invalid,
        )
