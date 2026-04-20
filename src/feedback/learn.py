"""Derive few-shot preferences from operator edits."""

from __future__ import annotations

import difflib
import logging
from typing import Any

from src.config import settings
from src.feedback.repository import FeedbackRepository
from src.retrieval.embedder import get_model

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    return dot / ((na ** 0.5) * (nb ** 0.5) + 1e-9)


def summarize_diff(original: str, edited: str, max_lines: int = 8) -> str:
    a = original.splitlines()
    b = edited.splitlines()
    diff = difflib.unified_diff(a, b, lineterm="", n=2)
    lines = list(diff)[2:]  # drop file headers
    return "\n".join(lines[:max_lines])


def extract_correction_bullets(original: str, edited: str, max_bullets: int = 5) -> list[str]:
    """Line-level diff: reusable operator preferences without character-level noise."""
    if len(original) > 400 and len(edited) < max(120, len(original) * 0.22):
        return [
            "Edit is much shorter than the original draft (possible placeholder or full replace); "
            "see diff_summary. For line-level tips, submit a full revised draft."
        ]
    a = original.splitlines()
    b = edited.splitlines()
    sm = difflib.SequenceMatcher(None, a, b)
    bullets: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old_lines = " ".join(a[i1:i2]).strip()
        new_lines = " ".join(b[j1:j2]).strip()
        if tag == "insert" and new_lines:
            bullets.append(f"Added: {new_lines[:280]}")
        elif tag == "delete" and old_lines:
            bullets.append(f"Removed: {old_lines[:280]}")
        elif tag == "replace":
            if new_lines or old_lines:
                bullets.append(f"Prefer: {new_lines[:220]} (was: {old_lines[:160]})")
        if len(bullets) >= max_bullets:
            break
    return bullets


def build_few_shot_pair(original: str, edited: str) -> str:
    """Compact before/after for prompt injection."""
    o_full, e_full = original.strip(), edited.strip()
    if len(o_full) > 400 and len(e_full) < 200:
        o = o_full[:500]
        e = e_full[:500]
        return f"Before (excerpt):\n{o}\n\nAfter (short operator note):\n{e}"
    o = o_full[:600]
    e = e_full[:600]
    return f"Before:\n{o}\n\nAfter (preferred):\n{e}"


class FeedbackLearner:
    def __init__(self, repo: FeedbackRepository | None = None) -> None:
        self._repo = repo or FeedbackRepository()

    def record_operator_edit(
        self,
        draft_id: str,
        edited_text: str,
    ) -> dict[str, Any]:
        rec = self._repo.get_draft(draft_id)
        if not rec:
            raise ValueError(f"Unknown draft_id: {draft_id}")
        diff_summary = summarize_diff(rec.draft_text, edited_text)
        bullets = extract_correction_bullets(rec.draft_text, edited_text)
        pair = build_few_shot_pair(rec.draft_text, edited_text)
        self._repo.save_edit(
            draft_id,
            edited_text,
            diff_summary=diff_summary or "(no textual diff)",
            few_shot_pair=pair,
        )
        return {
            "diff_summary": diff_summary,
            "correction_bullets": bullets,
            "few_shot_pair": pair,
        }

    def get_augmentation_block(self, task: str, query: str) -> str | None:
        rows = self._repo.list_edits_for_task(task, limit=settings.feedback_few_shot_limit * 3)
        if not rows:
            return None
        limit = settings.feedback_few_shot_limit
        if len(rows) <= limit:
            selected = rows[:limit]
        else:
            selected = self._rank_by_similarity(query, task, rows, limit)
        parts = []
        for i, r in enumerate(selected, start=1):
            pair = r.get("few_shot_pair") or ""
            parts.append(f"Example {i}:\n{pair}")
        return "\n\n".join(parts)

    def _rank_by_similarity(
        self,
        query: str,
        task: str,
        rows: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            model = get_model()
            target = f"{task} {query}".strip()
            t_vec = model.encode([target], convert_to_numpy=True)[0]
            t_list: list[float] = t_vec.tolist() if hasattr(t_vec, "tolist") else list(t_vec)
            texts = [f"{task} {(r.get('query') or '')} {r.get('few_shot_pair', '')[:400]}" for r in rows]
            embs = model.encode(texts, convert_to_numpy=True)
            scored: list[tuple[float, int]] = []
            for idx, row in enumerate(rows):
                v = embs[idx]
                v_list: list[float] = v.tolist() if hasattr(v, "tolist") else list(v)
                scored.append((_cosine_similarity(v_list, t_list), idx))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [rows[i] for _, i in scored[:limit]]
        except Exception as e:
            logger.warning("Similarity ranking failed, using recency: %s", e)
            return rows[:limit]
