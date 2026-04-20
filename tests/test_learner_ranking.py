"""Similarity-ranked few-shot selection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.feedback.learn import FeedbackLearner
from src.feedback.repository import FeedbackRepository


class _FakeModel:
    """Deterministic encoder: score equals substring overlap with the target."""

    def __init__(self, keyword: str) -> None:
        self._keyword = keyword

    def encode(self, texts, convert_to_numpy=False):  # noqa: ARG002
        import numpy as np

        vecs = []
        for t in texts:
            # Simple 4-dim vector biased on presence of the keyword.
            base = 0.1 if self._keyword in t else 0.0
            vecs.append([base + 0.01, 0.0, 0.0, base])
        return np.array(vecs, dtype=float)


def test_similarity_ranking_selects_matching_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as td:
        repo = FeedbackRepository(Path(td) / "rank.db")
        learner = FeedbackLearner(repo)
        monkeypatch.setattr(
            "src.config.settings.feedback_few_shot_limit", 2, raising=False
        )
        for i in range(8):
            body = "evictions and notices" if i % 2 == 0 else "purchase agreement"
            did = repo.save_draft("notice_summary", f"q-{i} {body}", f"draft {i}: {body}", [])
            learner.record_operator_edit(did, f"edited {i}: {body} with preferences")

        monkeypatch.setattr(
            "src.feedback.learn.get_model", lambda: _FakeModel(keyword="evictions")
        )
        block = learner.get_augmentation_block("notice_summary", "evictions")
        assert block
        # Selected examples should mention eviction content, not purchase-only ones.
        assert "evictions" in block


def test_augmentation_block_none_when_no_edits() -> None:
    with tempfile.TemporaryDirectory() as td:
        repo = FeedbackRepository(Path(td) / "none.db")
        learner = FeedbackLearner(repo)
        assert learner.get_augmentation_block("internal_memo", "anything") is None


def test_record_operator_edit_unknown_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        repo = FeedbackRepository(Path(td) / "ue.db")
        learner = FeedbackLearner(repo)
        with pytest.raises(ValueError, match="Unknown draft_id"):
            learner.record_operator_edit("00000000-0000-0000-0000-000000000000", "x")
