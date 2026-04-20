import tempfile
from pathlib import Path

from src.feedback.learn import (
    FeedbackLearner,
    build_few_shot_pair,
    extract_correction_bullets,
    summarize_diff,
)
from src.feedback.repository import FeedbackRepository


def test_diff_summary():
    s = summarize_diff("line a\nline b", "line a\nline c")
    assert "line" in s or s == ""


def test_few_shot_pair():
    p = build_few_shot_pair("old text", "new text")
    assert "Before" in p and "After" in p


def test_correction_bullets_line_level():
    bullets = extract_correction_bullets("Hello\nWorld", "Hello\nUniverse")
    assert bullets and any("Universe" in b for b in bullets)


def test_correction_bullets_short_replacement():
    long = "paragraph\n" * 80
    bullets = extract_correction_bullets(long, "short note")
    assert len(bullets) == 1
    assert "shorter" in bullets[0].lower()


def test_repo_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "t.db"
        r = FeedbackRepository(db)
        eid = r.save_draft("internal_memo", "q", "draft", [{"index": 1}])
        rec = r.get_draft(eid)
        assert rec and rec.draft_text == "draft"
        learner = FeedbackLearner(r)
        learner.record_operator_edit(eid, "draft edited")
        rows = r.list_edits_for_task("internal_memo")
        assert len(rows) == 1


def test_augmentation_falls_back_when_similarity_ranking_fails(monkeypatch):
    """When embeddings fail, few-shot block still returns recent examples."""
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "rank.db"
        r = FeedbackRepository(db)
        learner = FeedbackLearner(r)
        for i in range(8):
            did = r.save_draft("case_fact_summary", f"query-{i}", f"draft-{i}", [])
            learner.record_operator_edit(did, f"edited-{i} with more text for pairing")

        def _fail_encode(*_a, **_k):
            raise RuntimeError("embedding unavailable")

        monkeypatch.setattr("src.feedback.learn.get_model", _fail_encode)
        block = learner.get_augmentation_block("case_fact_summary", "focus")
        assert block
        assert "Example" in block
        assert "edited-" in block
