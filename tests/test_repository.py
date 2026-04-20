"""SQLite feedback store: schema, CRUD, and referential integrity."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.feedback.repository import DraftRecord, FeedbackRepository


def test_init_db_creates_tables() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "schema.db"
        FeedbackRepository(db)
        conn = sqlite3.connect(db)
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            names = {r[0] for r in cur.fetchall()}
        finally:
            conn.close()
        assert "drafts" in names
        assert "edits" in names


def test_save_get_draft_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        r = FeedbackRepository(Path(td) / "r.db")
        eid = r.save_draft(
            "notice_summary",
            "deadlines",
            "draft body",
            [{"index": 1, "text": "ev"}],
        )
        rec = r.get_draft(eid)
        assert isinstance(rec, DraftRecord)
        assert rec.task == "notice_summary"
        assert rec.query == "deadlines"
        assert rec.draft_text == "draft body"
        assert rec.evidence == [{"index": 1, "text": "ev"}]


def test_get_draft_missing_returns_none() -> None:
    with tempfile.TemporaryDirectory() as td:
        r = FeedbackRepository(Path(td) / "m.db")
        assert r.get_draft("00000000-0000-0000-0000-000000000000") is None


def test_save_edit_rejects_unknown_draft_id() -> None:
    with tempfile.TemporaryDirectory() as td:
        r = FeedbackRepository(Path(td) / "fk.db")
        with pytest.raises(sqlite3.IntegrityError):
            r.save_edit(
                "not-a-saved-draft-id",
                "edited",
                diff_summary="x",
                few_shot_pair="y",
            )


def test_save_edit_after_draft() -> None:
    with tempfile.TemporaryDirectory() as td:
        r = FeedbackRepository(Path(td) / "e.db")
        did = r.save_draft("internal_memo", None, "v1", [])
        r.save_edit(did, "v2", diff_summary="d", few_shot_pair="f")
        rows = r.list_edits_for_task("internal_memo", limit=5)
        assert len(rows) == 1
        assert rows[0]["edited_text"] == "v2"


def test_list_edits_for_task_orders_newest_first() -> None:
    with tempfile.TemporaryDirectory() as td:
        r = FeedbackRepository(Path(td) / "order.db")
        d1 = r.save_draft("document_summary", "q1", "a", [])
        d2 = r.save_draft("document_summary", "q2", "b", [])
        r.save_edit(d1, "edit-old", diff_summary="1", few_shot_pair="p1")
        r.save_edit(d2, "edit-new", diff_summary="2", few_shot_pair="p2")
        rows = r.list_edits_for_task("document_summary", limit=10)
        assert [row["edited_text"] for row in rows] == ["edit-new", "edit-old"]
