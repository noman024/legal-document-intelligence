"""SQLite persistence for drafts and operator edits."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DraftRecord:
    id: str
    task: str
    query: str | None
    draft_text: str
    evidence: list[dict[str, Any]]
    created_at: str


class FeedbackRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or settings.feedback_db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS drafts (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    query TEXT,
                    draft_text TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edits (
                    draft_id TEXT NOT NULL,
                    edited_text TEXT NOT NULL,
                    diff_summary TEXT,
                    few_shot_pair TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (draft_id) REFERENCES drafts(id)
                )
                """
            )
            conn.commit()

    def save_draft(
        self,
        task: str,
        query: str | None,
        draft_text: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        did = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO drafts (id, task, query, draft_text, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    did,
                    task,
                    query,
                    draft_text,
                    json.dumps(evidence),
                    _utc_now(),
                ),
            )
            conn.commit()
        return did

    def get_draft(self, draft_id: str) -> DraftRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, task, query, draft_text, evidence_json, created_at FROM drafts WHERE id = ?",
                (draft_id,),
            ).fetchone()
        if not row:
            return None
        ev = json.loads(row[4])
        return DraftRecord(
            id=row[0],
            task=row[1],
            query=row[2],
            draft_text=row[3],
            evidence=ev,
            created_at=row[5],
        )

    def save_edit(
        self,
        draft_id: str,
        edited_text: str,
        diff_summary: str,
        few_shot_pair: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO edits (draft_id, edited_text, diff_summary, few_shot_pair, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (draft_id, edited_text, diff_summary, few_shot_pair, _utc_now()),
            )
            conn.commit()

    def list_edits_for_task(self, task: str, limit: int = 20) -> list[dict[str, Any]]:
        """Join edits with drafts filtered by task."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.edited_text, e.diff_summary, e.few_shot_pair, d.query, d.draft_text, d.created_at
                FROM edits e
                JOIN drafts d ON d.id = e.draft_id
                WHERE d.task = ?
                ORDER BY e.created_at DESC
                LIMIT ?
                """,
                (task, limit),
            ).fetchall()
        return [
            {
                "edited_text": r[0],
                "diff_summary": r[1],
                "few_shot_pair": r[2],
                "query": r[3],
                "draft_text": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]
