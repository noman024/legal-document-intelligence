"""Prompt builder and evidence formatting."""

from __future__ import annotations

import pytest

from src.generation.prompts import (
    TASK_INSTRUCTIONS,
    build_system_prompt,
    build_user_prompt,
    format_evidence_blocks,
)
from src.retrieval.store import RetrievedChunk


ALL_TASKS = [
    "case_fact_summary",
    "document_summary",
    "title_review_summary",
    "internal_memo",
    "notice_summary",
    "document_checklist",
]


def test_task_instructions_cover_required_tasks() -> None:
    """Supported task types must have instructions in TASK_INSTRUCTIONS."""
    for t in ALL_TASKS:
        assert t in TASK_INSTRUCTIONS
        assert TASK_INSTRUCTIONS[t].strip()


@pytest.mark.parametrize("task", ALL_TASKS)
def test_system_prompt_mentions_citations_and_task(task: str) -> None:
    p = build_system_prompt(task)
    assert "[1]" in p
    assert "only state facts" in p.lower() or "supported" in p.lower()


def test_system_prompt_unknown_task_falls_back() -> None:
    fallback = build_system_prompt("not_a_real_task")
    default = build_system_prompt("document_summary")
    # Should produce the same instruction (mapping falls back to document_summary).
    assert TASK_INSTRUCTIONS["document_summary"] in fallback
    assert TASK_INSTRUCTIONS["document_summary"] in default


def test_system_prompt_injects_few_shot_block() -> None:
    block = "Example 1: prefer short bullets"
    p = build_system_prompt("internal_memo", few_shot_block=block)
    assert block in p
    assert "Operator preferences" in p


def test_user_prompt_includes_evidence_and_query() -> None:
    user = build_user_prompt("focus on dates", "EVIDENCE_HERE")
    assert "EVIDENCE_HERE" in user
    assert "focus on dates" in user
    assert "Draft your output now." in user


def test_user_prompt_without_query() -> None:
    user = build_user_prompt(None, "E")
    assert "User focus" not in user
    assert "E" in user


def test_format_evidence_blocks_numbers_sequentially() -> None:
    chunks = [
        RetrievedChunk("id-1", "text one", 0.9, {"doc_id": "d", "page": 1}),
        RetrievedChunk("id-2", "text two", 0.8, {"doc_id": "d", "page": 2}),
    ]
    out = format_evidence_blocks(chunks)
    assert "[1]" in out
    assert "[2]" in out
    assert "text one" in out
    assert "text two" in out
    assert "page=1" in out


def test_format_evidence_blocks_empty_input() -> None:
    assert format_evidence_blocks([]) == ""
