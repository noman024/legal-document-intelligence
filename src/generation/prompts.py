"""Task prompts and evidence formatting."""

from __future__ import annotations

from src.retrieval.store import RetrievedChunk


TASK_INSTRUCTIONS: dict[str, str] = {
    "case_fact_summary": (
        "Produce a concise case fact summary for internal review. "
        "Focus on parties, dates, key obligations, and disputed facts only where supported."
    ),
    "document_summary": (
        "Summarize the document: purpose, key terms, and notable clauses, grounded in the text."
    ),
    "title_review_summary": (
        "Draft a title review style summary: property/legal description references, "
        "exceptions, and schedule items that appear in the evidence."
    ),
    "internal_memo": (
        "Write a short internal memo suitable for a partner skim: background, key points, "
        "risks called out in the materials, and open questions."
    ),
    "notice_summary": (
        "Summarize notice-related content: what was noticed, to whom, deadlines, and statutory "
        "references if present in the evidence."
    ),
    "document_checklist": (
        "Produce a checklist of document types or items mentioned in the evidence, with brief notes."
    ),
}


def format_evidence_blocks(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        meta = ch.metadata or {}
        page = meta.get("page", "?")
        doc = meta.get("doc_id", "?")
        lines.append(f"[{i}] (doc={doc}, page={page}, chunk_id={ch.chunk_id})")
        lines.append(ch.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def build_system_prompt(task: str, few_shot_block: str | None = None) -> str:
    instr = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS["document_summary"])
    base = (
        "You are a drafting assistant for a law firm. You must only state facts and conclusions "
        "that are supported by the numbered evidence blocks below. If something is unclear or not "
        "in the evidence, say so explicitly.\n\n"
        f"Task: {instr}\n\n"
        "Rules:\n"
        "- After each paragraph or bullet, add inline citations like [1] or [1][2] referring ONLY "
        "to evidence indices provided.\n"
        "- Do not invent party names, dates, or citations not present in the evidence.\n"
        "- Use clear headings and bullet points where appropriate.\n"
    )
    if few_shot_block:
        base += "\nOperator preferences from past edits (follow tone/structure when consistent with evidence):\n"
        base += few_shot_block + "\n"
    return base


def build_user_prompt(query: str | None, evidence_block: str) -> str:
    q = (query or "").strip()
    parts = ["Evidence (only trustworthy source):\n", evidence_block]
    if q:
        parts.insert(0, f"User focus / question: {q}\n\n")
    parts.append("\n\nDraft your output now.")
    return "".join(parts)
