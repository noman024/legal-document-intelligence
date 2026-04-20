"""
End-to-end demo: create sample PDF, ingest, draft (requires Ollama), submit feedback, draft again.

Run from project root with venv activated:
  python scripts/create_sample_pdf.py
  python scripts/ingest.py examples/sample_memo.pdf
  python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import settings  # noqa: E402
from src.feedback.learn import FeedbackLearner  # noqa: E402
from src.feedback.repository import FeedbackRepository  # noqa: E402
from src.generation.drafter import Drafter  # noqa: E402
from src.retrieval.store import VectorStore  # noqa: E402


def main() -> None:
    pdf = _ROOT / "examples" / "sample_memo.pdf"
    if not pdf.is_file():
        print("Run: python scripts/create_sample_pdf.py", file=sys.stderr)
        sys.exit(1)

    store = VectorStore()
    repo = FeedbackRepository()
    learner = FeedbackLearner(repo)
    drafter = Drafter(store, learner)

    task = "internal_memo"
    print("--- First draft (no prior feedback) ---")
    r1 = drafter.draft(task, query="Summarize risks and deadlines.")
    did = repo.save_draft(task, r1.query, r1.text, r1.evidence)
    print(r1.text[:1500])
    print(f"\n[draft_id={did}] citations valid: {r1.citations_valid}")

    edited = r1.text.replace("REC", "recognized environmental condition (REC)")
    if edited == r1.text:
        edited = r1.text + "\n\nNote: Operator prefers explicit defined terms on first use [1]."

    print("\n--- Recording operator edit ---")
    info = learner.record_operator_edit(did, edited)
    print("Diff summary:\n", info["diff_summary"][:800])
    print("\n--- Second draft (operator few-shots may augment the prompt) ---")
    r2 = drafter.draft(task, query="Summarize risks and deadlines.")
    did2 = repo.save_draft(task, r2.query, r2.text, r2.evidence)
    print(r2.text[:1500])
    print(f"\n[draft_id={did2}]")


if __name__ == "__main__":
    main()
