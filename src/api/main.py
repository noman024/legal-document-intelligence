"""FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from src.config import settings
from src.logging_setup import configure_app_logging

configure_app_logging()

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.feedback.learn import FeedbackLearner
from src.feedback.repository import FeedbackRepository
from src.generation.drafter import Drafter
from src.ingestion.pipeline import ingest_pdf_bytes
from src.retrieval.store import VectorStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _PROJECT_ROOT / "static"

_store: VectorStore | None = None
_repo: FeedbackRepository | None = None
_learner: FeedbackLearner | None = None
_drafter: Drafter | None = None


def get_repo() -> FeedbackRepository:
    global _repo
    if _repo is None:
        settings.feedback_db_path.parent.mkdir(parents=True, exist_ok=True)
        _repo = FeedbackRepository()
    return _repo


def get_store() -> VectorStore:
    global _store
    if _store is None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        _store = VectorStore()
    return _store


def get_learner() -> FeedbackLearner:
    global _learner
    if _learner is None:
        _learner = FeedbackLearner(get_repo())
    return _learner


def get_drafter() -> Drafter:
    global _drafter
    if _drafter is None:
        _drafter = Drafter(get_store(), get_learner())
    return _drafter


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    get_drafter()
    logger.info("Startup complete; Ollama model=%s", settings.ollama_model)
    yield


app = FastAPI(
    title="Legal document intelligence",
    description="Grounded retrieval + drafting + operator feedback loop",
    version="0.1.0",
    lifespan=lifespan,
)

if _STATIC_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico() -> RedirectResponse:
    """Browsers request /favicon.ico by default; serve the SVG under /ui/."""
    if _STATIC_DIR.is_dir() and (_STATIC_DIR / "favicon.svg").is_file():
        return RedirectResponse(url="/ui/favicon.svg", status_code=307)
    return RedirectResponse(url="/docs", status_code=302)


@app.get("/")
def root() -> RedirectResponse:
    """Browser entry: simple operator UI when `static/` is present; otherwise use `/docs`."""
    if _STATIC_DIR.is_dir():
        return RedirectResponse(url="/ui/", status_code=302)
    return RedirectResponse(url="/docs", status_code=302)


class DraftRequest(BaseModel):
    task: str = Field(..., description="Task type, e.g. case_fact_summary")
    query: str | None = Field(None, description="Optional focus question")
    doc_id_filter: str | None = Field(None, description="Limit retrieval to one doc_id")


class DraftResponse(BaseModel):
    draft_id: str
    text: str
    task: str
    query: str | None
    evidence: list[dict[str, Any]]
    citations_valid: list[int]
    citations_invalid: list[int]


class FeedbackRequest(BaseModel):
    draft_id: str
    edited_text: str


class FeedbackResponse(BaseModel):
    ok: bool
    diff_summary: str
    correction_bullets: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "legal-doc-intelligence", "version": app.version}


@app.get("/health/ready", summary="Check Ollama reachability and configured model")
def health_ready() -> dict[str, Any]:
    """Use for load balancers / orchestration: verifies the LLM backend is up."""
    base = str(settings.ollama_base_url).rstrip("/")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=5.0)
        r.raise_for_status()
        payload = r.json()
        names = [m.get("name", "") for m in payload.get("models", [])]
        want = settings.ollama_model
        model_ok = any(
            n == want or n.startswith(f"{want}:") or n.split(":")[0].endswith(want.split("/")[-1])
            for n in names
        )
        return {
            "status": "ready" if model_ok else "degraded",
            "ollama": "reachable",
            "configured_model": want,
            "model_available": model_ok,
            "installed_models_sample": names[:12],
        }
    except Exception as e:
        logger.warning("Ollama health check failed: %s", e)
        return {
            "status": "not_ready",
            "ollama": "unreachable",
            "configured_model": settings.ollama_model,
            "error": str(e),
        }


@app.post("/ingest", summary="Upload and index a PDF")
async def ingest(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a .pdf file")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        chunks = ingest_pdf_bytes(data, filename=file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    n = get_store().upsert_chunks(chunks)
    doc_id = chunks[0].doc_id if chunks else None
    structured = (chunks[0].extra or {}).get("structured") if chunks else None
    return {"indexed_chunks": n, "doc_id": doc_id, "filename": file.filename, "structured": structured}


@app.post("/draft", response_model=DraftResponse)
def draft(req: DraftRequest) -> DraftResponse:
    try:
        d = get_drafter().draft(
            req.task,
            query=req.query,
            doc_id_filter=req.doc_id_filter,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    draft_id = get_repo().save_draft(
        task=req.task,
        query=req.query,
        draft_text=d.text,
        evidence=d.evidence,
    )
    return DraftResponse(
        draft_id=draft_id,
        text=d.text,
        task=d.task,
        query=d.query,
        evidence=d.evidence,
        citations_valid=d.citations_valid,
        citations_invalid=d.citations_invalid,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    try:
        info = get_learner().record_operator_edit(req.draft_id, req.edited_text)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return FeedbackResponse(
        ok=True,
        diff_summary=info.get("diff_summary", ""),
        correction_bullets=info.get("correction_bullets", []),
    )
