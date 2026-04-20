"""Application settings."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load `.env` into the process environment so `HF_TOKEN` (and other non-LEGAL_ vars) are
# visible to huggingface_hub before any model download.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEGAL_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    chroma_path: Path = Path("data/chroma")
    feedback_db_path: Path = Path("data/feedback.db")
    uploads_dir: Path = Path("data/uploads")

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_collection: str = "legal_chunks"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"

    chunk_size: int = 900
    chunk_overlap: int = 120
    retrieve_top_k: int = 8
    #: Fetch `top_k * multiplier` hits then dedupe identical chunk text (repeated ingests).
    retrieve_dedup_multiplier: int = 2

    ocr_min_chars_per_page: int = 50
    #: Max frames/pages to OCR from a multi-page TIFF (or animated image).
    ingest_max_image_frames: int = 48
    feedback_few_shot_limit: int = 5
    #: When True, POST /draft returns 422 if the model cites evidence indices outside retrieval.
    strict_citations: bool = False

    log_file: Path = Path("logs/log.log")

    @field_validator("log_file", mode="after")
    @classmethod
    def _anchor_log_file_to_project_root(cls, v: Path) -> Path:
        """Relative paths resolve against the repo root so logging works regardless of process cwd."""
        if v.is_absolute():
            return v
        return _PROJECT_ROOT / v


settings = Settings()
