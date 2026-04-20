"""Settings: defaults, env-var overrides, log path anchoring."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_defaults_cover_required_keys() -> None:
    from src.config import settings

    assert settings.ollama_model
    assert settings.ollama_base_url
    assert settings.embedding_model
    assert settings.chroma_collection
    assert settings.chunk_size > 0
    assert 0 <= settings.chunk_overlap < settings.chunk_size
    assert settings.retrieve_top_k > 0
    assert settings.ocr_min_chars_per_page >= 0
    assert settings.feedback_few_shot_limit > 0


def test_env_prefix_overrides(monkeypatch) -> None:
    monkeypatch.setenv("LEGAL_OLLAMA_MODEL", "mistral-test")
    monkeypatch.setenv("LEGAL_CHUNK_SIZE", "111")
    import src.config as cfg

    importlib.reload(cfg)
    try:
        assert cfg.settings.ollama_model == "mistral-test"
        assert cfg.settings.chunk_size == 111
    finally:
        # Reset so other tests see originals.
        monkeypatch.delenv("LEGAL_OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("LEGAL_CHUNK_SIZE", raising=False)
        importlib.reload(cfg)


def test_log_file_is_anchored_to_project_root() -> None:
    from src.config import settings

    assert settings.log_file.is_absolute()
    # The log file sits under the repo root (parent of src/).
    root = Path(__file__).resolve().parents[1]
    assert str(settings.log_file).startswith(str(root))
