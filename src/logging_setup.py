"""Application logging: primary file `logs/log.log` plus stderr."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_configured = False


def _pytest_active() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or ("pytest" in sys.modules)


def _root_has_file_for_path(root: logging.Logger, path: Path) -> bool:
    resolved = path.resolve()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            p = getattr(h, "baseFilename", None)
            if p and Path(p).resolve() == resolved:
                return True
    return False


def _root_has_stderr_stream(root: logging.Logger) -> bool:
    """FileHandler subclasses StreamHandler; only count true stderr streams."""
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            if getattr(h, "stream", None) is sys.stderr:
                return True
    return False


def configure_app_logging() -> None:
    """Attach file + stderr handlers to the root logger; route uvicorn loggers through root."""
    global _configured
    if _configured:
        return
    from src.config import settings

    log_path = Path(settings.log_file)
    pytest_running = _pytest_active()
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not pytest_running:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not _root_has_file_for_path(root, log_path):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            fh.setLevel(logging.INFO)
            root.addHandler(fh)

    if not _root_has_stderr_stream(root):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        root.addHandler(sh)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.setLevel(logging.INFO)
        lg.propagate = True

    # HF / embedding stack is very chatty at INFO (every HEAD/GET). Keep app logs readable.
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
