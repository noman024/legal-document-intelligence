"""Minimal Ollama HTTP client."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


def generate_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    base_url: str | None = None,
) -> str:
    url = f"{base_url or settings.ollama_base_url}/api/chat"
    payload: dict[str, Any] = {
        "model": model or settings.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url or settings.ollama_base_url}. "
            "Start Ollama and ensure the model is pulled (e.g. ollama pull llama3.2)."
        ) from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Ollama returned HTTP {e.response.status_code}: {e.response.text[:500]}"
        ) from e
    except httpx.TimeoutException as e:
        raise RuntimeError("Ollama request timed out; retry or increase timeout.") from e
    msg = data.get("message") or {}
    content = msg.get("content")
    if not content:
        logger.warning("Unexpected Ollama response: %s", json.dumps(data)[:500])
    return (content or "").strip()
