"""Ollama HTTP client: success and all error branches (no real network calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.generation.ollama_client import generate_chat


def _fake_post_response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = str(payload)
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def test_generate_chat_returns_content() -> None:
    messages = [{"role": "user", "content": "hi"}]
    response = _fake_post_response({"message": {"role": "assistant", "content": "hello!"}})
    with patch("src.generation.ollama_client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = response
        out = generate_chat(messages, model="m", base_url="http://x")
    assert out == "hello!"


def test_generate_chat_empty_content_returns_empty_string() -> None:
    response = _fake_post_response({"message": {"role": "assistant", "content": ""}})
    with patch("src.generation.ollama_client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = response
        out = generate_chat([{"role": "user", "content": "hi"}])
    assert out == ""


def test_generate_chat_missing_message_returns_empty_string() -> None:
    response = _fake_post_response({"unexpected": "payload"})
    with patch("src.generation.ollama_client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = response
        out = generate_chat([{"role": "user", "content": "hi"}])
    assert out == ""


def test_generate_chat_connect_error_raises_runtime_error() -> None:
    with patch("src.generation.ollama_client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError(
            "refused"
        )
        with pytest.raises(RuntimeError, match="Cannot reach Ollama"):
            generate_chat([{"role": "user", "content": "x"}])


def test_generate_chat_http_status_error_raises_runtime_error() -> None:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 500
    resp.text = "internal"

    def _raise() -> None:
        raise httpx.HTTPStatusError("boom", request=MagicMock(), response=resp)

    response = MagicMock()
    response.raise_for_status.side_effect = _raise
    with patch("src.generation.ollama_client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = response
        with pytest.raises(RuntimeError, match="Ollama returned HTTP 500"):
            generate_chat([{"role": "user", "content": "x"}])


def test_generate_chat_timeout_raises_runtime_error() -> None:
    with patch("src.generation.ollama_client.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.side_effect = httpx.ReadTimeout(
            "slow"
        )
        with pytest.raises(RuntimeError, match="timed out"):
            generate_chat([{"role": "user", "content": "x"}])
