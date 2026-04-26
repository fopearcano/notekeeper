"""Tests for ``LMStudioProvider`` against an injected ``httpx.MockTransport``."""

from __future__ import annotations

import asyncio
import json
from typing import Callable

import httpx
import pytest

from app.config.settings import LMStudioLLMSettings
from app.llm.lmstudio_provider import LMStudioProvider


def _build(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
    model: str = "test-model",
):
    settings = LMStudioLLMSettings(
        base_url=base_url, api_key=api_key, model=model, timeout_seconds=10
    )
    provider = LMStudioProvider(settings)
    provider._transport = httpx.MockTransport(handler)
    return provider


# ---------- non-streaming ---------------------------------------------------


def test_complete_returns_message_content():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"total_tokens": 12},
            },
        )

    provider = _build(handler)
    response = asyncio.run(provider.complete(system="sys", user="usr"))
    asyncio.run(provider.aclose())

    assert response.text == "hello"
    assert response.model == "test-model"
    assert response.usage == {"total_tokens": 12}
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


# ---------- streaming -------------------------------------------------------


def _sse_body(deltas: list[str]) -> bytes:
    """Build an OpenAI-style SSE body terminated by [DONE]."""
    lines = []
    for delta in deltas:
        chunk = {"choices": [{"delta": {"content": delta}}]}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def test_stream_complete_yields_concatenated_deltas():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["stream"] is True
        return httpx.Response(
            200,
            content=_sse_body(["Hello", ", ", "world", "!"]),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = _build(handler)

    async def _drive():
        out = []
        async for delta in provider.stream_complete(system="s", user="u"):
            out.append(delta)
        return out

    deltas = asyncio.run(_drive())
    asyncio.run(provider.aclose())

    assert "".join(deltas) == "Hello, world!"


def test_stream_complete_skips_unparseable_lines():
    """Comment / keepalive / malformed JSON lines must be ignored, not crash."""

    def handler(request: httpx.Request) -> httpx.Response:
        body_lines = [
            ":\n\n",  # SSE keepalive comment
            "data: not-json\n\n",
            'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        return httpx.Response(
            200,
            content="".join(body_lines).encode(),
            headers={"Content-Type": "text/event-stream"},
        )

    provider = _build(handler)

    async def _drive():
        return [d async for d in provider.stream_complete(system="s", user="u")]

    deltas = asyncio.run(_drive())
    asyncio.run(provider.aclose())

    assert deltas == ["ok"]


# ---------- list_models -----------------------------------------------------


def test_list_models_returns_ids():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url).endswith("/v1/models")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gemma-3-27b", "object": "model"},
                    {"id": "qwen2.5-coder", "object": "model"},
                ]
            },
        )

    provider = _build(handler)
    models = asyncio.run(provider.list_models())
    asyncio.run(provider.aclose())
    assert models == ["gemma-3-27b", "qwen2.5-coder"]


def test_list_models_handles_empty_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    provider = _build(handler)
    assert asyncio.run(provider.list_models()) == []
    asyncio.run(provider.aclose())


def test_list_models_propagates_http_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="bad gateway")

    provider = _build(handler)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(provider.list_models())
    asyncio.run(provider.aclose())
