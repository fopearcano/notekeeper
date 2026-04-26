"""Connection-tester probes."""

from __future__ import annotations

import asyncio
from typing import Callable

import httpx
import pytest

from app.config.settings import load_settings
from app.services import connection_tester
from app.services.connection_tester import (
    ProbeResult,
    probe_anthropic,
    probe_lmstudio,
    probe_openai,
    probe_transcription,
)


def _settings(**overrides):
    return load_settings(bootstrap=False, overrides=overrides)


def _patched_async_client(handler: Callable[[httpx.Request], httpx.Response]):
    """Return a context manager that injects ``MockTransport(handler)``."""
    from contextlib import asynccontextmanager

    class _Patcher:
        def __init__(self):
            self._original = httpx.AsyncClient

        def install(self, monkeypatch):
            transport = httpx.MockTransport(handler)

            class _PatchedClient(self._original):
                def __init__(self_inner, *args, **kwargs):
                    kwargs["transport"] = transport
                    super().__init__(*args, **kwargs)

            monkeypatch.setattr(connection_tester.httpx, "AsyncClient", _PatchedClient)

    return _Patcher()


# ---------- transcription --------------------------------------------------


def test_transcription_faster_whisper_importable(monkeypatch):
    """When the import hook returns a class, the probe reports success."""
    from app.transcription import faster_whisper_provider as fw_mod

    class _FakeWhisper:
        pass

    monkeypatch.setattr(fw_mod, "_import_whisper_model", lambda: _FakeWhisper)
    settings = _settings(transcription={"provider": "faster_whisper"})
    result = asyncio.run(probe_transcription(settings))
    assert result.ok is True
    assert "_FakeWhisper" in result.message or "WhisperModel" in result.message


def test_transcription_faster_whisper_import_error(monkeypatch):
    from app.transcription import faster_whisper_provider as fw_mod

    def _boom():
        raise ImportError("no faster-whisper installed")

    monkeypatch.setattr(fw_mod, "_import_whisper_model", _boom)
    settings = _settings(transcription={"provider": "faster_whisper"})
    result = asyncio.run(probe_transcription(settings))
    assert result.ok is False
    assert "not importable" in result.message


def test_transcription_lmstudio_audio_returns_stub_warning():
    settings = _settings(transcription={"provider": "lmstudio_audio"})
    result = asyncio.run(probe_transcription(settings))
    assert result.ok is False
    assert "stub" in result.message.lower()


def test_transcription_openai_audio_uses_models_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"data": [{"id": "whisper-1", "object": "model"}]})

    _patched_async_client(handler).install(monkeypatch)
    settings = _settings(transcription={"provider": "openai_audio"})
    result = asyncio.run(probe_transcription(settings))

    assert result.ok is True
    assert "whisper-1" in result.message
    assert captured["auth"] == "Bearer sk-test"
    assert "/models" in captured["url"]


# ---------- LM Studio ------------------------------------------------------


def test_lmstudio_ok_lists_models(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "qwen"}, {"id": "gemma"}, {"id": "mixtral"}, {"id": "llama"}]},
        )

    _patched_async_client(handler).install(monkeypatch)
    result = asyncio.run(probe_lmstudio(_settings()))
    assert result.ok is True
    assert "4 model(s)" in result.message
    assert "+1 more" in result.message  # only first three names listed inline


def test_lmstudio_network_error_reported(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _patched_async_client(handler).install(monkeypatch)
    result = asyncio.run(probe_lmstudio(_settings()))
    assert result.ok is False
    assert "failed" in result.message.lower()


# ---------- OpenAI ---------------------------------------------------------


def test_openai_no_key_reported(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = asyncio.run(probe_openai(_settings()))
    assert result.ok is False
    assert "OPENAI_API_KEY" in result.message


def test_openai_auth_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "bad")

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "auth"})

    _patched_async_client(handler).install(monkeypatch)
    result = asyncio.run(probe_openai(_settings()))
    assert result.ok is False
    assert "authentication failed" in result.message.lower()


def test_openai_404(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk")

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    _patched_async_client(handler).install(monkeypatch)
    result = asyncio.run(probe_openai(_settings()))
    assert result.ok is False
    assert "404" in result.message


# ---------- Anthropic ------------------------------------------------------


def test_anthropic_no_key_reported(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = asyncio.run(probe_anthropic(_settings()))
    assert result.ok is False
    assert "ANTHROPIC_API_KEY" in result.message


def test_anthropic_ok_lists_models(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["url"] = str(request.url)
        return httpx.Response(
            200, json={"data": [{"id": "claude-sonnet-4-5"}, {"id": "claude-haiku"}]}
        )

    _patched_async_client(handler).install(monkeypatch)
    result = asyncio.run(probe_anthropic(_settings()))

    assert result.ok is True
    assert "claude-sonnet-4-5" in result.message
    assert captured["headers"].get("x-api-key") == "secret"
    assert captured["headers"].get("anthropic-version")
    assert "/models" in captured["url"]


def test_anthropic_unexpected_shape_is_reported(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")

    def handler(_req):
        return httpx.Response(200, json={"weird": "payload"})

    _patched_async_client(handler).install(monkeypatch)
    result = asyncio.run(probe_anthropic(_settings()))
    assert result.ok is False
    assert "shape" in result.message.lower()


def test_probe_result_dataclass_is_immutable():
    r = ProbeResult(True, "x")
    with pytest.raises(Exception):
        r.message = "y"  # type: ignore[misc]
