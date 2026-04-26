"""Tests for ``OpenAIAudioProvider`` using ``httpx.MockTransport``."""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Callable

import httpx
import pytest

from app.audio.audio_buffer import AudioChunk
from app.config.settings import OpenAIAudioSettings, TranscriptionSettings
from app.transcription.openai_audio_provider import (
    OpenAIAudioProvider,
    _chunk_to_wav_bytes,
)


def _silence_chunk(*, sample_rate=16000, seconds=1.0, timestamp=0.0) -> AudioChunk:
    n = int(sample_rate * seconds)
    return AudioChunk(
        data=b"\x00\x00" * n, sample_rate=sample_rate, channels=1, timestamp=timestamp
    )


def _build(handler: Callable[[httpx.Request], httpx.Response], *, language="en"):
    """Construct a provider with an injected MockTransport."""
    transcription = TranscriptionSettings(
        sample_rate=16000, chunk_seconds=1, language=language
    )
    openai_audio = OpenAIAudioSettings(
        base_url="https://example.test/v1",
        api_key_env="TEST_AUDIO_KEY",
        model="whisper-1",
    )
    provider = OpenAIAudioProvider(transcription, openai_audio)
    provider._transport = httpx.MockTransport(handler)
    return provider


async def _drive(provider: OpenAIAudioProvider, chunks: list[AudioChunk]):
    async def _iter():
        for c in chunks:
            yield c

    out = []
    await provider.start()
    try:
        async for seg in provider.stream(_iter()):
            out.append(seg)
    finally:
        await provider.stop()
    return out


# ---------- WAV helper ------------------------------------------------------


def test_chunk_to_wav_bytes_is_a_valid_wav():
    chunk = _silence_chunk(sample_rate=16000, seconds=0.5)
    wav = _chunk_to_wav_bytes(chunk)

    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # int16
        assert wf.getframerate() == 16000
        # 0.5 s × 16 kHz = 8000 frames.
        assert wf.getnframes() == 8000


# ---------- happy path ------------------------------------------------------


def test_successful_transcription_yields_text_segment(monkeypatch):
    monkeypatch.setenv("TEST_AUDIO_KEY", "sk-test")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "hello world"})

    provider = _build(handler)
    segments = asyncio.run(_drive(provider, [_silence_chunk(timestamp=2.0)]))

    assert captured["url"].endswith("/v1/audio/transcriptions")
    assert captured["auth"] == "Bearer sk-test"
    # multipart body should mention both the file and the model field.
    assert b"audio.wav" in captured["body"]
    assert b'name="model"' in captured["body"]
    assert b"whisper-1" in captured["body"]
    assert b'name="language"' in captured["body"]

    assert len(segments) == 1
    assert segments[0].text == "hello world"
    assert segments[0].start_s == pytest.approx(2.0)
    assert segments[0].metadata.get("model") == "whisper-1"
    assert segments[0].metadata.get("latency_ms", -1) >= 0


def test_language_auto_omits_field(monkeypatch):
    monkeypatch.setenv("TEST_AUDIO_KEY", "sk-test")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, json={"text": "ok"})

    provider = _build(handler, language="auto")
    asyncio.run(_drive(provider, [_silence_chunk(timestamp=0.0)]))

    assert b'name="language"' not in captured["body"]


# ---------- failure paths ---------------------------------------------------


def test_404_emits_clear_warning_and_raises_not_implemented(monkeypatch):
    monkeypatch.setenv("TEST_AUDIO_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    provider = _build(handler)
    warnings: list[str] = []
    provider.set_warning_callback(warnings.append)

    with pytest.raises(NotImplementedError) as exc_info:
        asyncio.run(_drive(provider, [_silence_chunk(timestamp=0.0)]))

    assert "does not support audio transcription" in str(exc_info.value)
    assert any("does not support audio transcription" in w for w in warnings)


@pytest.mark.parametrize("status", [401, 403])
def test_auth_error_emits_warning_and_raises_not_implemented(monkeypatch, status):
    monkeypatch.setenv("TEST_AUDIO_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "auth"})

    provider = _build(handler)
    warnings: list[str] = []
    provider.set_warning_callback(warnings.append)

    with pytest.raises(NotImplementedError):
        asyncio.run(_drive(provider, [_silence_chunk(timestamp=0.0)]))

    assert any("authentication failed" in w.lower() for w in warnings)


def test_timeout_warns_and_continues_to_next_chunk(monkeypatch):
    monkeypatch.setenv("TEST_AUDIO_KEY", "sk-test")
    seen_calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_calls["count"] += 1
        if seen_calls["count"] == 1:
            raise httpx.ReadTimeout("upstream slow", request=request)
        return httpx.Response(200, json={"text": "second chunk"})

    provider = _build(handler)
    warnings: list[str] = []
    provider.set_warning_callback(warnings.append)

    segments = asyncio.run(
        _drive(
            provider,
            [_silence_chunk(timestamp=0.0), _silence_chunk(timestamp=1.0)],
        )
    )

    assert seen_calls["count"] == 2
    assert any("timed out" in w.lower() for w in warnings)
    assert [s.text for s in segments] == ["second chunk"]


def test_5xx_warns_and_continues(monkeypatch):
    monkeypatch.setenv("TEST_AUDIO_KEY", "sk-test")
    state = {"first": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["first"]:
            state["first"] = False
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json={"text": "recovered"})

    provider = _build(handler)
    warnings: list[str] = []
    provider.set_warning_callback(warnings.append)

    segments = asyncio.run(
        _drive(
            provider,
            [_silence_chunk(timestamp=0.0), _silence_chunk(timestamp=1.0)],
        )
    )

    assert any("transcription request failed" in w.lower() for w in warnings)
    assert [s.text for s in segments] == ["recovered"]


def test_empty_text_emits_no_speech_warning(monkeypatch):
    monkeypatch.setenv("TEST_AUDIO_KEY", "sk-test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "   "})

    provider = _build(handler)
    warnings: list[str] = []
    provider.set_warning_callback(warnings.append)

    segments = asyncio.run(_drive(provider, [_silence_chunk(timestamp=0.0)]))

    assert segments == []
    assert any("no speech detected" in w.lower() for w in warnings)


def test_missing_api_key_raises_at_first_call(monkeypatch):
    monkeypatch.delenv("TEST_AUDIO_KEY", raising=False)
    provider = _build(lambda req: httpx.Response(200, json={"text": "x"}))

    # The error surfaces inside stream() — the pipeline catches generic
    # exceptions as warnings, but here we drive the provider directly.
    with pytest.raises(RuntimeError, match="TEST_AUDIO_KEY"):
        asyncio.run(_drive(provider, [_silence_chunk(timestamp=0.0)]))
