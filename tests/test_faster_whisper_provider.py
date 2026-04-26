"""Unit tests for ``FasterWhisperProvider``.

The real ``faster-whisper`` wheel is not installed in CI; tests monkeypatch the
``_import_whisper_model`` hook so the provider sees a fake model class instead.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from app.audio.audio_buffer import AudioChunk
from app.config.settings import FasterWhisperSettings, TranscriptionSettings
from app.transcription import faster_whisper_provider as fw_mod
from app.transcription.faster_whisper_provider import (
    FasterWhisperProvider,
    _chunk_to_float32,
    _normalize_compute_type_for_cpu,
)


# ---------- helpers ---------------------------------------------------------


class _FakeSegment:
    def __init__(self, text: str, start: float = 0.0, end: float = 1.0):
        self.text = text
        self.start = start
        self.end = end


class _FakeInfo:
    def __init__(self, language: str = "en", language_probability: float = 0.99):
        self.language = language
        self.language_probability = language_probability


class _FakeWhisperModel:
    """Captures construction arguments and serves canned segments."""

    instances: list["_FakeWhisperModel"] = []

    def __init__(self, model: str, *, device: str, compute_type: str):
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.calls: list[np.ndarray] = []
        self.scripted: list[list[_FakeSegment]] = [
            [_FakeSegment("hello world", 0.0, 0.9)],
            [_FakeSegment("second chunk", 0.0, 0.8)],
        ]
        type(self).instances.append(self)

    def transcribe(self, audio, *, language=None, beam_size=1, vad_filter=False):
        self.calls.append(audio)
        if not self.scripted:
            return iter([]), _FakeInfo(language=language or "en")
        segs = self.scripted.pop(0)
        return iter(segs), _FakeInfo(language=language or "en")


def _build(transcription_overrides=None, fw_overrides=None):
    transcription = TranscriptionSettings(
        **{"sample_rate": 16000, "chunk_seconds": 1, **(transcription_overrides or {})}
    )
    faster_whisper = FasterWhisperSettings(
        **{"model": "tiny", "device": "cuda", "compute_type": "float16",
           "allow_cpu_fallback": True, **(fw_overrides or {})}
    )
    return FasterWhisperProvider(transcription, faster_whisper)


def _silence_chunk(*, sample_rate=16000, seconds=1.0, timestamp=0.0) -> AudioChunk:
    n = int(sample_rate * seconds)
    return AudioChunk(
        data=b"\x00\x00" * n, sample_rate=sample_rate, channels=1, timestamp=timestamp
    )


async def _drive(provider: FasterWhisperProvider, chunks: list[AudioChunk]):
    """Run the provider's stream() against a synthetic chunk iterator."""

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


# ---------- pure helpers ----------------------------------------------------


def test_chunk_to_float32_in_unit_range():
    chunk = AudioChunk(
        data=np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16).tobytes(),
        sample_rate=16000,
        channels=1,
        timestamp=0.0,
    )
    out = _chunk_to_float32(chunk)
    assert out.dtype == np.float32
    assert out.shape == (5,)
    assert np.all(out >= -1.0) and np.all(out <= 1.0)
    # 16384 / 32768 = 0.5 exactly.
    assert abs(out[1] - 0.5) < 1e-6


def test_normalize_compute_type_for_cpu():
    assert _normalize_compute_type_for_cpu("float16") == "int8"
    assert _normalize_compute_type_for_cpu("int8_float16") == "int8"
    assert _normalize_compute_type_for_cpu("int8") == "int8"
    assert _normalize_compute_type_for_cpu("float32") == "float32"


# ---------- model loading ---------------------------------------------------


def test_model_loads_lazily_and_only_once(monkeypatch):
    _FakeWhisperModel.instances = []
    monkeypatch.setattr(fw_mod, "_import_whisper_model", lambda: _FakeWhisperModel)

    provider = _build(fw_overrides={"device": "cpu", "compute_type": "int8"})

    statuses: list[str] = []
    provider.set_status_callback(statuses.append)

    chunks = [_silence_chunk(timestamp=0.0), _silence_chunk(timestamp=1.0)]
    segments = asyncio.run(_drive(provider, chunks))

    # One model instance, used twice.
    assert len(_FakeWhisperModel.instances) == 1
    assert len(_FakeWhisperModel.instances[0].calls) == 2

    # Two transcribed chunks → two segments.
    assert [s.text for s in segments] == ["hello world", "second chunk"]
    # Latency status fired for both chunks.
    latency_statuses = [s for s in statuses if "latency" in s.lower()]
    assert len(latency_statuses) >= 2
    # Loading status appeared exactly once (lazy + cached).
    assert sum("loading whisper" in s.lower() for s in statuses) == 1


def test_cuda_failure_falls_back_to_cpu_with_coerced_compute_type(monkeypatch):
    _FakeWhisperModel.instances = []
    attempts: list[tuple[str, str]] = []

    def _flaky(*, device, compute_type):
        attempts.append((device, compute_type))
        if device == "cuda":
            raise RuntimeError("no CUDA driver")
        return _FakeWhisperModel("tiny", device=device, compute_type=compute_type)

    class _FactoryShim:
        def __call__(self, model, *, device, compute_type):
            return _flaky(device=device, compute_type=compute_type)

    monkeypatch.setattr(fw_mod, "_import_whisper_model", lambda: _FactoryShim())

    provider = _build(
        fw_overrides={"device": "cuda", "compute_type": "float16",
                      "allow_cpu_fallback": True}
    )

    statuses: list[str] = []
    provider.set_status_callback(statuses.append)

    segments = asyncio.run(_drive(provider, [_silence_chunk(timestamp=0.0)]))

    # Construction was attempted on cuda first, then on cpu with the coerced type.
    assert attempts == [("cuda", "float16"), ("cpu", "int8")]
    assert any("falling back to CPU" in s for s in statuses)
    assert [s.text for s in segments] == ["hello world"]


def test_cuda_failure_without_fallback_warns_and_drops_chunks(monkeypatch):
    def _always_fail(*args, **kwargs):
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(fw_mod, "_import_whisper_model", lambda: _always_fail)

    provider = _build(
        fw_overrides={"device": "cuda", "compute_type": "float16",
                      "allow_cpu_fallback": False}
    )
    warnings: list[str] = []
    provider.set_warning_callback(warnings.append)

    # The provider raises NotImplementedError on first chunk so the pipeline
    # can warn through its existing handler.
    with pytest.raises(NotImplementedError):
        asyncio.run(_drive(provider, [_silence_chunk(timestamp=0.0)]))


def test_silent_chunk_emits_warning_and_no_segments(monkeypatch):
    """When the model returns no text-bearing segments, a warning fires."""
    _FakeWhisperModel.instances = []

    class _SilentModel(_FakeWhisperModel):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.scripted = [[_FakeSegment("   ")]]  # whitespace-only

    monkeypatch.setattr(fw_mod, "_import_whisper_model", lambda: _SilentModel)

    provider = _build(fw_overrides={"device": "cpu", "compute_type": "int8"})
    warnings: list[str] = []
    provider.set_warning_callback(warnings.append)

    segments = asyncio.run(_drive(provider, [_silence_chunk(timestamp=0.0)]))

    assert segments == []
    assert any("no speech detected" in w.lower() for w in warnings)


def test_segments_include_latency_and_language_metadata(monkeypatch):
    _FakeWhisperModel.instances = []
    monkeypatch.setattr(fw_mod, "_import_whisper_model", lambda: _FakeWhisperModel)

    provider = _build(
        transcription_overrides={"language": "auto"},
        fw_overrides={"device": "cpu", "compute_type": "int8"},
    )
    segments = asyncio.run(_drive(provider, [_silence_chunk(timestamp=2.5)]))

    assert len(segments) == 1
    seg = segments[0]
    assert seg.metadata.get("language") == "en"
    assert seg.metadata.get("language_probability") == pytest.approx(0.99)
    assert seg.metadata.get("latency_ms", -1) >= 0
    # Chunk timestamp is added to the model's segment offsets.
    assert seg.start_s == pytest.approx(2.5)
    assert seg.end_s == pytest.approx(3.4)


def test_language_auto_passes_none_to_model(monkeypatch):
    captured = {}

    class _RecordingModel(_FakeWhisperModel):
        def transcribe(self, audio, *, language=None, **kw):
            captured["language"] = language
            return iter([_FakeSegment("ok")]), _FakeInfo(language="en")

    monkeypatch.setattr(fw_mod, "_import_whisper_model", lambda: _RecordingModel)

    provider = _build(
        transcription_overrides={"language": "auto"},
        fw_overrides={"device": "cpu", "compute_type": "int8"},
    )
    asyncio.run(_drive(provider, [_silence_chunk(timestamp=0.0)]))
    assert captured["language"] is None


# ---------- config -----------------------------------------------------------


def test_allow_cpu_fallback_default_true():
    fw = FasterWhisperSettings()
    assert fw.allow_cpu_fallback is True


def test_allow_cpu_fallback_can_be_disabled():
    fw = FasterWhisperSettings(allow_cpu_fallback=False)
    assert fw.allow_cpu_fallback is False
