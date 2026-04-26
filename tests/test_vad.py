"""Energy-based VAD."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.audio_buffer import AudioChunk
from app.audio.vad import VoiceActivityDetector, build_default_vad, chunk_rms
from app.config.settings import AudioSettings


def _chunk(values: list[int]) -> AudioChunk:
    arr = np.array(values, dtype=np.int16)
    return AudioChunk(
        data=arr.tobytes(), sample_rate=16000, channels=1, timestamp=0.0
    )


def _silent() -> AudioChunk:
    return _chunk([0] * 1600)  # 0.1 s of pure zeros


def _loud(amplitude: int = 16000) -> AudioChunk:
    """A simple alternating waveform — RMS ≈ amplitude / 32768."""
    arr = ([amplitude, -amplitude] * 800)
    return _chunk(arr)


def test_chunk_rms_zero_for_silence():
    assert chunk_rms(_silent()) == 0.0


def test_chunk_rms_handles_empty_chunk():
    chunk = AudioChunk(data=b"", sample_rate=16000, channels=1, timestamp=0.0)
    assert chunk_rms(chunk) == 0.0


def test_chunk_rms_for_full_scale_square_wave_is_close_to_1():
    chunk = _loud(amplitude=32767)
    assert 0.99 <= chunk_rms(chunk) <= 1.0


def test_disabled_vad_passes_silence():
    vad = VoiceActivityDetector(enabled=False, threshold=0.5)
    assert vad.is_speech(_silent()) is True


def test_enabled_vad_blocks_silence():
    vad = VoiceActivityDetector(enabled=True, threshold=0.01)
    assert vad.is_speech(_silent()) is False


def test_enabled_vad_passes_speech_above_threshold():
    vad = VoiceActivityDetector(enabled=True, threshold=0.05)
    assert vad.is_speech(_loud(amplitude=16000)) is True
    assert vad.is_speech(_loud(amplitude=500)) is False


def test_set_threshold_runtime_mutable():
    vad = VoiceActivityDetector(enabled=True, threshold=0.5)
    chunk = _loud(amplitude=16000)  # rms ~ 0.49
    assert vad.is_speech(chunk) is False
    vad.set_threshold(0.1)
    assert vad.is_speech(chunk) is True


def test_set_enabled_runtime_mutable():
    vad = VoiceActivityDetector(enabled=True, threshold=0.99)
    assert vad.is_speech(_loud()) is False
    vad.set_enabled(False)
    assert vad.is_speech(_loud()) is True


def test_classify_returns_rms_only_once():
    vad = VoiceActivityDetector(enabled=True, threshold=0.05)
    is_speech, rms = vad.classify(_loud(amplitude=16000))
    assert is_speech is True
    assert 0.4 <= rms <= 0.6


def test_negative_threshold_rejected():
    with pytest.raises(ValueError):
        VoiceActivityDetector(threshold=-0.1)
    vad = VoiceActivityDetector()
    with pytest.raises(ValueError):
        vad.set_threshold(-1.0)


def test_build_default_vad_uses_settings():
    settings = AudioSettings(vad_enabled=False, vad_threshold=0.123)
    vad = build_default_vad(settings)
    assert vad.enabled is False
    assert vad.threshold == pytest.approx(0.123)
