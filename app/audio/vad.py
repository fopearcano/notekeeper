"""Energy-based voice activity detection.

Computes the root-mean-square amplitude of an :class:`AudioChunk`'s int16
PCM data on a [0, 1] scale and compares it against a configurable
threshold. The detector is intentionally tiny — for production-grade
turn-taking we'd swap in WebRTC VAD or Silero, but for "skip dead air
before sending it to the transcriber" a single energy check is enough.

Two knobs, both runtime-mutable so the GUI can flip the toggle while the
session is live without re-creating the pipeline:

* ``enabled`` — when ``False`` :meth:`is_speech` always returns ``True``.
* ``threshold`` — RMS in [0, 1]; chunks below this are reported as silent.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from app.audio.audio_buffer import AudioChunk


def chunk_rms(chunk: AudioChunk) -> float:
    """Return RMS amplitude of ``chunk`` in [0, 1].

    Pure helper exposed for tests and for the recorder's level meter to
    share the same arithmetic the VAD uses.
    """
    if not chunk.data:
        return 0.0
    arr = np.frombuffer(chunk.data, dtype=np.int16)
    if arr.size == 0:
        return 0.0
    floats = arr.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(np.square(floats, dtype=np.float64))))


class VoiceActivityDetector:
    """Thread-safe energy-based VAD.

    Both knobs are mutable so the UI can flip them on the fly without
    tearing the pipeline down.
    """

    def __init__(self, *, enabled: bool = True, threshold: float = 0.01):
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        self._lock = threading.Lock()
        self._enabled = bool(enabled)
        self._threshold = float(threshold)

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def threshold(self) -> float:
        with self._lock:
            return self._threshold

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def set_threshold(self, threshold: float) -> None:
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        with self._lock:
            self._threshold = float(threshold)

    def rms(self, chunk: AudioChunk) -> float:
        return chunk_rms(chunk)

    def is_speech(self, chunk: AudioChunk) -> bool:
        """Return ``True`` when ``chunk`` should be sent to the transcriber.

        With VAD off every chunk passes through. With VAD on the chunk's
        RMS must meet or exceed :pyattr:`threshold`. Empty / zero chunks
        are always treated as silence so the transcriber never sees a
        zero-length blob.
        """
        with self._lock:
            enabled = self._enabled
            threshold = self._threshold
        if not enabled:
            return True
        rms = chunk_rms(chunk)
        return rms >= threshold

    def classify(self, chunk: AudioChunk) -> tuple[bool, float]:
        """Return ``(is_speech, rms)`` so callers don't recompute RMS twice."""
        with self._lock:
            enabled = self._enabled
            threshold = self._threshold
        rms = chunk_rms(chunk)
        if not enabled:
            return True, rms
        return rms >= threshold, rms


def build_default_vad(audio_settings) -> VoiceActivityDetector:
    """Construct a :class:`VoiceActivityDetector` from :class:`AudioSettings`."""
    return VoiceActivityDetector(
        enabled=audio_settings.vad_enabled,
        threshold=audio_settings.vad_threshold,
    )
