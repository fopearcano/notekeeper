"""Microphone capture (placeholder).

Real capture will use ``sounddevice.RawInputStream`` to push int16 frames
into an :class:`AudioBuffer`. The current implementation only manages state
so the rest of the app can drive the same start/stop API today.
"""

from __future__ import annotations

import enum
import threading
import time
from typing import Optional

from app.audio.audio_buffer import AudioBuffer
from app.config.settings import AudioSettings
from app.utils.logging import get_logger

log = get_logger(__name__)


class RecorderState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    STOPPED = "stopped"


class AudioRecorder:
    """Manages a microphone capture session.

    The real version will spin up a ``sounddevice`` stream on start and tear
    it down on stop. Today it only flips state and the buffer is filled by
    whatever stub the transcription pipeline is using.
    """

    def __init__(self, settings: AudioSettings, buffer: Optional[AudioBuffer] = None):
        self.settings = settings
        self.buffer = buffer or AudioBuffer()
        self._state = RecorderState.IDLE
        self._lock = threading.Lock()
        self._started_at: Optional[float] = None

    @property
    def state(self) -> RecorderState:
        with self._lock:
            return self._state

    def start(self) -> None:
        with self._lock:
            if self._state == RecorderState.RECORDING:
                log.debug("Recorder already running")
                return
            self.buffer.clear()
            self._state = RecorderState.RECORDING
            self._started_at = time.monotonic()
        # TODO: open sounddevice.RawInputStream and push chunks into self.buffer
        log.info(
            "AudioRecorder started (sample_rate=%d, channels=%d, device=%s)",
            self.settings.sample_rate,
            self.settings.channels,
            self.settings.device,
        )

    def stop(self) -> None:
        with self._lock:
            if self._state != RecorderState.RECORDING:
                return
            self._state = RecorderState.STOPPED
        # TODO: stop and close the sounddevice stream
        log.info("AudioRecorder stopped")

    def elapsed_s(self) -> float:
        with self._lock:
            if self._started_at is None:
                return 0.0
            return time.monotonic() - self._started_at
