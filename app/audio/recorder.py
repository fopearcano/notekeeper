"""Microphone capture using ``sounddevice``.

The recorder owns a ``sounddevice.InputStream`` whose callback runs on a
PortAudio worker thread. Captured frames are converted to int16 PCM and fed
through a :class:`ChunkAssembler`; finished chunks are pushed onto the
shared :class:`AudioBuffer` for the transcript pipeline to pick up. An RMS
level is computed per block and forwarded to a registered callback so the
GUI can render a level meter.

The audio callback is held to bytearray + numpy work only — no I/O, no Qt
calls — so the OS audio thread never stalls. UI updates are delivered via
plain Python callables that the GUI side wires onto Qt signals (which is
the safe way to cross threads in Qt).
"""

from __future__ import annotations

import enum
import threading
import time
from typing import Callable, Optional

import numpy as np

from app.audio.audio_buffer import AudioBuffer, ChunkAssembler
from app.config.settings import AudioSettings
from app.utils.logging import get_logger

log = get_logger(__name__)

LevelCallback = Callable[[float], None]
ErrorCallback = Callable[[str], None]


class RecorderState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    STOPPED = "stopped"
    ERROR = "error"


class AudioRecorder:
    """Manages a microphone capture session driven by ``sounddevice``.

    Construction is cheap and side-effect free — the PortAudio stream is
    only opened in :meth:`start`, so this object can be created on the GUI
    thread and tested without an audio device present.
    """

    #: Throttle level-callback invocations to roughly this many per second.
    _LEVEL_HZ = 30

    def __init__(
        self,
        settings: AudioSettings,
        sample_rate: int,
        chunk_seconds: float,
        buffer: Optional[AudioBuffer] = None,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")

        self.settings = settings
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        # The product spec is mono — we down-mix multi-channel input ourselves.
        self.channels = 1

        # ``buffer or AudioBuffer()`` is wrong here — an empty buffer is falsy
        # (``__len__`` returns 0) so the caller's buffer would silently be replaced.
        self.buffer = buffer if buffer is not None else AudioBuffer()
        self._assembler = ChunkAssembler(
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            channels=self.channels,
        )

        self._state = RecorderState.IDLE
        self._lock = threading.Lock()
        self._stream = None  # type: ignore[assignment]
        self._started_at: Optional[float] = None
        self._last_level_emit: float = 0.0
        self._current_level: float = 0.0

        self._level_callback: Optional[LevelCallback] = None
        self._error_callback: Optional[ErrorCallback] = None

    # ----- callbacks -----------------------------------------------------

    def set_level_callback(self, callback: Optional[LevelCallback]) -> None:
        """Register a function that receives RMS level in [0.0, 1.0]."""
        self._level_callback = callback

    def set_error_callback(self, callback: Optional[ErrorCallback]) -> None:
        """Register a function that receives a human-readable error message."""
        self._error_callback = callback

    # ----- state ---------------------------------------------------------

    @property
    def state(self) -> RecorderState:
        with self._lock:
            return self._state

    @property
    def current_level(self) -> float:
        return self._current_level

    def elapsed_s(self) -> float:
        with self._lock:
            if self._started_at is None:
                return 0.0
            return time.monotonic() - self._started_at

    # ----- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Open the input stream and begin pushing chunks into the buffer.

        Errors opening the device (no PortAudio backend, no input device,
        unsupported sample rate, etc.) are caught — the recorder transitions
        to ``ERROR`` and the registered error callback receives a message.
        Callers should check :pyattr:`state` after calling.
        """
        with self._lock:
            if self._state == RecorderState.RECORDING:
                log.debug("Recorder already running")
                return
            self.buffer.clear()
            self._assembler = ChunkAssembler(
                sample_rate=self.sample_rate,
                chunk_seconds=self.chunk_seconds,
                channels=self.channels,
            )
            self._started_at = time.monotonic()
            # Set well in the past so the first audio block always fires the
            # level callback — otherwise short captures would never report a level.
            self._last_level_emit = 0.0
            self._current_level = 0.0
            self._state = RecorderState.RECORDING

        try:
            import sounddevice as sd
        except (OSError, ImportError) as exc:
            self._fail(f"Audio backend unavailable: {exc}")
            return

        device = self.settings.device
        device_arg = None if device in ("default", "", None) else device

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=device_arg,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:  # broad: PortAudio errors wrap many root causes
            self._fail(f"Could not open microphone: {exc}")
            return

        log.info(
            "AudioRecorder started (sample_rate=%d, channels=%d, device=%s, chunk=%.2fs)",
            self.sample_rate,
            self.channels,
            device,
            self.chunk_seconds,
        )

    def stop(self) -> None:
        with self._lock:
            if self._state != RecorderState.RECORDING:
                return
            self._state = RecorderState.STOPPED

        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.exception("Error closing audio stream")

        # Flush any partial buffer so a short final chunk still reaches the pipeline.
        tail = self._assembler.flush(timestamp=self._monotonic_offset())
        if tail is not None:
            self.buffer.push(tail)

        # Reset the level meter so the UI doesn't keep showing the last frame's reading.
        self._current_level = 0.0
        if self._level_callback is not None:
            try:
                self._level_callback(0.0)
            except Exception:
                log.exception("Level callback raised on stop")

        log.info("AudioRecorder stopped")

    # ----- internals -----------------------------------------------------

    def _fail(self, message: str) -> None:
        with self._lock:
            self._state = RecorderState.ERROR
        log.error(message)
        if self._error_callback is not None:
            try:
                self._error_callback(message)
            except Exception:
                log.exception("Error callback raised")

    def _monotonic_offset(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        """Runs on the PortAudio worker thread — keep it lean."""
        if status:
            # ``status`` is a CallbackFlags bitfield (input overflow, etc.).
            log.warning("sounddevice status flag: %s", status)

        # Down-mix to mono if the device delivered multiple channels (rare with
        # channels=1, but defensive against driver quirks).
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        elif indata.ndim == 2:
            mono = indata[:, 0]
        else:
            mono = indata

        # RMS in [0, 1] (mono is float32 in [-1, 1]).
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        # A boost of ~4× makes everyday speech read as a useful range on the meter
        # without saturating; clipped to [0, 1].
        level = min(1.0, rms * 4.0)
        self._current_level = level

        now = time.monotonic()
        if (
            self._level_callback is not None
            and now - self._last_level_emit >= 1.0 / self._LEVEL_HZ
        ):
            self._last_level_emit = now
            try:
                self._level_callback(level)
            except Exception:
                log.exception("Level callback raised")

        # Convert to int16 PCM bytes.
        pcm = np.clip(mono * 32767.0, -32768.0, 32767.0).astype(np.int16, copy=False)
        offset = now - (self._started_at or now)
        for chunk in self._assembler.feed(pcm.tobytes(), timestamp=offset):
            self.buffer.push(chunk)
