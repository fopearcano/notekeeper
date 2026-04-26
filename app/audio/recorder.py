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

Pause / resume close and re-open the underlying ``InputStream`` so the OS
mic indicator reflects reality. Paused time is excluded from the recorder's
elapsed-second clock so the UI's "REC 00:42" display tracks audio actually
captured rather than wall-clock time since the user hit Start.
"""

from __future__ import annotations

import enum
import threading
import time
from typing import Any, Callable, Optional

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
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class AudioRecorder:
    """Manages a microphone capture session driven by ``sounddevice``."""

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
        self.channels = 1  # product spec is mono — we down-mix multi-ch input.

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
        self._paused_at: Optional[float] = None
        self._total_paused_s: float = 0.0
        self._last_level_emit: float = 0.0
        self._current_level: float = 0.0

        self._level_callback: Optional[LevelCallback] = None
        self._error_callback: Optional[ErrorCallback] = None

    # ----- callbacks -----------------------------------------------------

    def set_level_callback(self, callback: Optional[LevelCallback]) -> None:
        self._level_callback = callback

    def set_error_callback(self, callback: Optional[ErrorCallback]) -> None:
        self._error_callback = callback

    # ----- state ---------------------------------------------------------

    @property
    def state(self) -> RecorderState:
        with self._lock:
            return self._state

    @property
    def is_paused(self) -> bool:
        return self.state == RecorderState.PAUSED

    @property
    def is_recording(self) -> bool:
        return self.state in (RecorderState.RECORDING, RecorderState.PAUSED)

    @property
    def current_level(self) -> float:
        return self._current_level

    def elapsed_s(self) -> float:
        """Active-recording elapsed seconds (paused intervals excluded)."""
        with self._lock:
            if self._started_at is None:
                return 0.0
            end = (
                self._paused_at
                if self._state == RecorderState.PAUSED
                else time.monotonic()
            )
            return max(0.0, end - self._started_at - self._total_paused_s)

    # ----- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Open the input stream and begin pushing chunks into the buffer."""
        with self._lock:
            if self._state in (RecorderState.RECORDING, RecorderState.PAUSED):
                log.debug("Recorder already running (state=%s)", self._state)
                return
            self.buffer.clear()
            self._assembler = ChunkAssembler(
                sample_rate=self.sample_rate,
                chunk_seconds=self.chunk_seconds,
                channels=self.channels,
            )
            self._started_at = time.monotonic()
            self._paused_at = None
            self._total_paused_s = 0.0
            # Set well in the past so the first audio block always fires the
            # level callback — otherwise short captures would never report a level.
            self._last_level_emit = 0.0
            self._current_level = 0.0
            self._state = RecorderState.RECORDING

        if not self._open_stream():
            return

        log.info(
            "AudioRecorder started (sample_rate=%d, channels=%d, device=%s, chunk=%.2fs)",
            self.sample_rate,
            self.channels,
            self.settings.input_device,
            self.chunk_seconds,
        )

    def pause(self) -> None:
        """Stop capturing audio without ending the session.

        The PortAudio stream is closed so the OS mic indicator goes off.
        Elapsed-time accounting freezes; :meth:`resume` adds the paused
        interval to ``_total_paused_s`` and re-opens the stream.
        """
        with self._lock:
            if self._state != RecorderState.RECORDING:
                return
            self._paused_at = time.monotonic()
            self._state = RecorderState.PAUSED

        self._close_stream()
        if self._level_callback is not None:
            try:
                self._level_callback(0.0)
            except Exception:
                log.exception("Level callback raised on pause")
        log.info("AudioRecorder paused")

    def resume(self) -> None:
        """Re-open the input stream after a :meth:`pause`."""
        with self._lock:
            if self._state != RecorderState.PAUSED:
                return
            now = time.monotonic()
            if self._paused_at is not None:
                self._total_paused_s += now - self._paused_at
            self._paused_at = None
            self._state = RecorderState.RECORDING

        if not self._open_stream():
            return
        log.info("AudioRecorder resumed")

    def stop(self) -> None:
        with self._lock:
            if self._state not in (RecorderState.RECORDING, RecorderState.PAUSED):
                return
            # If we were paused, fold the pending pause interval into the total.
            if self._state == RecorderState.PAUSED and self._paused_at is not None:
                self._total_paused_s += time.monotonic() - self._paused_at
                self._paused_at = None
            self._state = RecorderState.STOPPED

        self._close_stream()

        # Flush any partial buffer so a short final chunk still reaches the pipeline.
        tail = self._assembler.flush(timestamp=self._monotonic_offset())
        if tail is not None:
            self.buffer.push(tail)

        self._current_level = 0.0
        if self._level_callback is not None:
            try:
                self._level_callback(0.0)
            except Exception:
                log.exception("Level callback raised on stop")

        log.info("AudioRecorder stopped")

    # ----- internals -----------------------------------------------------

    def _open_stream(self) -> bool:
        """Open ``sd.InputStream`` and start it. Returns True on success."""
        try:
            import sounddevice as sd
        except (OSError, ImportError) as exc:
            self._fail(f"Audio backend unavailable: {exc}")
            return False

        device_arg = self._device_arg(self.settings.input_device)
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=device_arg,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:  # broad: PortAudio wraps many root causes.
            self._fail(f"Could not open microphone: {exc}")
            return False
        return True

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.exception("Error closing audio stream")

    @staticmethod
    def _device_arg(value: Any) -> Any:
        """Normalize the configured device value for ``sounddevice``."""
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "" or stripped.lower() == "default":
                return None
            return stripped
        return value  # int index passes through.

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
        return time.monotonic() - self._started_at - self._total_paused_s

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        """Runs on the PortAudio worker thread — keep it lean."""
        if status:
            log.warning("sounddevice status flag: %s", status)

        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        elif indata.ndim == 2:
            mono = indata[:, 0]
        else:
            mono = indata

        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
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

        pcm = np.clip(mono * 32767.0, -32768.0, 32767.0).astype(np.int16, copy=False)
        offset = self._monotonic_offset()
        for chunk in self._assembler.feed(pcm.tobytes(), timestamp=offset):
            self.buffer.push(chunk)


# --------------------------------------------------------------------------- #
# Device discovery                                                            #
# --------------------------------------------------------------------------- #


def list_input_devices() -> list[dict[str, Any]]:
    """Return a small list of ``{index, name, default}`` dicts for the UI.

    Returns an empty list if PortAudio is unavailable or query fails. The UI
    always shows "System default" as the first entry on top of this.
    """
    try:
        import sounddevice as sd
    except Exception as exc:
        log.debug("sounddevice unavailable for device listing: %s", exc)
        return []

    try:
        devices = sd.query_devices()
        try:
            default_in, _ = sd.default.device  # may be (None, None)
        except Exception:
            default_in = None
    except Exception as exc:
        log.warning("query_devices failed: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for idx, dev in enumerate(devices or []):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        out.append(
            {
                "index": idx,
                "name": dev.get("name", f"device {idx}"),
                "default": (idx == default_in),
            }
        )
    return out
