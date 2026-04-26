"""Top-level coordinator for a recording + transcription session.

Owns the audio recorder, the transcript pipeline, and the in-memory
transcript that the UI displays. The Qt layer drives this from a worker
thread so the UI never blocks.
"""

from __future__ import annotations

import enum
import threading
from typing import Callable, Optional

from app.audio.audio_buffer import AudioBuffer
from app.audio.recorder import AudioRecorder
from app.config.settings import AppSettings
from app.llm.factory import create_llm_provider
from app.notes.repository import NoteRepository
from app.services.note_processor import NoteProcessor
from app.services.transcript_pipeline import TranscriptPipeline
from app.transcription.base import TranscriptSegment
from app.transcription.factory import create_transcription_provider
from app.utils.logging import get_logger

log = get_logger(__name__)

SegmentCallback = Callable[[TranscriptSegment], None]
LevelCallback = Callable[[float], None]
WarningCallback = Callable[[str], None]
StatusCallback = Callable[[str], None]


class SessionState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    STOPPED = "stopped"


class SessionManager:
    """Coordinates a single recording session.

    Designed to be safe to construct on the UI thread; the heavy lifting
    happens on the worker loop that the UI passes in via :meth:`start`.
    """

    def __init__(self, settings: AppSettings, repository: NoteRepository):
        self.settings = settings
        self.repository = repository

        self._buffer = AudioBuffer()
        self.recorder = AudioRecorder(
            settings.audio,
            sample_rate=settings.transcription.sample_rate,
            chunk_seconds=settings.transcription.chunk_seconds,
            buffer=self._buffer,
        )
        self.recorder.set_level_callback(self._on_level)
        self.recorder.set_error_callback(self._on_recorder_error)

        self.transcription_provider = create_transcription_provider(settings)
        self.pipeline = TranscriptPipeline(self.transcription_provider, self._buffer)
        self.pipeline.add_listener(self._on_segment)
        self.pipeline.add_warning_listener(self._on_warning)
        self.pipeline.add_status_listener(self._on_status)

        self.llm_provider = create_llm_provider(settings)
        self.note_processor = NoteProcessor(self.llm_provider)

        self._state = SessionState.IDLE
        self._lock = threading.Lock()
        self._segments: list[TranscriptSegment] = []

        self._segment_listeners: list[SegmentCallback] = []
        self._level_listeners: list[LevelCallback] = []
        self._warning_listeners: list[WarningCallback] = []
        self._status_listeners: list[StatusCallback] = []

        log.info(
            "SessionManager ready (transcription=%s [stub=%s], llm=%s)",
            self.transcription_provider.provider_key,
            self.transcription_provider.is_stub,
            self.llm_provider.provider_key,
        )

    # ----- subscription --------------------------------------------------

    def add_segment_listener(self, listener: SegmentCallback) -> None:
        self._segment_listeners.append(listener)

    def remove_segment_listener(self, listener: SegmentCallback) -> None:
        try:
            self._segment_listeners.remove(listener)
        except ValueError:
            pass

    def add_level_listener(self, listener: LevelCallback) -> None:
        self._level_listeners.append(listener)

    def remove_level_listener(self, listener: LevelCallback) -> None:
        try:
            self._level_listeners.remove(listener)
        except ValueError:
            pass

    def add_warning_listener(self, listener: WarningCallback) -> None:
        self._warning_listeners.append(listener)

    def remove_warning_listener(self, listener: WarningCallback) -> None:
        try:
            self._warning_listeners.remove(listener)
        except ValueError:
            pass

    def add_status_listener(self, listener: StatusCallback) -> None:
        self._status_listeners.append(listener)

    def remove_status_listener(self, listener: StatusCallback) -> None:
        try:
            self._status_listeners.remove(listener)
        except ValueError:
            pass

    # ----- internal callbacks (run on recorder / pipeline threads) ------

    def _on_segment(self, segment: TranscriptSegment) -> None:
        with self._lock:
            self._segments.append(segment)
            listeners = list(self._segment_listeners)
        for cb in listeners:
            try:
                cb(segment)
            except Exception:
                log.exception("Segment UI listener raised")

    def _on_level(self, level: float) -> None:
        for cb in list(self._level_listeners):
            try:
                cb(level)
            except Exception:
                log.exception("Level listener raised")

    def _on_warning(self, message: str) -> None:
        for cb in list(self._warning_listeners):
            try:
                cb(message)
            except Exception:
                log.exception("Warning listener raised")

    def _on_status(self, message: str) -> None:
        for cb in list(self._status_listeners):
            try:
                cb(message)
            except Exception:
                log.exception("Status listener raised")

    def _on_recorder_error(self, message: str) -> None:
        # Recorder failures are surfaced through the same warning channel.
        self._on_warning(message)

    # ----- transcript inspection ----------------------------------------

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def transcription_is_stub(self) -> bool:
        return self.transcription_provider.is_stub

    def transcript_text(self) -> str:
        with self._lock:
            return " ".join(s.text for s in self._segments).strip()

    def clear_transcript(self) -> None:
        with self._lock:
            self._segments.clear()

    # ----- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        with self._lock:
            if self._state == SessionState.RECORDING:
                return
            self._segments.clear()
            self._state = SessionState.RECORDING
        self.recorder.start()
        self.pipeline.start()
        log.info("Session started")

    async def stop(self) -> None:
        with self._lock:
            if self._state != SessionState.RECORDING:
                return
            self._state = SessionState.STOPPED
        self.recorder.stop()
        await self.pipeline.stop()
        log.info("Session stopped")

    async def aclose(self) -> None:
        await self.note_processor.aclose()

    # ----- LLM convenience ----------------------------------------------

    async def run_action(self, action: Optional[str] = None) -> str:
        """Process the current transcript with ``action`` (default: configured task)."""
        action = action or self.settings.llm.default_task
        transcript = self.transcript_text()
        result = await self.note_processor.process(action=action, transcript=transcript)
        return result.text
