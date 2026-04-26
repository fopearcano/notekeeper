"""Top-level coordinator for a recording + transcription session.

Owns the audio recorder, the transcript pipeline, and the in-memory
transcript that the UI displays. The Qt layer drives this from a worker
thread so the UI never blocks.
"""

from __future__ import annotations

import asyncio
import enum
import threading
from typing import Awaitable, Callable, Optional

from app.audio.audio_buffer import AudioBuffer
from app.audio.recorder import AudioRecorder
from app.config.settings import AppSettings
from app.notes.repository import NoteRepository
from app.services.note_processor import NoteProcessor
from app.services.transcript_pipeline import TranscriptPipeline
from app.transcription.base import TranscriptSegment, build_provider as build_transcription
from app.llm.base import build_provider as build_llm
from app.utils.logging import get_logger

log = get_logger(__name__)

SegmentCallback = Callable[[TranscriptSegment], None]


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
        self.recorder = AudioRecorder(settings.audio, self._buffer)

        transcription_provider = build_transcription(settings.transcription)
        self.pipeline = TranscriptPipeline(transcription_provider, self._buffer)
        self.pipeline.add_listener(self._on_segment)

        self.note_processor = NoteProcessor(build_llm(settings.llm))

        self._state = SessionState.IDLE
        self._lock = threading.Lock()
        self._segments: list[TranscriptSegment] = []
        self._segment_listeners: list[SegmentCallback] = []

    # ----- subscription --------------------------------------------------

    def add_segment_listener(self, listener: SegmentCallback) -> None:
        self._segment_listeners.append(listener)

    def remove_segment_listener(self, listener: SegmentCallback) -> None:
        try:
            self._segment_listeners.remove(listener)
        except ValueError:
            pass

    def _on_segment(self, segment: TranscriptSegment) -> None:
        with self._lock:
            self._segments.append(segment)
            listeners = list(self._segment_listeners)
        for cb in listeners:
            try:
                cb(segment)
            except Exception:
                log.exception("Segment UI listener raised")

    # ----- transcript inspection ----------------------------------------

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

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

    async def run_action(self, action: str) -> str:
        transcript = self.transcript_text()
        result = await self.note_processor.process(action=action, transcript=transcript)
        return result.text
