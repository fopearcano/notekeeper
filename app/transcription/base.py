"""Transcription provider interface.

Providers consume a stream of :class:`AudioChunk` objects and yield
:class:`TranscriptSegment` events as they become available. Concrete
providers wrap faster-whisper, OpenAI's audio API, etc.

Providers are constructed via :func:`app.transcription.factory.create_transcription_provider`.

Two side channels are available for non-segment messages so the UI can show
loading state, latency, or "no speech" warnings without polluting the
transcript view:

* :meth:`TranscriptionProvider.set_status_callback` — transient updates such
  as "Loading Whisper model…" or "Latency: 245 ms".
* :meth:`TranscriptionProvider.set_warning_callback` — recoverable problems
  such as "no speech detected in chunk".
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

from app.audio.audio_buffer import AudioChunk

log = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]
WarningCallback = Callable[[str], None]


@dataclass(frozen=True)
class TranscriptSegment:
    """One unit of transcribed text."""

    text: str
    is_final: bool = False
    start_s: float = 0.0
    end_s: float = 0.0
    speaker: str | None = None
    metadata: dict = field(default_factory=dict)


class TranscriptionProvider(ABC):
    """Async streaming transcription contract."""

    #: Short identifier used for status display (e.g. ``"faster_whisper"``).
    provider_key: str = "unknown"

    #: Set ``True`` when the provider does not actually transcribe audio
    #: (placeholder/scripted output, or a not-yet-implemented backend). The
    #: UI surfaces this as a warning so users know transcription isn't real.
    is_stub: bool = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # ----- side channels -------------------------------------------------

    def set_status_callback(self, callback: Optional[StatusCallback]) -> None:
        self._status_callback = callback

    def set_warning_callback(self, callback: Optional[WarningCallback]) -> None:
        self._warning_callback = callback

    def _emit_status(self, message: str) -> None:
        cb: Optional[StatusCallback] = getattr(self, "_status_callback", None)
        if cb is None:
            return
        try:
            cb(message)
        except Exception:
            log.exception("Status callback raised")

    def _emit_warning(self, message: str) -> None:
        cb: Optional[WarningCallback] = getattr(self, "_warning_callback", None)
        if cb is None:
            return
        try:
            cb(message)
        except Exception:
            log.exception("Warning callback raised")

    # ----- lifecycle -----------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Open any underlying model / connection."""

    @abstractmethod
    async def stop(self) -> None:
        """Tear everything down."""

    @abstractmethod
    async def stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptSegment]:
        """Consume audio chunks and yield transcript segments."""
