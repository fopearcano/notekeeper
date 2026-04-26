"""Transcription provider interface.

Providers consume a stream of :class:`AudioChunk` objects and yield
:class:`TranscriptSegment` events as they become available. Concrete
providers wrap faster-whisper, OpenAI's audio API, etc.

Providers are constructed via :func:`app.transcription.factory.create_transcription_provider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from app.audio.audio_buffer import AudioChunk


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

    @property
    def name(self) -> str:
        return self.__class__.__name__

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
