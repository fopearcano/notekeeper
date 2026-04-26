"""Transcription provider interface.

Providers consume a stream of :class:`AudioChunk` objects and yield
:class:`TranscriptSegment` events as they become available. Concrete
providers wrap faster-whisper, OpenAI's audio API, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from app.audio.audio_buffer import AudioChunk
from app.config.settings import TranscriptionSettings


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

    def __init__(self, settings: TranscriptionSettings):
        self.settings = settings

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


def build_provider(settings: TranscriptionSettings) -> TranscriptionProvider:
    """Factory dispatch on ``settings.provider``."""
    # Local imports avoid pulling heavy ML deps unless the user actually selects them.
    if settings.provider == "faster_whisper":
        from app.transcription.faster_whisper_provider import FasterWhisperProvider

        return FasterWhisperProvider(settings)
    if settings.provider == "openai":
        from app.transcription.openai_audio_provider import OpenAIAudioProvider

        return OpenAIAudioProvider(settings)
    if settings.provider == "lmstudio_stub":
        from app.transcription.lmstudio_audio_provider_stub import LMStudioAudioProviderStub

        return LMStudioAudioProviderStub(settings)
    raise ValueError(f"Unknown transcription provider: {settings.provider!r}")
