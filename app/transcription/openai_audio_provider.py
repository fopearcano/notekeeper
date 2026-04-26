"""OpenAI audio transcription provider (placeholder).

Real implementation will buffer PCM into short WAV blobs and POST them to the
``/v1/audio/transcriptions`` endpoint. For now it raises if invoked while
real-time mode is requested — this gives the UI a clear error path before
the API integration lands.
"""

from __future__ import annotations

from typing import AsyncIterator

from app.audio.audio_buffer import AudioChunk
from app.transcription.base import TranscriptionProvider, TranscriptSegment
from app.utils.logging import get_logger

log = get_logger(__name__)


class OpenAIAudioProvider(TranscriptionProvider):
    async def start(self) -> None:
        log.info("OpenAIAudioProvider.start (model=%s)", self.settings.model)
        # TODO: build httpx.AsyncClient with the configured api_key.

    async def stop(self) -> None:
        log.info("OpenAIAudioProvider.stop")

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]  # noqa: ARG002 - placeholder
    ) -> AsyncIterator[TranscriptSegment]:
        raise NotImplementedError(
            "OpenAI audio transcription is not implemented yet. "
            "Switch transcription.provider to 'faster_whisper' or 'lmstudio_stub'."
        )
        # Make this an async generator at the type level.
        yield  # type: ignore[unreachable]
