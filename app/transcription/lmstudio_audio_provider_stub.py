"""LM Studio audio provider — stub.

LM Studio does not currently expose a streaming audio transcription endpoint.
This stub exists so the configuration enum stays exhaustive and so the UI
can display a clear "not available" status when the user picks it.
"""

from __future__ import annotations

from typing import AsyncIterator

from app.audio.audio_buffer import AudioChunk
from app.transcription.base import TranscriptionProvider, TranscriptSegment
from app.utils.logging import get_logger

log = get_logger(__name__)


class LMStudioAudioProviderStub(TranscriptionProvider):
    async def start(self) -> None:
        log.warning(
            "LMStudioAudioProviderStub.start — LM Studio has no real-time audio API yet."
        )

    async def stop(self) -> None:
        log.info("LMStudioAudioProviderStub.stop")

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]  # noqa: ARG002 - placeholder
    ) -> AsyncIterator[TranscriptSegment]:
        raise NotImplementedError(
            "LM Studio does not expose real-time audio transcription. "
            "Use 'faster_whisper' or 'openai' providers instead."
        )
        yield  # type: ignore[unreachable]
