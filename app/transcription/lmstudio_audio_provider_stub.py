"""LM Studio audio provider — stub.

LM Studio does not currently expose a streaming audio transcription endpoint
in any documented form. This stub stays in place until a compatible endpoint
is explicitly available so the configuration enum stays exhaustive and the
UI can show a clear "not available" status when the user picks it.

The ``[lmstudio_audio].enabled`` flag in config defaults to ``false`` for
exactly this reason — flipping it on without a real backend will fail at
``stream`` time, not at start.
"""

from __future__ import annotations

from typing import AsyncIterator

from app.audio.audio_buffer import AudioChunk
from app.config.settings import LMStudioAudioSettings, TranscriptionSettings
from app.transcription.base import TranscriptionProvider, TranscriptSegment
from app.utils.logging import get_logger

log = get_logger(__name__)


class LMStudioAudioProviderStub(TranscriptionProvider):
    provider_key = "lmstudio_audio"

    def __init__(
        self,
        transcription: TranscriptionSettings,
        lmstudio_audio: LMStudioAudioSettings,
    ):
        self.transcription = transcription
        self.lmstudio_audio = lmstudio_audio

    async def start(self) -> None:
        if not self.lmstudio_audio.enabled:
            log.warning(
                "LMStudioAudioProviderStub.start — disabled in config "
                "(lmstudio_audio.enabled = false). No audio endpoint will be called."
            )
            return
        log.warning(
            "LMStudioAudioProviderStub.start — enabled but no compatible LM Studio "
            "audio endpoint is implemented. Calls will raise NotImplementedError."
        )

    async def stop(self) -> None:
        log.info("LMStudioAudioProviderStub.stop")

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]  # noqa: ARG002 - placeholder
    ) -> AsyncIterator[TranscriptSegment]:
        raise NotImplementedError(
            "LM Studio does not expose a real-time audio transcription endpoint. "
            "Use 'faster_whisper' or 'openai_audio' instead."
        )
        yield  # type: ignore[unreachable]
