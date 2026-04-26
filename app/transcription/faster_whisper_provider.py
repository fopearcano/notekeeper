"""faster-whisper backed transcription provider (placeholder).

Real implementation will lazy-import ``faster_whisper.WhisperModel`` and feed
PCM frames into it. For the scaffold we yield a deterministic stub stream so
the rest of the pipeline has something to forward to the UI.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from app.audio.audio_buffer import AudioChunk
from app.transcription.base import TranscriptionProvider, TranscriptSegment
from app.utils.logging import get_logger

log = get_logger(__name__)

_DEMO_PHRASES = [
    "Welcome to Notekeeper.",
    "This transcript is a placeholder.",
    "Real audio capture will plug into the same pipeline.",
    "You can summarize, organize, or format any selection.",
]


class FasterWhisperProvider(TranscriptionProvider):
    async def start(self) -> None:
        log.info(
            "FasterWhisperProvider.start (model=%s, language=%s) — using stub output",
            self.settings.model,
            self.settings.language,
        )
        # TODO: load WhisperModel(model, compute_type=...) here.

    async def stop(self) -> None:
        log.info("FasterWhisperProvider.stop")

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptSegment]:
        # Drain chunks in the background so the real signature matches the
        # eventual implementation. Until the model is wired up we ignore them.
        async def _drain() -> None:
            async for _ in chunks:
                pass

        drain_task = asyncio.create_task(_drain())
        try:
            elapsed = 0.0
            for phrase in _DEMO_PHRASES:
                await asyncio.sleep(self.settings.emit_interval_ms / 1000.0)
                start, end = elapsed, elapsed + 1.0
                yield TranscriptSegment(
                    text=phrase, is_final=True, start_s=start, end_s=end
                )
                elapsed = end
        finally:
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
