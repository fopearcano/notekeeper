"""Pipeline status/warning forwarding tests."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from app.audio.audio_buffer import AudioChunk
from app.services.transcript_pipeline import TranscriptPipeline
from app.transcription.base import TranscriptionProvider, TranscriptSegment


class _ChattyProvider(TranscriptionProvider):
    """Provider that exercises both side channels during stream()."""

    provider_key = "chatty"
    is_stub = False

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptSegment]:
        self._emit_status("Loading…")
        async for chunk in chunks:
            self._emit_status(f"latency: {chunk.timestamp:.1f} ms")
            if chunk.timestamp < 0:
                self._emit_warning("silly chunk")
                continue
            yield TranscriptSegment(text=f"hi@{chunk.timestamp:.1f}", is_final=True)


def test_pipeline_forwards_provider_status_and_warnings():
    pipeline = TranscriptPipeline(_ChattyProvider())

    statuses: list[str] = []
    warnings: list[str] = []
    segments: list[TranscriptSegment] = []
    pipeline.add_status_listener(statuses.append)
    pipeline.add_warning_listener(warnings.append)
    pipeline.add_listener(segments.append)

    async def _run():
        task = pipeline.start()
        # Push a normal chunk and a "silly" one to exercise both channels.
        pipeline.buffer.push(
            AudioChunk(data=b"\x00" * 4, sample_rate=16000, channels=1, timestamp=0.5)
        )
        pipeline.buffer.push(
            AudioChunk(data=b"\x00" * 4, sample_rate=16000, channels=1, timestamp=-1.0)
        )
        await asyncio.sleep(0.3)
        await pipeline.stop()
        if not task.done():
            task.cancel()

    asyncio.run(_run())

    assert any("loading" in s.lower() for s in statuses)
    assert any("latency" in s.lower() for s in statuses)
    assert any("silly" in w.lower() for w in warnings)
    assert any(seg.text.startswith("hi@") for seg in segments)
