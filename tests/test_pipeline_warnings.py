"""The pipeline must surface stub/missing providers as warnings, not crashes."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from app.audio.audio_buffer import AudioChunk
from app.config.settings import load_settings
from app.services.transcript_pipeline import TranscriptPipeline
from app.transcription.base import TranscriptionProvider, TranscriptSegment
from app.transcription.factory import create_transcription_provider


class _NotImplementedProvider(TranscriptionProvider):
    provider_key = "test_not_implemented"
    is_stub = True

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]  # noqa: ARG002
    ) -> AsyncIterator[TranscriptSegment]:
        raise NotImplementedError("not wired up")
        yield  # type: ignore[unreachable]


def test_pipeline_handles_not_implemented_without_crashing():
    pipeline = TranscriptPipeline(_NotImplementedProvider())

    warnings: list[str] = []
    pipeline.add_warning_listener(warnings.append)

    async def _drive():
        task = pipeline.start()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(_drive())

    # Both: the "is_stub" pre-warning AND the NotImplementedError warning.
    assert any("stub" in w.lower() for w in warnings)
    assert any("not implemented" in w.lower() or "not wired" in w.lower() for w in warnings)


def test_lmstudio_audio_via_factory_warns_through_pipeline():
    """End-to-end through the factory: the LM Studio audio stub must warn, not crash."""
    settings = load_settings(
        bootstrap=False,
        overrides={
            "transcription": {"provider": "lmstudio_audio"},
            "lmstudio_audio": {"enabled": True},
        },
    )
    provider = create_transcription_provider(settings)
    assert provider.is_stub

    pipeline = TranscriptPipeline(provider)
    warnings: list[str] = []
    pipeline.add_warning_listener(warnings.append)

    async def _drive():
        task = pipeline.start()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(_drive())

    assert warnings, "expected a warning from the lmstudio_audio stub"


def test_pipeline_stamps_segments_with_received_offset():
    """Segments forwarded by the pipeline carry a ``received_s`` offset in metadata."""

    class _ScriptedProvider(TranscriptionProvider):
        provider_key = "scripted"
        is_stub = False

        async def start(self) -> None: ...

        async def stop(self) -> None: ...

        async def stream(
            self, chunks: AsyncIterator[AudioChunk]
        ) -> AsyncIterator[TranscriptSegment]:
            async for chunk in chunks:
                yield TranscriptSegment(
                    text=f"chunk@{chunk.timestamp:.2f}",
                    is_final=True,
                    start_s=chunk.timestamp,
                    end_s=chunk.timestamp + chunk.duration_s,
                )

    provider = _ScriptedProvider()
    pipeline = TranscriptPipeline(provider)

    captured: list[TranscriptSegment] = []
    pipeline.add_listener(captured.append)

    async def _run():
        task = pipeline.start()
        # Push two synthetic chunks then stop.
        pipeline.buffer.push(
            AudioChunk(data=b"\x00\x00", sample_rate=16000, channels=1, timestamp=0.5)
        )
        pipeline.buffer.push(
            AudioChunk(data=b"\x00\x00", sample_rate=16000, channels=1, timestamp=1.5)
        )
        await asyncio.sleep(0.4)
        await pipeline.stop()
        # ``stop`` already awaited the task; cancel any remainder defensively.
        if not task.done():
            task.cancel()

    asyncio.run(_run())

    assert len(captured) >= 2
    received = [s.metadata.get("received_s") for s in captured[:2]]
    assert received == pytest.approx([0.5, 1.5])
