"""Pipeline VAD filter and chunk-event channel."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np

from app.audio.audio_buffer import AudioChunk
from app.audio.vad import VoiceActivityDetector
from app.services.transcript_pipeline import TranscriptPipeline
from app.transcription.base import TranscriptionProvider, TranscriptSegment


class _RepeatingProvider(TranscriptionProvider):
    """Yields one segment per chunk so we can match events 1:1 to chunks."""

    provider_key = "repeater"
    is_stub = False

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptSegment]:
        async for chunk in chunks:
            yield TranscriptSegment(
                text=f"chunk@{chunk.timestamp:.1f}",
                is_final=True,
                start_s=chunk.timestamp,
                end_s=chunk.timestamp + chunk.duration_s,
            )


def _silent_chunk(*, ts: float = 0.0) -> AudioChunk:
    return AudioChunk(
        data=b"\x00\x00" * 1600,
        sample_rate=16000,
        channels=1,
        timestamp=ts,
    )


def _loud_chunk(*, ts: float = 0.0, amplitude: int = 16000) -> AudioChunk:
    arr = np.array([amplitude, -amplitude] * 800, dtype=np.int16)
    return AudioChunk(data=arr.tobytes(), sample_rate=16000, channels=1, timestamp=ts)


def _drive(pipeline: TranscriptPipeline, chunks: list[AudioChunk]):
    async def _run():
        task = pipeline.start()
        for c in chunks:
            pipeline.buffer.push(c)
        await asyncio.sleep(0.4)
        await pipeline.stop()
        if not task.done():
            task.cancel()

    asyncio.run(_run())


def test_vad_skipped_chunks_emit_skipped_event_and_no_segment():
    vad = VoiceActivityDetector(enabled=True, threshold=0.05)
    pipeline = TranscriptPipeline(_RepeatingProvider(), vad=vad)

    events: list[tuple[str, float]] = []
    segments: list[str] = []
    pipeline.add_chunk_listener(lambda kind, ts, _: events.append((kind, ts)))
    pipeline.add_listener(lambda seg: segments.append(seg.text))

    _drive(pipeline, [_silent_chunk(ts=0.5), _loud_chunk(ts=1.5)])

    kinds_by_ts = {ts: kind for kind, ts in events}
    assert kinds_by_ts.get(0.5) == "skipped"
    # The loud chunk made it through and produced a segment.
    assert any(kind == "received" and ts == 1.5 for kind, ts in events)
    assert any(kind == "transcribed" and ts == 1.5 for kind, ts in events)
    assert segments == ["chunk@1.5"]


def test_disabled_vad_passes_all_chunks():
    vad = VoiceActivityDetector(enabled=False, threshold=0.5)
    pipeline = TranscriptPipeline(_RepeatingProvider(), vad=vad)

    events: list[tuple[str, float]] = []
    segments: list[str] = []
    pipeline.add_chunk_listener(lambda kind, ts, _: events.append((kind, ts)))
    pipeline.add_listener(lambda seg: segments.append(seg.text))

    _drive(pipeline, [_silent_chunk(ts=0.5)])
    # With VAD off, even a silent chunk reaches the provider.
    assert any(kind == "received" for kind, _ in events)
    assert segments == ["chunk@0.5"]


def test_pipeline_without_vad_emits_received_and_transcribed():
    """``vad=None`` is the default — chunks flow through with full event coverage."""
    pipeline = TranscriptPipeline(_RepeatingProvider())
    events: list[str] = []
    pipeline.add_chunk_listener(lambda kind, _ts, _rms: events.append(kind))

    _drive(pipeline, [_loud_chunk(ts=0.5)])

    assert "received" in events
    assert "transcribed" in events
    assert "skipped" not in events


def test_skipped_chunk_advances_timeline_offset():
    """Even when VAD drops a chunk, ``_last_chunk_offset`` updates so the next
    segment's ``received_s`` reflects real session time."""
    vad = VoiceActivityDetector(enabled=True, threshold=0.05)
    pipeline = TranscriptPipeline(_RepeatingProvider(), vad=vad)

    received: list[float] = []
    pipeline.add_listener(lambda seg: received.append(seg.metadata.get("received_s", -1)))

    _drive(pipeline, [_silent_chunk(ts=0.0), _silent_chunk(ts=1.0), _loud_chunk(ts=2.0)])

    assert received == [2.0]
