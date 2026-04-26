"""Tests for ``AudioBuffer`` and ``ChunkAssembler``."""

from __future__ import annotations

import pytest

from app.audio.audio_buffer import AudioBuffer, AudioChunk, ChunkAssembler


# ----- ChunkAssembler ------------------------------------------------------- #


def test_chunk_assembler_emits_full_chunks_only():
    asm = ChunkAssembler(sample_rate=16000, chunk_seconds=0.5)
    # 0.5 s @ 16 kHz mono int16 = 16000 bytes.
    assert asm.bytes_per_chunk == 16000

    half = bytes(8000)
    assert asm.feed(half, timestamp=0.5) == []
    assert asm.pending_bytes == 8000

    chunks = asm.feed(half, timestamp=1.0)
    assert len(chunks) == 1
    assert len(chunks[0].data) == 16000
    assert chunks[0].sample_rate == 16000
    assert chunks[0].channels == 1
    assert chunks[0].timestamp == 1.0
    assert chunks[0].duration_s == pytest.approx(0.5)
    assert asm.pending_bytes == 0


def test_chunk_assembler_emits_multiple_chunks_at_once():
    asm = ChunkAssembler(sample_rate=16000, chunk_seconds=0.25)
    # 2.5 chunks worth of audio in one call.
    target = asm.bytes_per_chunk
    chunks = asm.feed(bytes(target * 2 + target // 2), timestamp=2.0)
    assert len(chunks) == 2
    assert all(len(c.data) == target for c in chunks)
    # The remaining half-chunk stays in the buffer for next time.
    assert asm.pending_bytes == target // 2


def test_chunk_assembler_flush_returns_partial_or_none():
    asm = ChunkAssembler(sample_rate=16000, chunk_seconds=1.0)
    assert asm.flush(timestamp=0.0) is None  # nothing pending

    asm.feed(bytes(1234), timestamp=0.1)
    tail = asm.flush(timestamp=0.5)
    assert tail is not None
    assert len(tail.data) == 1234
    assert asm.pending_bytes == 0
    # After a flush, a second flush returns None again.
    assert asm.flush(timestamp=1.0) is None


def test_chunk_assembler_stereo_doubles_bytes_per_chunk():
    asm = ChunkAssembler(sample_rate=16000, chunk_seconds=0.5, channels=2)
    # 0.5 s × 16 kHz × 2 ch × 2 bytes = 32000.
    assert asm.bytes_per_chunk == 32000


@pytest.mark.parametrize(
    "sr, secs, ch",
    [
        (0, 0.5, 1),
        (16000, 0, 1),
        (16000, -1, 1),
        (16000, 0.5, 3),
    ],
)
def test_chunk_assembler_rejects_bad_args(sr, secs, ch):
    with pytest.raises(ValueError):
        ChunkAssembler(sample_rate=sr, chunk_seconds=secs, channels=ch)


# ----- AudioBuffer ---------------------------------------------------------- #


def test_audio_buffer_fifo_round_trip():
    buf = AudioBuffer(max_chunks=4)
    a = AudioChunk(data=b"\x00" * 2, sample_rate=16000, channels=1, timestamp=0.0)
    b = AudioChunk(data=b"\x01" * 2, sample_rate=16000, channels=1, timestamp=0.1)
    buf.push(a)
    buf.push(b)
    assert len(buf) == 2
    assert buf.pop(timeout=0.1) is a
    assert buf.pop(timeout=0.1) is b
    # ``pop`` returns ``None`` after the timeout when the buffer is empty.
    assert buf.pop(timeout=0.05) is None


def test_audio_buffer_drops_oldest_when_full():
    buf = AudioBuffer(max_chunks=2)
    chunks = [
        AudioChunk(data=bytes([i]) * 2, sample_rate=16000, channels=1, timestamp=float(i))
        for i in range(3)
    ]
    for c in chunks:
        buf.push(c)
    # The oldest chunk (timestamp=0.0) was evicted by the deque's maxlen.
    assert len(buf) == 2
    assert buf.pop(timeout=0.0).timestamp == 1.0
    assert buf.pop(timeout=0.0).timestamp == 2.0
