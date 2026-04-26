"""Thread-safe ring buffer + chunk assembly for raw PCM audio.

The recorder pushes finished :class:`AudioChunk` objects into an
:class:`AudioBuffer`; the transcription pipeline pops them out. The
:class:`ChunkAssembler` helper handles the inner accumulation step so the
recorder's audio callback stays a thin shim.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

# 16-bit PCM, the format we standardize on internally.
_BYTES_PER_SAMPLE = 2


@dataclass(frozen=True)
class AudioChunk:
    """A single block of PCM audio frames.

    ``data`` is raw little-endian int16 PCM at ``sample_rate`` Hz, interleaved
    if ``channels > 1``. ``timestamp`` is monotonic seconds since the recorder
    started — set at the moment the chunk was completed.
    """

    data: bytes
    sample_rate: int
    channels: int
    timestamp: float

    @property
    def duration_s(self) -> float:
        bytes_per_frame = _BYTES_PER_SAMPLE * self.channels
        return len(self.data) / bytes_per_frame / self.sample_rate


class AudioBuffer:
    """Bounded thread-safe queue of :class:`AudioChunk` objects."""

    def __init__(self, max_chunks: int = 1024):
        self._chunks: Deque[AudioChunk] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def push(self, chunk: AudioChunk) -> None:
        with self._not_empty:
            self._chunks.append(chunk)
            self._not_empty.notify()

    def pop(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        with self._not_empty:
            if not self._chunks:
                self._not_empty.wait(timeout=timeout)
            if not self._chunks:
                return None
            return self._chunks.popleft()

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)


class ChunkAssembler:
    """Accumulates int16 PCM bytes and emits :class:`AudioChunk` of fixed length.

    The audio callback drops in raw PCM via :meth:`feed`; whenever enough
    samples have arrived to make up ``chunk_seconds`` of audio the assembler
    returns one or more chunks. :meth:`flush` returns whatever partial buffer
    is left (or ``None``).

    Not thread-safe — own one assembler per recorder, called only from the
    audio callback.
    """

    def __init__(self, sample_rate: int, chunk_seconds: float, channels: int = 1):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        if channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")

        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.channels = channels
        self._frames_per_chunk = int(round(sample_rate * chunk_seconds))
        self._bytes_per_chunk = self._frames_per_chunk * _BYTES_PER_SAMPLE * channels
        self._buf = bytearray()

    @property
    def bytes_per_chunk(self) -> int:
        return self._bytes_per_chunk

    @property
    def pending_bytes(self) -> int:
        return len(self._buf)

    def feed(self, pcm: bytes, *, timestamp: float) -> list[AudioChunk]:
        """Append ``pcm`` and return any newly completed chunks.

        ``timestamp`` is the current monotonic offset; each emitted chunk is
        tagged with the offset at which *that chunk's last frame* arrived,
        which is what listeners need to align transcripts with wall-clock time.
        """
        self._buf.extend(pcm)
        out: list[AudioChunk] = []
        if self._bytes_per_chunk <= 0:
            return out
        while len(self._buf) >= self._bytes_per_chunk:
            chunk_bytes = bytes(self._buf[: self._bytes_per_chunk])
            del self._buf[: self._bytes_per_chunk]
            out.append(
                AudioChunk(
                    data=chunk_bytes,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    timestamp=timestamp,
                )
            )
        return out

    def flush(self, *, timestamp: float) -> Optional[AudioChunk]:
        if not self._buf:
            return None
        chunk = AudioChunk(
            data=bytes(self._buf),
            sample_rate=self.sample_rate,
            channels=self.channels,
            timestamp=timestamp,
        )
        self._buf.clear()
        return chunk
