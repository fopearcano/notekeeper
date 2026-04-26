"""Thread-safe ring buffer for raw PCM audio chunks.

The recorder pushes PCM frames in; the transcription pipeline pops them out.
Kept deliberately simple — once real audio is wired up we can swap in a
``collections.deque`` with a configurable byte budget.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass(frozen=True)
class AudioChunk:
    """A single block of PCM audio frames.

    ``data`` is raw little-endian int16 PCM in mono at ``sample_rate`` Hz.
    """

    data: bytes
    sample_rate: int
    channels: int
    timestamp: float  # monotonic seconds since recording started

    @property
    def duration_s(self) -> float:
        bytes_per_frame = 2 * self.channels  # int16 == 2 bytes
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
