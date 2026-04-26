"""Drives a transcription provider and dispatches segments to listeners.

The pipeline is async-first so it can be reused outside of Qt (tests, CLI).
The Qt layer wraps it in a worker thread that runs an event loop.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Awaitable, Callable, Optional

from app.audio.audio_buffer import AudioBuffer, AudioChunk
from app.transcription.base import TranscriptionProvider, TranscriptSegment
from app.utils.logging import get_logger

log = get_logger(__name__)

SegmentListener = Callable[[TranscriptSegment], Awaitable[None] | None]


class TranscriptPipeline:
    def __init__(
        self,
        provider: TranscriptionProvider,
        buffer: Optional[AudioBuffer] = None,
    ):
        self.provider = provider
        self.buffer = buffer or AudioBuffer()
        self._listeners: list[SegmentListener] = []
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def add_listener(self, listener: SegmentListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: SegmentListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    async def _chunk_iter(self) -> AsyncIterator[AudioChunk]:
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            # ``AudioBuffer.pop`` blocks; offload to a thread so we don't stall the loop.
            chunk = await loop.run_in_executor(None, self.buffer.pop, 0.1)
            if chunk is None:
                continue
            yield chunk

    async def _dispatch(self, segment: TranscriptSegment) -> None:
        for listener in list(self._listeners):
            try:
                result = listener(segment)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("Transcript listener raised")

    async def run(self) -> None:
        log.info("TranscriptPipeline starting (provider=%s)", self.provider.name)
        await self.provider.start()
        try:
            async for segment in self.provider.stream(self._chunk_iter()):
                await self._dispatch(segment)
                if self._stop_event.is_set():
                    break
        finally:
            await self.provider.stop()
            log.info("TranscriptPipeline stopped")

    def start(self) -> asyncio.Task:
        if self._task is not None and not self._task.done():
            return self._task
        self._stop_event.clear()
        self._task = asyncio.create_task(self.run())
        return self._task

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                log.warning("Transcript pipeline did not stop in time; cancelling")
                self._task.cancel()
            self._task = None
