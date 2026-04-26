"""Drives a transcription provider and dispatches segments to listeners.

The pipeline pulls :class:`AudioChunk` objects out of an :class:`AudioBuffer`,
hands them to the configured :class:`TranscriptionProvider`, and forwards
emitted :class:`TranscriptSegment` objects to subscribed listeners.

Two cross-cutting features:

1. **Timestamp augmentation** — providers may emit segments with
   provider-relative offsets, but listeners want session-relative times. The
   pipeline tracks the timestamp of the most recently consumed chunk and
   stamps each segment in ``metadata['received_s']`` so the UI can keep
   transcripts aligned with wall-clock time.
2. **Stub-safe iteration** — when a provider's ``stream()`` raises
   :class:`NotImplementedError` (because it's a placeholder or its backend
   isn't wired up yet) the pipeline catches the failure, fires the warning
   callback, and shuts down cleanly instead of bringing the worker thread
   down with it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import AsyncIterator, Awaitable, Callable, Optional

from app.audio.audio_buffer import AudioBuffer, AudioChunk
from app.transcription.base import TranscriptionProvider, TranscriptSegment
from app.utils.logging import get_logger

log = get_logger(__name__)

SegmentListener = Callable[[TranscriptSegment], Awaitable[None] | None]
WarningListener = Callable[[str], None]


class TranscriptPipeline:
    def __init__(
        self,
        provider: TranscriptionProvider,
        buffer: Optional[AudioBuffer] = None,
    ):
        self.provider = provider
        # See AudioRecorder for the same gotcha — an empty AudioBuffer is falsy.
        self.buffer = buffer if buffer is not None else AudioBuffer()
        self._listeners: list[SegmentListener] = []
        self._warning_listeners: list[WarningListener] = []
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        self._started_at: Optional[float] = None
        self._last_chunk_offset: float = 0.0

    # ----- subscription --------------------------------------------------

    def add_listener(self, listener: SegmentListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: SegmentListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def add_warning_listener(self, listener: WarningListener) -> None:
        self._warning_listeners.append(listener)

    def remove_warning_listener(self, listener: WarningListener) -> None:
        try:
            self._warning_listeners.remove(listener)
        except ValueError:
            pass

    def _emit_warning(self, message: str) -> None:
        log.warning(message)
        for listener in list(self._warning_listeners):
            try:
                listener(message)
            except Exception:
                log.exception("Warning listener raised")

    # ----- chunk / segment plumbing --------------------------------------

    async def _chunk_iter(self) -> AsyncIterator[AudioChunk]:
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            chunk = await loop.run_in_executor(None, self.buffer.pop, 0.1)
            if chunk is None:
                continue
            self._last_chunk_offset = chunk.timestamp
            yield chunk

    def _stamp(self, segment: TranscriptSegment) -> TranscriptSegment:
        """Attach a session-relative ``received_s`` timestamp to ``segment``.

        Providers may not know the real audio clock; we record the offset of
        the most recently consumed chunk, which is good enough for the UI.
        """
        meta = dict(segment.metadata)
        meta.setdefault("received_s", self._last_chunk_offset)
        if self._started_at is not None:
            meta.setdefault("wall_clock_s", time.monotonic() - self._started_at)
        return replace(segment, metadata=meta)

    async def _dispatch(self, segment: TranscriptSegment) -> None:
        stamped = self._stamp(segment)
        for listener in list(self._listeners):
            try:
                result = listener(stamped)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("Transcript listener raised")

    async def run(self) -> None:
        log.info(
            "TranscriptPipeline starting (provider=%s, is_stub=%s)",
            self.provider.name,
            self.provider.is_stub,
        )
        self._started_at = time.monotonic()
        self._last_chunk_offset = 0.0

        await self.provider.start()
        if self.provider.is_stub:
            self._emit_warning(
                f"Transcription provider '{self.provider.provider_key}' is a stub — "
                "audio is being captured but transcripts are placeholder text."
            )

        try:
            async for segment in self.provider.stream(self._chunk_iter()):
                await self._dispatch(segment)
                if self._stop_event.is_set():
                    break
        except NotImplementedError as exc:
            self._emit_warning(
                f"Transcription provider '{self.provider.provider_key}' is not "
                f"implemented: {exc}. Audio is still being captured."
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit_warning(
                f"Transcription provider '{self.provider.provider_key}' failed: {exc}"
            )
            log.exception("Provider stream raised")
        finally:
            try:
                await self.provider.stop()
            except Exception:
                log.exception("Provider stop raised")
            log.info("TranscriptPipeline stopped")

    # ----- lifecycle -----------------------------------------------------

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
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
