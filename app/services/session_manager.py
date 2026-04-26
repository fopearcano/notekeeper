"""Top-level coordinator for a recording + transcription session.

Owns the audio recorder, the transcript pipeline, and the persistent note
under edit. The Qt layer drives this from a worker thread so the UI never
blocks; the session manager exposes:

* listener APIs for segments, level meter, warnings, status messages.
* a single "current note" pointer that the UI uses for sidebar selection
  and the autosave timer hooks.
* :meth:`save_note` (manual or autosave) and :meth:`load_note`.
* :meth:`stream_action` which both streams LLM deltas to the UI and, on
  completion, updates the current note's processed text / title / tags
  and records a :class:`ProcessingRun`.
"""

from __future__ import annotations

import enum
import threading
from typing import Callable, Optional

from app.audio.audio_buffer import AudioBuffer
from app.audio.recorder import AudioRecorder, RecorderState
from app.audio.vad import build_default_vad
from app.config.settings import AppSettings
from app.services.health_monitor import (
    HealthMonitor,
    HealthSnapshot,
    LLMServerStatus,
)
from app.llm.factory import create_llm_provider
from app.llm.prompt_templates import StructuredOutput, render_template
from app.notes.models import (
    Note,
    NoteDraft,
    NoteSummary,
    ProcessingRunDraft,
    SegmentDraft,
)
from app.notes.repository import NoteRepository
from app.services.note_processor import NoteProcessor, StreamDelta, StreamFinal
from app.services.transcript_pipeline import TranscriptPipeline
from app.transcription.base import TranscriptSegment
from app.transcription.factory import create_transcription_provider
from app.utils.listeners import ListenerList
from app.utils.logging import get_logger

log = get_logger(__name__)

SegmentCallback = Callable[[TranscriptSegment], None]
LevelCallback = Callable[[float], None]
WarningCallback = Callable[[str], None]
StatusCallback = Callable[[str], None]
NoteSavedCallback = Callable[[Note], None]
ChunkEventCallback = Callable[[str, float, float], None]
HealthCallback = Callable[[HealthSnapshot], None]


class SessionState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPED = "stopped"


class SessionManager:
    """Coordinates a single recording session and its persistent note."""

    def __init__(self, settings: AppSettings, repository: NoteRepository):
        self.settings = settings
        self.repository = repository

        self._buffer = AudioBuffer()
        self.recorder = AudioRecorder(
            settings.audio,
            sample_rate=settings.transcription.sample_rate,
            chunk_seconds=settings.transcription.chunk_seconds,
            buffer=self._buffer,
        )
        self.recorder.set_level_callback(self._on_level)
        self.recorder.set_error_callback(self._on_recorder_error)

        self.vad = build_default_vad(settings.audio)
        self.transcription_provider = create_transcription_provider(settings)
        self.pipeline = TranscriptPipeline(
            self.transcription_provider, self._buffer, vad=self.vad
        )
        self.pipeline.add_listener(self._on_segment)
        self.pipeline.add_warning_listener(self._on_warning)
        self.pipeline.add_status_listener(self._on_status)
        self.pipeline.add_chunk_listener(self._on_chunk_event)

        self.llm_provider = create_llm_provider(settings)
        self.note_processor = NoteProcessor(self.llm_provider)
        self.health_monitor = HealthMonitor(settings.active_lmstudio_settings())

        self._state = SessionState.IDLE
        self._lock = threading.Lock()
        self._segments: list[TranscriptSegment] = []
        self._closed = False

        # Subscription channels — one ListenerList per event kind. The public
        # ``add_*_listener`` / ``remove_*_listener`` methods delegate here so
        # the locking + exception swallowing live in one place.
        self._segments_channel: ListenerList[TranscriptSegment] = ListenerList(
            label="segment"
        )
        self._level_channel: ListenerList[float] = ListenerList(label="level")
        self._warnings_channel: ListenerList[str] = ListenerList(label="warning")
        self._status_channel: ListenerList[str] = ListenerList(label="status")
        self._note_saved_channel: ListenerList[Note] = ListenerList(label="note-saved")
        self._chunk_events_channel: ListenerList[
            tuple[str, float, float]
        ] = ListenerList(label="chunk")
        # Track add→adapter mapping so ``remove_chunk_event_listener`` can
        # find the wrapper function the channel actually stored.
        self._chunk_event_adapters: dict[ChunkEventCallback, Callable[
            [tuple[str, float, float]], None
        ]] = {}
        self._health_channel: ListenerList[HealthSnapshot] = ListenerList(
            label="health"
        )
        self.health_monitor.add_listener(self._on_health_snapshot)

        # Persistent note pointer. None = nothing saved yet for this run.
        self._current_note_id: Optional[int] = None
        self._current_note: Optional[Note] = None
        # Track segment count last persisted so autosave can no-op until new
        # transcript text actually arrives.
        self._segments_persisted: int = 0

        log.info(
            "SessionManager ready (transcription=%s [stub=%s], llm=%s)",
            self.transcription_provider.provider_key,
            self.transcription_provider.is_stub,
            self.llm_provider.provider_key,
        )

    # ----- subscription --------------------------------------------------
    #
    # All subscription methods delegate to a per-channel :class:`ListenerList`
    # for thread-safe add / remove / notify with consistent exception handling.

    def add_segment_listener(self, listener: SegmentCallback) -> None:
        self._segments_channel.add(listener)

    def remove_segment_listener(self, listener: SegmentCallback) -> None:
        self._segments_channel.remove(listener)

    def add_level_listener(self, listener: LevelCallback) -> None:
        self._level_channel.add(listener)

    def remove_level_listener(self, listener: LevelCallback) -> None:
        self._level_channel.remove(listener)

    def add_warning_listener(self, listener: WarningCallback) -> None:
        self._warnings_channel.add(listener)

    def remove_warning_listener(self, listener: WarningCallback) -> None:
        self._warnings_channel.remove(listener)

    def add_status_listener(self, listener: StatusCallback) -> None:
        self._status_channel.add(listener)

    def remove_status_listener(self, listener: StatusCallback) -> None:
        self._status_channel.remove(listener)

    def add_note_saved_listener(self, listener: NoteSavedCallback) -> None:
        self._note_saved_channel.add(listener)

    def remove_note_saved_listener(self, listener: NoteSavedCallback) -> None:
        self._note_saved_channel.remove(listener)

    def add_chunk_event_listener(self, listener: ChunkEventCallback) -> None:
        # ChunkEventCallback takes three args; the channel stores tuples.
        def _adapter(payload: tuple[str, float, float]) -> None:
            listener(*payload)

        # Track the adapter under the user's listener so ``remove`` works.
        self._chunk_event_adapters[listener] = _adapter
        self._chunk_events_channel.add(_adapter)

    def remove_chunk_event_listener(self, listener: ChunkEventCallback) -> None:
        adapter = self._chunk_event_adapters.pop(listener, None)
        if adapter is not None:
            self._chunk_events_channel.remove(adapter)

    def add_health_listener(self, listener: HealthCallback) -> None:
        self._health_channel.add(listener)
        # Replay current snapshot for late subscribers (Qt signal connections
        # are made after SessionManager is constructed).
        try:
            listener(self.health_monitor.snapshot)
        except Exception:
            log.exception("Health listener raised on attach")

    def remove_health_listener(self, listener: HealthCallback) -> None:
        self._health_channel.remove(listener)

    def _on_health_snapshot(self, snapshot: HealthSnapshot) -> None:
        self._health_channel.notify(snapshot)

    # ----- internal callbacks (run on recorder / pipeline threads) ------

    def _on_segment(self, segment: TranscriptSegment) -> None:
        with self._lock:
            self._segments.append(segment)
        self._segments_channel.notify(segment)

    def _on_level(self, level: float) -> None:
        self._level_channel.notify(level)

    def _on_warning(self, message: str) -> None:
        self._warnings_channel.notify(message)

    def _on_status(self, message: str) -> None:
        self._status_channel.notify(message)

    def _on_recorder_error(self, message: str) -> None:
        self._on_warning(message)

    def _on_chunk_event(self, kind: str, timestamp: float, rms: float) -> None:
        self._chunk_events_channel.notify((kind, timestamp, rms))

    def _emit_note_saved(self, note: Note) -> None:
        self._note_saved_channel.notify(note)

    # ----- transcript inspection ----------------------------------------

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def transcription_is_stub(self) -> bool:
        return self.transcription_provider.is_stub

    def elapsed_s(self) -> float:
        """Active recording elapsed seconds (paused intervals excluded)."""
        return self.recorder.elapsed_s()

    def is_paused(self) -> bool:
        return self.recorder.is_paused

    # ----- VAD ----------------------------------------------------------

    def set_vad_enabled(self, enabled: bool) -> None:
        self.vad.set_enabled(enabled)

    def set_vad_threshold(self, threshold: float) -> None:
        self.vad.set_threshold(threshold)

    @property
    def vad_enabled(self) -> bool:
        return self.vad.enabled

    @property
    def vad_threshold(self) -> float:
        return self.vad.threshold

    # ----- input device -------------------------------------------------

    def set_input_device(self, device) -> None:
        """Update the configured input device. Applies on next ``start()``.

        Calling this while recording is a no-op so the live stream keeps
        running on the device the user originally chose; the new value will
        be picked up on the next session.
        """
        if self.recorder.is_recording:
            self._on_warning(
                "Microphone selection ignored — stop the current recording "
                "(toolbar Stop button or Ctrl+R) before switching devices, "
                "otherwise the open audio stream would have to be torn down "
                "mid-capture."
            )
            return
        self.settings.audio.input_device = device
        self.recorder.settings = self.settings.audio

    @property
    def current_note(self) -> Optional[Note]:
        return self._current_note

    def transcript_text(self) -> str:
        with self._lock:
            return " ".join(s.text for s in self._segments).strip()

    def _snapshot_segments(self) -> list[TranscriptSegment]:
        with self._lock:
            return list(self._segments)

    def _segment_count(self) -> int:
        with self._lock:
            return len(self._segments)

    def clear_transcript(self) -> None:
        with self._lock:
            self._segments.clear()

    # ----- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        """Begin a fresh recording — drops any in-memory transcript and current note."""
        with self._lock:
            if self._state in (SessionState.RECORDING, SessionState.PAUSED):
                return
            self._segments.clear()
            self._state = SessionState.RECORDING
        self._current_note_id = None
        self._current_note = None
        self._segments_persisted = 0
        self.recorder.start()
        self.pipeline.start()
        log.info("Session started")

    async def pause(self) -> None:
        """Stop capturing audio; keep transcript + pipeline alive."""
        with self._lock:
            if self._state != SessionState.RECORDING:
                return
            self._state = SessionState.PAUSED
        self.recorder.pause()
        log.info("Session paused")

    async def resume(self) -> None:
        with self._lock:
            if self._state != SessionState.PAUSED:
                return
            self._state = SessionState.RECORDING
        self.recorder.resume()
        log.info("Session resumed")

    async def stop(self) -> None:
        with self._lock:
            if self._state not in (SessionState.RECORDING, SessionState.PAUSED):
                return
            self._state = SessionState.STOPPED
        self.recorder.stop()
        await self.pipeline.stop()
        log.info("Session stopped")

    async def clear_session(self) -> None:
        """Reset the live state: stop if active, drop transcript + current note.

        The persisted note (if any) stays on disk — only the in-memory state
        and the *current note pointer* are cleared. The next save creates a
        fresh note.
        """
        if self._state in (SessionState.RECORDING, SessionState.PAUSED):
            await self.stop()
        with self._lock:
            self._segments.clear()
            self._state = SessionState.IDLE
        self._current_note_id = None
        self._current_note = None
        self._segments_persisted = 0
        log.info("Session cleared")

    async def aclose(self) -> None:
        """Tear everything down. Idempotent — safe to call from ``shutdown``.

        Order matters:

        1. Stop recording first so the audio callback can't push more
           chunks after the pipeline is gone.
        2. Stop the pipeline so any in-flight provider call settles.
        3. Stop the periodic health monitor so its asyncio task
           releases the worker loop.
        4. Close the LLM provider's httpx client.
        """
        if self._closed:
            return
        self._closed = True

        if self._state in (SessionState.RECORDING, SessionState.PAUSED):
            try:
                await self.stop()
            except Exception:
                log.exception("Error stopping recording during aclose")

        try:
            await self.health_monitor.stop()
        except Exception:
            log.exception("Error stopping health monitor")

        try:
            await self.note_processor.aclose()
        except Exception:
            log.exception("Error closing LLM provider")

        log.info("SessionManager closed")

    async def apply_settings(self, new_settings: AppSettings) -> None:
        """Replace the live providers / recorder config from ``new_settings``.

        Refuses while the session is recording or paused — config changes
        that affect the audio pipeline (sample rate, chunk seconds,
        transcription provider) must not happen mid-stream. Stop first.

        After this returns:
        * ``self.settings`` is the new settings object.
        * ``self.transcription_provider`` and ``self.llm_provider`` have
          been re-created from the new settings.
        * The pipeline is rebuilt around the new transcription provider
          and re-attached to the existing ``_buffer`` and the same
          internal callback dispatch path, so external listeners (UI
          subscribers) continue to receive events without re-subscribing.
        * The recorder picks up the new audio + sample-rate settings
          on the next :meth:`start` call.
        * VAD enabled / threshold are updated in-place.
        """
        if self._state in (SessionState.RECORDING, SessionState.PAUSED):
            raise RuntimeError(
                "Stop the recording before applying new settings — "
                "live audio settings (sample rate, chunk seconds, "
                "transcription provider) can't safely change mid-stream."
            )

        # Tear down the old LLM client so its httpx connection pool closes
        # cleanly. The transcription provider's ``stop`` is a no-op when the
        # pipeline isn't running, but call it for symmetry.
        try:
            await self.note_processor.aclose()
        except Exception:
            log.exception("Error closing previous LLM provider")
        try:
            await self.transcription_provider.stop()
        except Exception:
            log.exception("Error stopping previous transcription provider")

        self.settings = new_settings

        # Recorder: update settings + sample rate / chunk for the next start.
        self.recorder.settings = new_settings.audio
        self.recorder.sample_rate = new_settings.transcription.sample_rate
        self.recorder.chunk_seconds = new_settings.transcription.chunk_seconds

        # VAD is runtime-mutable; just update its knobs in place.
        self.vad.set_enabled(new_settings.audio.vad_enabled)
        self.vad.set_threshold(new_settings.audio.vad_threshold)

        # Rebuild the transcription provider + pipeline. The pipeline wires
        # ``provider.set_warning_callback`` / ``set_status_callback`` in its
        # __init__ so the new provider's side channels reach the existing
        # listener lists.
        self.transcription_provider = create_transcription_provider(new_settings)
        self.pipeline = TranscriptPipeline(
            self.transcription_provider, self._buffer, vad=self.vad
        )
        self.pipeline.add_listener(self._on_segment)
        self.pipeline.add_warning_listener(self._on_warning)
        self.pipeline.add_status_listener(self._on_status)
        self.pipeline.add_chunk_listener(self._on_chunk_event)

        # Rebuild the LLM provider + processor.
        self.llm_provider = create_llm_provider(new_settings)
        self.note_processor = NoteProcessor(self.llm_provider)

        # Repoint the existing health monitor at the new active server so
        # the status indicator updates without rebuilding the periodic task.
        self.health_monitor.update_settings(new_settings.active_lmstudio_settings())
        try:
            await self.health_monitor.probe_once()
        except Exception:
            log.exception("Health probe raised during apply_settings")

        log.info(
            "Settings reloaded (transcription=%s [stub=%s], llm=%s, active_server=%s)",
            self.transcription_provider.provider_key,
            self.transcription_provider.is_stub,
            self.llm_provider.provider_key,
            new_settings.active_server_label(),
        )

    async def start_health_monitor(self) -> None:
        """Begin periodic health probes (idempotent)."""
        self.health_monitor.start()
        # Kick off an immediate probe so the UI doesn't sit on UNKNOWN for
        # the full interval before the first paint.
        try:
            await self.health_monitor.probe_once()
        except Exception:
            log.exception("Initial health probe raised")

    async def select_server(self, index: int) -> None:
        """Switch the active LM Studio server and re-apply settings."""
        if not 0 <= index < len(self.settings.llm_servers):
            raise IndexError(f"server index {index} out of range")
        new_settings = self.settings.model_copy(deep=True)
        new_settings.llm.active_server = index
        await self.apply_settings(new_settings)

    # ----- persistence ---------------------------------------------------

    @staticmethod
    def _segments_to_drafts(
        segments: list[TranscriptSegment],
    ) -> list[SegmentDraft]:
        out: list[SegmentDraft] = []
        for s in segments:
            confidence = s.metadata.get("language_probability") if s.metadata else None
            out.append(
                SegmentDraft(
                    start_time=float(s.start_s),
                    end_time=float(s.end_s),
                    text=s.text,
                    confidence=(
                        float(confidence)
                        if isinstance(confidence, (int, float))
                        else None
                    ),
                )
            )
        return out

    def _detected_language(self) -> Optional[str]:
        """Most-recent non-empty language reported by the transcription provider."""
        with self._lock:
            for s in reversed(self._segments):
                lang = s.metadata.get("language") if s.metadata else None
                if isinstance(lang, str) and lang:
                    return lang
        return None

    def save_note(
        self,
        *,
        title: Optional[str] = None,
        processed_text: Optional[str] = None,
        tags: Optional[list[str]] = None,
        source: str = "recording",
    ) -> Note:
        """Manual save — creates a note on first call, updates it thereafter."""
        return self._persist_note(
            title=title,
            processed_text=processed_text,
            tags=tags,
            source=source,
            force=True,
        )

    def autosave(self) -> Optional[Note]:
        """Persist if there's new transcript content; otherwise no-op.

        Called by the UI's 30-second autosave timer while recording. Returns
        the saved :class:`Note` or ``None`` when nothing changed.
        """
        if self._segment_count() == self._segments_persisted:
            return None
        if self._segment_count() == 0:
            return None
        return self._persist_note(force=False)

    def load_note(self, note_id: int) -> Note:
        """Load a saved note as the current pointer (does not touch live state)."""
        note = self.repository.get_note(note_id)
        self._current_note_id = note.id
        self._current_note = note
        self._segments_persisted = 0  # not relevant for loaded notes
        return note

    def list_notes(self, *, limit: Optional[int] = None) -> list[NoteSummary]:
        return self.repository.list_notes(limit=limit)

    def _persist_note(
        self,
        *,
        title: Optional[str] = None,
        processed_text: Optional[str] = None,
        tags: Optional[list[str]] = None,
        source: str = "recording",
        force: bool = True,
    ) -> Note:
        segments = self._snapshot_segments()
        raw = " ".join(s.text for s in segments).strip()
        language = self._detected_language()
        # Treat the in-memory segment list as authoritative only when it's
        # non-empty. Saving while the live pipeline has produced nothing
        # (e.g. right after loading an existing note) must not wipe the
        # saved transcript or its persisted segments.
        has_live_segments = bool(segments)

        if self._current_note_id is None:
            draft = NoteDraft(
                title=title or "Untitled note",
                raw_transcript=raw,
                processed_text=processed_text or "",
                source=source,
                language=language,
                tags=tags or [],
            )
            note = self.repository.create_note(draft)
            self._current_note_id = note.id
        else:
            update_kwargs: dict[str, object] = {}
            if has_live_segments:
                update_kwargs["raw_transcript"] = raw
            if title is not None:
                update_kwargs["title"] = title
            if processed_text is not None:
                update_kwargs["processed_text"] = processed_text
            if tags is not None:
                update_kwargs["tags"] = tags
            if language is not None and has_live_segments:
                update_kwargs["language"] = language
            if force:
                update_kwargs["source"] = source
            note = self.repository.update_note(self._current_note_id, **update_kwargs)

        if has_live_segments:
            self.repository.replace_segments(
                note.id, self._segments_to_drafts(segments)
            )
            self._segments_persisted = len(segments)
        self._current_note = note
        log.info(
            "Note %d saved (%d segments, %d chars raw, force=%s, live_segments=%s)",
            note.id, len(segments), len(raw), force, has_live_segments,
        )
        self._emit_note_saved(note)
        return note

    # ----- LLM ----------------------------------------------------------

    async def run_action(
        self,
        action: Optional[str] = None,
        *,
        instruction: Optional[str] = None,
        title: Optional[str] = None,
        text: Optional[str] = None,
    ) -> StructuredOutput:
        """Buffered run — returns the parsed :class:`StructuredOutput`.

        ``text`` overrides the live transcript when provided (used by the
        command bar to feed selected text instead of the whole note).
        """
        action = action or self.settings.llm.default_task
        transcript = text if text is not None else self.transcript_text()

        self.health_monitor.set_status_override(LLMServerStatus.GENERATING)
        try:
            result = await self.note_processor.process(
                action=action,
                transcript=transcript,
                title=title,
                instruction=instruction,
            )
        except Exception as exc:
            self.health_monitor.set_status_override(
                LLMServerStatus.ERROR, error=f"{type(exc).__name__}: {exc}"
            )
            raise

        # Restore real status by triggering a fresh probe on the worker loop.
        try:
            await self.health_monitor.probe_once()
        except Exception:
            log.exception("Health probe raised after run_action")

        self._record_processing_run(
            action=action, transcript=transcript, output=result, instruction=instruction
        )
        return result

    async def stream_action(
        self,
        action: Optional[str] = None,
        *,
        instruction: Optional[str] = None,
        title: Optional[str] = None,
        text: Optional[str] = None,
    ):
        """Streaming run.

        Yields ``StreamDelta`` events as the model produces them, then a final
        ``StreamFinal`` carrying the parsed structured output. As a side
        effect the manager:

        * ensures a current note exists (auto-creating one if needed),
        * updates the note's processed_text / title / tags,
        * records a :class:`ProcessingRunDraft` row.

        ``text`` overrides the live transcript when supplied — the command
        bar uses this to run a task on a selected substring while still
        attributing the resulting processing-run row to the current note.
        """
        action = action or self.settings.llm.default_task
        transcript = text if text is not None else self.transcript_text()
        last_final: Optional[StreamFinal] = None

        self.health_monitor.set_status_override(LLMServerStatus.GENERATING)
        try:
            async for event in self.note_processor.process_streaming(
                action=action,
                transcript=transcript,
                title=title,
                instruction=instruction,
            ):
                if isinstance(event, StreamFinal):
                    last_final = event
                yield event
        except Exception as exc:
            self.health_monitor.set_status_override(
                LLMServerStatus.ERROR, error=f"{type(exc).__name__}: {exc}"
            )
            raise

        # Stream finished; refresh status from a real probe.
        try:
            await self.health_monitor.probe_once()
        except Exception:
            log.exception("Health probe raised after stream_action")

        if last_final is not None:
            self._record_processing_run(
                action=action,
                transcript=transcript,
                output=last_final.output,
                instruction=instruction,
            )

    def _record_processing_run(
        self,
        *,
        action: str,
        transcript: str,
        output: StructuredOutput,
        instruction: Optional[str],
    ) -> None:
        """Persist a processing run and update the current note's processed fields."""
        if not transcript.strip():
            return
        # Make sure there's a note to attach the run to. If the user runs an
        # LLM task before saving, create the note implicitly so the run row
        # has a valid foreign key.
        if self._current_note_id is None:
            self._persist_note(force=False)

        note_id = self._current_note_id
        if note_id is None:
            return  # _persist_note logged; nothing else we can do.

        system, user = render_template(
            action, transcript, instruction=instruction
        )
        prompt_blob = f"SYSTEM:\n{system}\n\nUSER:\n{user}"
        try:
            self.repository.add_processing_run(
                note_id,
                ProcessingRunDraft(
                    provider=self.llm_provider.provider_key,
                    model=getattr(self.llm_provider, "settings", None)
                    and getattr(self.llm_provider.settings, "model", "unknown")
                    or "unknown",
                    task=action,
                    prompt=prompt_blob,
                    output=output.raw or output.markdown,
                ),
            )
        except Exception:
            log.exception("Failed to persist processing run")

        # Reflect the LLM result on the note row itself. We only overwrite
        # the saved title when the model *explicitly* emitted one in the
        # metadata footer — the fallback first-line title from the body is
        # for display, not for persistence, so running ``clean`` on a
        # loaded note never destroys the title the user already chose.
        try:
            update_kwargs: dict[str, object] = {
                "processed_text": output.markdown,
            }
            if output.title_explicit and output.title:
                update_kwargs["title"] = output.title
            if output.tags:
                update_kwargs["tags"] = list(output.tags)
            note = self.repository.update_note(note_id, **update_kwargs)
            self._current_note = note
            self._emit_note_saved(note)
        except Exception:
            log.exception("Failed to update note after LLM run")

    async def list_llm_models(self) -> list[str]:
        return await self.llm_provider.list_models()

    def processing_runs(self):
        """Return all processing runs for the current note (newest first)."""
        if self._current_note_id is None:
            return []
        runs = self.repository.list_processing_runs(self._current_note_id)
        # The repo orders by ascending created_at; reverse so the freshest
        # entry is on top of the dropdown.
        return list(reversed(runs))

    def export_payload(self):
        """Return the bundle the export functions need for the current note.

        Returns ``None`` when there's no current note saved yet — the UI
        should refuse export in that case rather than synthesise a draft.
        """
        from app.services.exporters import ExportPayload  # local: avoid cycle

        if self._current_note is None:
            return None
        note_id = self._current_note.id
        return ExportPayload(
            note=self._current_note,
            segments=self.repository.list_segments(note_id),
            # Repo returns oldest-first, which matches the order users want
            # to read processing history in chronologically.
            processing_runs=self.repository.list_processing_runs(note_id),
        )
