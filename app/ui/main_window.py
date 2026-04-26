"""Notekeeper main window.

Layout:

    +----------------------------------------------------+
    | Toolbar: Start | Stop | Save | Summarize | …      |
    +-----------+----------------------+-----------------+
    | Notebooks | Live transcript      | Processed note  |
    | (left)    | (center)             | (right)         |
    +-----------+----------------------+-----------------+
    | Status bar                                         |
    +----------------------------------------------------+

Async work runs on a dedicated :class:`AsyncWorker` QThread so the UI never
blocks. UI updates are delivered back to the GUI thread via Qt signals.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QSplitter,
    QToolBar,
    QWidget,
)

from app.config.settings import AppSettings
from app.llm.prompt_templates import UI_TASKS
from app.notes.models import Note, NoteSummary
from app.notes.repository import NoteRepository
from app.services.note_processor import StreamDelta, StreamFinal
from app.services.session_manager import SessionManager
from app.ui.widgets import NoteEditor, NotekeeperStatusBar, TranscriptView
from app.utils.logging import get_logger

#: Autosave cadence while recording (milliseconds).
AUTOSAVE_INTERVAL_MS = 30_000

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Async worker                                                                #
# --------------------------------------------------------------------------- #


class AsyncWorker(QObject):
    """Owns an asyncio loop running on a background QThread.

    UI code submits coroutines via :meth:`submit`; results / errors are
    delivered back via Qt signals so callers stay on the GUI thread.
    """

    started = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @Slot()
    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.started.emit()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
            self._loop = None

    def submit(self, coro) -> asyncio.Future:
        if self._loop is None:
            raise RuntimeError("Worker loop is not running yet")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


# --------------------------------------------------------------------------- #
# Main window                                                                 #
# --------------------------------------------------------------------------- #


class MainWindow(QMainWindow):
    # Cross-thread plumbing: emitted from non-GUI threads (asyncio worker,
    # PortAudio callback) and consumed in the GUI thread via queued connections.
    _segment_received = Signal(object)
    _state_changed = Signal(str)
    _error_raised = Signal(str)
    _level_received = Signal(float)
    _warning_received = Signal(str)
    _status_received = Signal(str)
    # LLM streaming (worker thread → GUI thread).
    _stream_started = Signal()
    _stream_delta = Signal(str)
    _stream_finished = Signal(str, list)  # title, tags
    _models_listed = Signal(list, str)  # ids, error_message
    # Persistence events (worker thread → GUI thread).
    _note_saved = Signal(object)  # Note
    _note_loaded = Signal(object)  # Note
    _notes_listed = Signal(list)  # list[NoteSummary]

    def __init__(self, settings: AppSettings, repository: NoteRepository):
        super().__init__()
        self.settings = settings
        self.repository = repository

        self.setWindowTitle(settings.ui.window_title)
        self.resize(settings.ui.window_width, settings.ui.window_height)

        # ----- widgets ----------------------------------------------------
        self.notes_list = QListWidget()
        self.notes_list.setAlternatingRowColors(True)
        self.notes_list.itemActivated.connect(self._on_note_clicked)
        self.notes_list.itemClicked.connect(self._on_note_clicked)

        self.transcript_view = TranscriptView()
        self.note_editor = NoteEditor()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.notes_list)
        splitter.addWidget(self.transcript_view)
        splitter.addWidget(self.note_editor)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([220, 600, 400])

        container = QWidget()
        from PySide6.QtWidgets import QHBoxLayout

        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(splitter)
        self.setCentralWidget(container)

        # ----- status bar --------------------------------------------------
        self.status = NotekeeperStatusBar(self)
        self.setStatusBar(self.status)
        # Show configured providers immediately; refreshed once the session manager
        # is up in case the provider keys ever differ from the bare config.
        self.status.set_providers(settings.transcription.provider, settings.llm.provider)

        # ----- toolbar + menu bar ----------------------------------------
        self._build_toolbar()
        self._build_menu_bar()

        # ----- async plumbing ---------------------------------------------
        self._worker_thread = QThread(self)
        self._worker = AsyncWorker()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.started.connect(self._on_worker_ready)
        self._worker_thread.start()

        self._session: Optional[SessionManager] = None

        # ----- autosave timer (GUI thread → worker thread) ----------------
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(AUTOSAVE_INTERVAL_MS)
        self._autosave_timer.timeout.connect(self._on_autosave_tick)

        # ----- cross-thread signal wiring ---------------------------------
        self._segment_received.connect(self.transcript_view.append_segment)
        self._state_changed.connect(self.status.set_state)
        self._error_raised.connect(self._show_error)
        self._level_received.connect(self.status.set_level)
        self._warning_received.connect(self._on_warning)
        self._status_received.connect(self._on_status)
        self._stream_started.connect(self.note_editor.begin_stream)
        self._stream_delta.connect(self.note_editor.append_stream)
        self._stream_finished.connect(self._on_stream_finished)
        self._models_listed.connect(self._show_models_dialog)
        self._note_saved.connect(self._on_note_saved)
        self._note_loaded.connect(self._on_note_loaded)
        self._notes_listed.connect(self._refresh_sidebar)

    # ----- toolbar ---------------------------------------------------------

    #: Display labels for the LLM task toolbar buttons.
    _TASK_LABELS: dict[str, str] = {
        "clean": "Clean",
        "summarize": "Summarize",
        "organize": "Organize",
        "format_markdown": "Format Markdown",
        "extract_tasks": "Extract Tasks",
        "generate_title": "Generate Title",
    }

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_start = QAction("Start Recording", self)
        self.act_stop = QAction("Stop Recording", self)
        self.act_save = QAction("Save Note", self)
        self.act_stop.setEnabled(False)
        for act in (self.act_start, self.act_stop, self.act_save):
            tb.addAction(act)
        tb.addSeparator()

        self._task_actions: dict[str, QAction] = {}
        for task in UI_TASKS:
            label = self._TASK_LABELS.get(task, task.replace("_", " ").title())
            action = QAction(label, self)
            action.triggered.connect(lambda _checked=False, t=task: self._on_run_action(t))
            tb.addAction(action)
            self._task_actions[task] = action

        self.act_start.triggered.connect(self._on_start)
        self.act_stop.triggered.connect(self._on_stop)
        self.act_save.triggered.connect(self._on_save)

    def _build_menu_bar(self) -> None:
        bar: QMenuBar = self.menuBar()
        tools = bar.addMenu("&Tools")
        self.act_test_lmstudio = QAction("Test LM Studio Connection…", self)
        self.act_test_lmstudio.triggered.connect(self._on_test_lmstudio)
        tools.addAction(self.act_test_lmstudio)

    # ----- worker lifecycle -----------------------------------------------

    @Slot()
    def _on_worker_ready(self) -> None:
        self._session = SessionManager(self.settings, self.repository)
        self._session.add_segment_listener(
            lambda seg: self._segment_received.emit(seg)
        )
        self._session.add_level_listener(
            lambda level: self._level_received.emit(level)
        )
        self._session.add_warning_listener(
            lambda message: self._warning_received.emit(message)
        )
        self._session.add_status_listener(
            lambda message: self._status_received.emit(message)
        )
        self._session.add_note_saved_listener(
            lambda note: self._note_saved.emit(note)
        )
        self.status.set_providers(
            self._session.transcription_provider.provider_key,
            self._session.llm_provider.provider_key,
        )
        self._state_changed.emit("Idle")
        self.status.set_message("Session manager ready")
        self._reload_notes_async()

    # ----- toolbar handlers ------------------------------------------------

    def _require_session(self) -> Optional[SessionManager]:
        if self._session is None:
            self._show_error("Session not initialized yet — try again in a moment.")
            return None
        return self._session

    @Slot()
    def _on_start(self) -> None:
        session = self._require_session()
        if session is None:
            return
        self.transcript_view.clear()
        self.note_editor.clear()
        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(True)
        self._state_changed.emit("Recording")
        if session.transcription_is_stub:
            self.status.set_message(
                f"Recording started — transcription provider "
                f"'{session.transcription_provider.provider_key}' is in stub mode "
                "(no real transcripts).",
                timeout_ms=8000,
            )
        else:
            self.status.set_message("Recording started")
        self._submit(session.start(), "Failed to start recording")
        self._autosave_timer.start()

    @Slot()
    def _on_stop(self) -> None:
        session = self._require_session()
        if session is None:
            return
        self.act_stop.setEnabled(False)
        self._autosave_timer.stop()

        async def _do_stop() -> None:
            await session.stop()

        future = self._submit(_do_stop(), "Failed to stop recording")
        if future is not None:
            future.add_done_callback(lambda _f: self._after_stop())

    def _after_stop(self) -> None:
        self.act_start.setEnabled(True)
        self._state_changed.emit("Stopped")
        self.status.set_message("Recording stopped")

    @Slot()
    def _on_save(self) -> None:
        session = self._require_session()
        if session is None:
            return

        # Snapshot the editor state on the GUI thread, then persist on the
        # worker thread (the SQLite connection is single-threaded so we keep
        # writes serialized through the worker).
        title_hint = self._title_for_save(session)
        processed = self.note_editor.text()

        async def _do() -> Note:
            return session.save_note(
                title=title_hint,
                processed_text=processed,
                source="recording",
            )

        future = self._submit(_do(), "Could not save note")
        if future is None:
            return

        def _done(fut) -> None:
            try:
                note = fut.result()
            except Exception as exc:
                self._error_raised.emit(f"Could not save note: {exc}")
                return
            self.status.set_message(
                f"Saved note #{note.id} — {note.title}", timeout_ms=4000
            )

        future.add_done_callback(_done)

    def _title_for_save(self, session: SessionManager) -> str:
        if session.current_note is not None and session.current_note.title:
            return session.current_note.title
        first_line = next(
            (l.strip() for l in self.transcript_view.text().splitlines() if l.strip()),
            "",
        )
        return first_line[:60] if first_line else "Untitled note"

    # ----- autosave -----------------------------------------------------

    @Slot()
    def _on_autosave_tick(self) -> None:
        session = self._session
        if session is None:
            return

        async def _do() -> Optional[Note]:
            return session.autosave()

        future = self._submit(_do(), "Autosave failed")
        if future is None:
            return

        def _done(fut) -> None:
            try:
                note = fut.result()
            except Exception as exc:
                self._error_raised.emit(f"Autosave failed: {exc}")
                return
            if note is not None:
                self.status.set_message(
                    f"Autosaved note #{note.id}", timeout_ms=2000
                )

        future.add_done_callback(_done)

    # ----- sidebar / load -----------------------------------------------

    def _reload_notes_async(self) -> None:
        session = self._session
        if session is None:
            return

        async def _do() -> list[NoteSummary]:
            return session.list_notes()

        future = self._submit(_do(), "Could not load note list")
        if future is None:
            return

        def _done(fut) -> None:
            try:
                summaries = fut.result()
            except Exception as exc:
                self._error_raised.emit(f"Could not load note list: {exc}")
                return
            self._notes_listed.emit(list(summaries))

        future.add_done_callback(_done)

    @Slot(list)
    def _refresh_sidebar(self, summaries: list) -> None:
        self.notes_list.clear()
        for summary in summaries:
            timestamp = summary.updated_at.strftime("%Y-%m-%d %H:%M")
            label = f"{summary.title}\n{timestamp}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, summary.id)
            tooltip_bits = [summary.title, timestamp, f"source: {summary.source}"]
            if summary.tags:
                tooltip_bits.append("tags: " + ", ".join(f"#{t}" for t in summary.tags))
            item.setToolTip("\n".join(tooltip_bits))
            self.notes_list.addItem(item)

    @Slot(QListWidgetItem)
    def _on_note_clicked(self, item: QListWidgetItem) -> None:
        note_id = item.data(Qt.ItemDataRole.UserRole)
        if note_id is None:
            return
        session = self._require_session()
        if session is None:
            return

        async def _do() -> Note:
            return session.load_note(int(note_id))

        future = self._submit(_do(), f"Could not load note {note_id}")
        if future is None:
            return

        def _done(fut) -> None:
            try:
                note = fut.result()
            except Exception as exc:
                self._error_raised.emit(f"Could not load note: {exc}")
                return
            self._note_loaded.emit(note)

        future.add_done_callback(_done)

    @Slot(object)
    def _on_note_loaded(self, note: Note) -> None:
        self.transcript_view.set_text(note.raw_transcript)
        self.note_editor.load_note(
            processed_text=note.processed_text,
            title=note.title,
            tags=note.tags,
        )
        self.status.set_message(f"Loaded note #{note.id} — {note.title}")

    @Slot(object)
    def _on_note_saved(self, _note: Note) -> None:
        self._reload_notes_async()

    def _on_run_action(self, action: str) -> None:
        session = self._require_session()
        if session is None:
            return
        transcript = self.transcript_view.text().strip()
        if not transcript:
            self._show_error("Nothing to process — record or paste a transcript first.")
            return

        label = self._TASK_LABELS.get(action, action)
        self._set_tasks_enabled(False)
        self._stream_started.emit()
        self.status.set_message(f"{label}…", timeout_ms=0)

        async def _drive() -> tuple[str, list[str]]:
            title = ""
            tags: list[str] = []
            async for event in session.stream_action(action):
                if isinstance(event, StreamDelta):
                    self._stream_delta.emit(event.text)
                elif isinstance(event, StreamFinal):
                    title = event.output.title
                    tags = list(event.output.tags)
            return title, tags

        future = self._submit(_drive(), f"{action} failed")
        if future is None:
            self._set_tasks_enabled(True)
            return

        def _done(fut) -> None:
            self._set_tasks_enabled(True)
            try:
                title, tags = fut.result()
            except Exception as exc:
                self._error_raised.emit(f"{label} failed: {exc}")
                return
            self._stream_finished.emit(title, tags)
            self.status.set_message(f"{label} done", timeout_ms=4000)

        future.add_done_callback(_done)

    def _set_tasks_enabled(self, enabled: bool) -> None:
        for action in self._task_actions.values():
            action.setEnabled(enabled)

    @Slot(str, list)
    def _on_stream_finished(self, title: str, tags: list) -> None:
        self.note_editor.end_stream(title, tags)

    # ----- Test LM Studio connection --------------------------------------

    @Slot()
    def _on_test_lmstudio(self) -> None:
        session = self._require_session()
        if session is None:
            return
        self.status.set_message("Querying LM Studio /v1/models…", timeout_ms=0)

        async def _do() -> list[str]:
            return await session.list_llm_models()

        future = self._submit(_do(), "Test connection failed")
        if future is None:
            return

        def _done(fut) -> None:
            try:
                models = fut.result()
            except Exception as exc:
                self._models_listed.emit([], str(exc))
                return
            self._models_listed.emit(list(models), "")

        future.add_done_callback(_done)

    @Slot(list, str)
    def _show_models_dialog(self, models: list, error: str) -> None:
        if error:
            QMessageBox.warning(
                self,
                "LM Studio connection",
                f"Could not reach the LM Studio server at "
                f"{self.settings.lmstudio.base_url}:\n\n{error}",
            )
            self.status.set_message("LM Studio connection failed", timeout_ms=5000)
            return
        if not models:
            text = (
                "Connected to LM Studio, but the server reported no loaded "
                "models. Load a model in LM Studio's GUI and try again."
            )
        else:
            joined = "\n".join(f"  • {m}" for m in models)
            text = f"Available models on LM Studio:\n\n{joined}"
        QMessageBox.information(self, "LM Studio connection", text)
        self.status.set_message(
            f"LM Studio: {len(models)} model(s) available", timeout_ms=4000
        )

    # ----- helpers ---------------------------------------------------------

    def _submit(self, coro, error_label: str):
        if self._session is None:
            self._show_error("Session not ready")
            coro.close()
            return None
        try:
            return self._worker.submit(coro)
        except Exception as exc:
            self._error_raised.emit(f"{error_label}: {exc}")
            return None

    @Slot(str)
    def _show_error(self, message: str) -> None:
        log.error(message)
        QMessageBox.warning(self, "Notekeeper", message)

    @Slot(str)
    def _on_warning(self, message: str) -> None:
        """Non-modal warning surface — shown in the status bar instead of a dialog."""
        log.warning(message)
        self.status.set_message(message, timeout_ms=8000)

    @Slot(str)
    def _on_status(self, message: str) -> None:
        """Transient status update (model loading, latency, …)."""
        self.status.set_message(message, timeout_ms=3000)

    # ----- close ----------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the async worker and release session resources."""
        if self._session is not None:
            try:
                future = self._worker.submit(self._session.aclose())
                future.result(timeout=2.0)
            except Exception:
                log.exception("Error closing session")
            self._session = None
        self._worker.stop()
        self._worker_thread.quit()
        self._worker_thread.wait(2000)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.shutdown()
        super().closeEvent(event)


__all__ = ["MainWindow", "AsyncWorker"]


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
