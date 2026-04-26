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

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QToolBar,
    QWidget,
)

from app.config.settings import AppSettings
from app.notes.models import NoteDraft
from app.notes.repository import NoteRepository
from app.services.session_manager import SessionManager, SessionState
from app.transcription.base import TranscriptSegment
from app.ui.widgets import NoteEditor, NotekeeperStatusBar, TranscriptView
from app.utils.logging import get_logger

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
    _processed_text = Signal(str)
    _state_changed = Signal(str)
    _error_raised = Signal(str)
    _level_received = Signal(float)
    _warning_received = Signal(str)
    _status_received = Signal(str)

    def __init__(self, settings: AppSettings, repository: NoteRepository):
        super().__init__()
        self.settings = settings
        self.repository = repository

        self.setWindowTitle(settings.ui.window_title)
        self.resize(settings.ui.window_width, settings.ui.window_height)

        # ----- widgets ----------------------------------------------------
        self.notebook_list = QListWidget()
        self.notebook_list.addItem(QListWidgetItem("All notes"))
        self.notebook_list.addItem(QListWidgetItem("(Notebook list placeholder)"))
        self.notebook_list.setCurrentRow(0)

        self.transcript_view = TranscriptView()
        self.note_editor = NoteEditor()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.notebook_list)
        splitter.addWidget(self.transcript_view)
        splitter.addWidget(self.note_editor)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([200, 600, 400])

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

        # ----- toolbar -----------------------------------------------------
        self._build_toolbar()

        # ----- async plumbing ---------------------------------------------
        self._worker_thread = QThread(self)
        self._worker = AsyncWorker()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.started.connect(self._on_worker_ready)
        self._worker_thread.start()

        self._session: Optional[SessionManager] = None

        # ----- cross-thread signal wiring ---------------------------------
        self._segment_received.connect(self.transcript_view.append_segment)
        self._processed_text.connect(self.note_editor.set_text)
        self._state_changed.connect(self.status.set_state)
        self._error_raised.connect(self._show_error)
        self._level_received.connect(self.status.set_level)
        self._warning_received.connect(self._on_warning)
        self._status_received.connect(self._on_status)

    # ----- toolbar ---------------------------------------------------------

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_start = QAction("Start Recording", self)
        self.act_stop = QAction("Stop Recording", self)
        self.act_save = QAction("Save Note", self)
        self.act_summarize = QAction("Summarize", self)
        self.act_organize = QAction("Organize", self)
        self.act_format = QAction("Format", self)

        self.act_stop.setEnabled(False)

        for act in (
            self.act_start,
            self.act_stop,
            self.act_save,
            self.act_summarize,
            self.act_organize,
            self.act_format,
        ):
            tb.addAction(act)

        self.act_start.triggered.connect(self._on_start)
        self.act_stop.triggered.connect(self._on_stop)
        self.act_save.triggered.connect(self._on_save)
        self.act_summarize.triggered.connect(lambda: self._on_run_action("summarize"))
        self.act_organize.triggered.connect(lambda: self._on_run_action("organize"))
        self.act_format.triggered.connect(lambda: self._on_run_action("format"))

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
        self.status.set_providers(
            self._session.transcription_provider.provider_key,
            self._session.llm_provider.provider_key,
        )
        self._state_changed.emit("Idle")
        self.status.set_message("Session manager ready")

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

    @Slot()
    def _on_stop(self) -> None:
        session = self._require_session()
        if session is None:
            return
        self.act_stop.setEnabled(False)

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
        if self._session is None:
            return
        draft = NoteDraft(
            title=f"Note from {self.windowTitle()}",
            raw_transcript=self.transcript_view.text(),
            processed_text=self.note_editor.text(),
        )
        try:
            note = self.repository.create_note(draft)
        except Exception as exc:
            self._show_error(f"Could not save note: {exc}")
            return
        self.status.set_message(f"Saved note #{note.id}")

    def _on_run_action(self, action: str) -> None:
        session = self._require_session()
        if session is None:
            return
        transcript = self.transcript_view.text().strip()
        if not transcript:
            self._show_error("Nothing to process — record or paste a transcript first.")
            return
        self.status.set_message(f"Running {action}…", timeout_ms=0)

        async def _do() -> str:
            return await session.run_action(action)

        future = self._submit(_do(), f"{action} failed")
        if future is None:
            return

        def _done(fut) -> None:
            try:
                result = fut.result()
            except Exception as exc:
                self._error_raised.emit(f"{action} failed: {exc}")
                return
            self._processed_text.emit(result)
            self.status.set_message(f"{action.capitalize()} done")

        future.add_done_callback(_done)

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
