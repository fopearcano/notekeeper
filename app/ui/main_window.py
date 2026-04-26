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
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app import __version__
from app.audio.recorder import list_input_devices
from app.config.settings import AppSettings
from app.llm.commands import (
    CommandError,
    command_help,
    is_command,
    merge_instructions,
    resolve_command,
)
from app.llm.prompt_templates import UI_TASKS
from app.notes.models import Note, NoteSummary, ProcessingRun
from app.notes.repository import NoteRepository
from app.services.exporters import (
    ExportPayload,
    safe_filename,
    write_export,
)
from app.services.note_processor import StreamDelta, StreamFinal
from app.services.session_manager import SessionManager
from app.ui.dialogs import DiagnosticsDialog, SettingsDialog
from app.ui.widgets import (
    CommandBar,
    HistoryDropdown,
    NoteEditor,
    NotekeeperStatusBar,
    TranscriptView,
)
from app.utils.logging import get_logger

#: Autosave cadence while recording (milliseconds).
AUTOSAVE_INTERVAL_MS = 30_000

#: Elapsed-time refresh cadence (milliseconds).
ELAPSED_TICK_MS = 500

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
    # Live-capture chunk events (pipeline → UI).
    _chunk_event = Signal(str, float, float)  # kind, timestamp, rms
    # Settings-dialog plumbing.
    _dialog_result = Signal(object, object)  # callback, result-or-exception
    _settings_applied = Signal(object)  # AppSettings
    # Health monitor → status bar.
    _health_received = Signal(object)  # HealthSnapshot

    def __init__(self, settings: AppSettings, repository: NoteRepository):
        super().__init__()
        self.settings = settings
        self.repository = repository

        # Window title carries the version so a screenshot tells you which
        # build produced a given note. Configurable base label keeps room
        # for users who rename their personal copy.
        self.setWindowTitle(f"{settings.ui.window_title} v{__version__}")
        self.resize(settings.ui.window_width, settings.ui.window_height)

        # ----- widgets ----------------------------------------------------
        self.notes_list = QListWidget()
        self.notes_list.setAlternatingRowColors(True)
        self.notes_list.itemActivated.connect(self._on_note_clicked)
        self.notes_list.itemClicked.connect(self._on_note_clicked)

        self.transcript_view = TranscriptView()
        self.note_editor = NoteEditor()
        self.command_bar = CommandBar()
        self.history_dropdown = HistoryDropdown()

        # Three export buttons live just below the command bar so they sit
        # right next to the processed-note panel they act on.
        self.export_md_button = QPushButton("Export Markdown")
        self.export_txt_button = QPushButton("Export TXT")
        self.export_json_button = QPushButton("Export JSON")
        from PySide6.QtWidgets import QHBoxLayout as _QHBoxLayout

        export_row = _QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.addWidget(self.export_md_button)
        export_row.addWidget(self.export_txt_button)
        export_row.addWidget(self.export_json_button)
        export_row.addStretch(1)

        # Right column: history picker on top, processed-note editor in the
        # middle, command bar above the export-button row at the bottom.
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.history_dropdown)
        right_layout.addWidget(self.note_editor, stretch=1)
        right_layout.addWidget(self.command_bar)
        right_layout.addLayout(export_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.notes_list)
        splitter.addWidget(self.transcript_view)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([220, 600, 460])

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

        # ----- elapsed-time tick ------------------------------------------
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(ELAPSED_TICK_MS)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

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
        self._chunk_event.connect(self._on_chunk_event)
        self.command_bar.command_submitted.connect(self._on_command_submitted)
        self.history_dropdown.run_selected.connect(self._show_processing_run)
        self._dialog_result.connect(self._deliver_dialog_result)
        self._settings_applied.connect(self._on_settings_applied)
        self._health_received.connect(self.status.set_llm_health)
        self.export_md_button.clicked.connect(lambda: self._on_export("md"))
        self.export_txt_button.clicked.connect(lambda: self._on_export("txt"))
        self.export_json_button.clicked.connect(lambda: self._on_export("json"))

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
        self.act_pause = QAction("Pause", self)
        self.act_stop = QAction("Stop Recording", self)
        self.act_clear = QAction("Clear", self)
        self.act_save = QAction("Save Note", self)
        self.act_pause.setEnabled(False)
        self.act_stop.setEnabled(False)

        self.act_save.setShortcut(QKeySequence("Ctrl+S"))

        for act in (
            self.act_start,
            self.act_pause,
            self.act_stop,
            self.act_clear,
            self.act_save,
        ):
            tb.addAction(act)

        # ``Ctrl+R`` is a single user-facing action that toggles record/stop
        # depending on session state. It's not on the toolbar (Start/Stop are
        # already there), but ``self.addAction`` makes it active everywhere.
        self.act_toggle_record = QAction("Toggle Recording", self)
        self.act_toggle_record.setShortcut(QKeySequence("Ctrl+R"))
        self.act_toggle_record.triggered.connect(self._on_toggle_record)
        self.addAction(self.act_toggle_record)

        # ``Ctrl+Shift+C`` cleans the current selection / transcript.
        self.act_clean_shortcut = QAction("Clean Transcript", self)
        self.act_clean_shortcut.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.act_clean_shortcut.triggered.connect(lambda: self._on_run_action("clean"))
        self.addAction(self.act_clean_shortcut)

        # ``Ctrl+Enter`` runs whatever is in the command bar — installed as a
        # window-wide action so it works regardless of focus.
        self.act_run_command = QAction("Run Command", self)
        self.act_run_command.setShortcut(QKeySequence("Ctrl+Return"))
        self.act_run_command.triggered.connect(self.command_bar.submit)
        self.addAction(self.act_run_command)

        tb.addSeparator()

        # Mic selector — populated lazily so we don't hit PortAudio in
        # construction tests. "System default" is row 0 with userData=None.
        tb.addWidget(QLabel(" Mic: "))
        self.mic_combo = QComboBox(self)
        self.mic_combo.setMinimumContentsLength(20)
        self.mic_combo.addItem("System default", userData=None)
        self.mic_combo.currentIndexChanged.connect(self._on_mic_changed)
        tb.addWidget(self.mic_combo)

        # VAD toggle next to the mic combo.
        self.vad_checkbox = QCheckBox("VAD", self)
        self.vad_checkbox.setToolTip(
            "Voice Activity Detection — skips silent audio chunks before "
            "transcription. Threshold lives in [audio].vad_threshold."
        )
        self.vad_checkbox.setChecked(self.settings.audio.vad_enabled)
        self.vad_checkbox.toggled.connect(self._on_vad_toggled)
        tb.addWidget(self.vad_checkbox)

        # Multi-server switcher. Only meaningful when [[llm_servers]] is
        # populated; we keep it visible-but-disabled otherwise so users see
        # how to enable it without hunting through the docs.
        tb.addWidget(QLabel(" Server: "))
        self.server_combo = QComboBox(self)
        self.server_combo.setMinimumContentsLength(20)
        self.server_combo.setToolTip(
            "Switch the active LM Studio server (from [[llm_servers]] in config)."
        )
        self.server_combo.currentIndexChanged.connect(self._on_server_changed)
        tb.addWidget(self.server_combo)

        tb.addSeparator()

        self._task_actions: dict[str, QAction] = {}
        for task in UI_TASKS:
            label = self._TASK_LABELS.get(task, task.replace("_", " ").title())
            action = QAction(label, self)
            action.triggered.connect(lambda _checked=False, t=task: self._on_run_action(t))
            tb.addAction(action)
            self._task_actions[task] = action

        self.act_start.triggered.connect(self._on_start)
        self.act_pause.triggered.connect(self._on_pause_resume)
        self.act_stop.triggered.connect(self._on_stop)
        self.act_clear.triggered.connect(self._on_clear)
        self.act_save.triggered.connect(self._on_save)

    def _build_menu_bar(self) -> None:
        bar: QMenuBar = self.menuBar()

        # ----- File menu --------------------------------------------------
        file_menu = bar.addMenu("&File")

        self.act_export_md = QAction("Export &Markdown…", self)
        self.act_export_md.setShortcut(QKeySequence("Ctrl+Shift+M"))
        self.act_export_md.triggered.connect(lambda: self._on_export("md"))
        file_menu.addAction(self.act_export_md)

        self.act_export_txt = QAction("Export &Text…", self)
        self.act_export_txt.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self.act_export_txt.triggered.connect(lambda: self._on_export("txt"))
        file_menu.addAction(self.act_export_txt)

        self.act_export_json = QAction("Export &JSON…", self)
        self.act_export_json.setShortcut(QKeySequence("Ctrl+Shift+J"))
        self.act_export_json.triggered.connect(lambda: self._on_export("json"))
        file_menu.addAction(self.act_export_json)

        self.act_export_pdf = QAction("Export &PDF…", self)
        self.act_export_pdf.setShortcut(QKeySequence("Ctrl+Shift+P"))
        self.act_export_pdf.triggered.connect(lambda: self._on_export("pdf"))
        file_menu.addAction(self.act_export_pdf)

        # ----- Edit menu --------------------------------------------------
        edit_menu = bar.addMenu("&Edit")

        self.act_copy_processed = QAction("Copy &Processed Note", self)
        self.act_copy_processed.setShortcut(QKeySequence("Ctrl+Shift+N"))
        self.act_copy_processed.triggered.connect(self._on_copy_processed)
        edit_menu.addAction(self.act_copy_processed)

        self.act_copy_transcript = QAction("Copy Raw &Transcript", self)
        self.act_copy_transcript.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.act_copy_transcript.triggered.connect(self._on_copy_transcript)
        edit_menu.addAction(self.act_copy_transcript)

        self.act_copy_selection = QAction("Copy &Selected Text", self)
        self.act_copy_selection.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_copy_selection.triggered.connect(self._on_copy_selection)
        edit_menu.addAction(self.act_copy_selection)

        # ----- Tools menu -------------------------------------------------
        tools = bar.addMenu("&Tools")

        self.act_settings = QAction("&Settings…", self)
        self.act_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.act_settings.triggered.connect(self._on_open_settings)
        tools.addAction(self.act_settings)

        tools.addSeparator()

        self.act_diagnostics = QAction("&Diagnostics…", self)
        self.act_diagnostics.triggered.connect(self._on_open_diagnostics)
        tools.addAction(self.act_diagnostics)

        self.act_test_lmstudio = QAction("Test LM Studio Connection…", self)
        self.act_test_lmstudio.triggered.connect(self._on_test_lmstudio)
        tools.addAction(self.act_test_lmstudio)

        # ----- Help menu --------------------------------------------------
        help_menu = bar.addMenu("&Help")
        self.act_about = QAction("&About Notekeeper…", self)
        self.act_about.triggered.connect(self._on_about)
        help_menu.addAction(self.act_about)

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
        self._session.add_chunk_event_listener(
            lambda kind, ts, rms: self._chunk_event.emit(kind, ts, rms)
        )
        self._session.add_health_listener(
            lambda snapshot: self._health_received.emit(snapshot)
        )
        self.status.set_providers(
            self._session.transcription_provider.provider_key,
            self._session.llm_provider.provider_key,
        )
        self._state_changed.emit("Idle")
        self.status.set_message("Session manager ready")
        self._populate_mic_combo()
        self._populate_server_combo()
        self._reload_notes_async()
        # Kick off periodic LM Studio health probes on the worker loop.
        try:
            self._submit(
                self._session.start_health_monitor(),
                "Could not start health monitor",
            )
        except Exception:
            log.exception("Could not schedule health monitor start")

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
        self.history_dropdown.clear()
        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(True)
        self.act_pause.setEnabled(True)
        self.act_pause.setText("Pause")
        self.mic_combo.setEnabled(False)
        self._state_changed.emit("Recording")
        self.status.set_elapsed(0.0)
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
        self._elapsed_timer.start()

    @Slot()
    def _on_pause_resume(self) -> None:
        session = self._require_session()
        if session is None:
            return
        if session.is_paused():
            async def _resume() -> None:
                await session.resume()

            self._submit(_resume(), "Failed to resume recording")
            self.act_pause.setText("Pause")
            self._state_changed.emit("Recording")
            self.status.set_message("Recording resumed")
            self._autosave_timer.start()
        else:
            async def _pause() -> None:
                await session.pause()

            self._submit(_pause(), "Failed to pause recording")
            self.act_pause.setText("Resume")
            self._state_changed.emit("Paused")
            self.status.set_message("Recording paused")
            self._autosave_timer.stop()

    @Slot()
    def _on_stop(self) -> None:
        session = self._require_session()
        if session is None:
            return
        self.act_stop.setEnabled(False)
        self.act_pause.setEnabled(False)
        self._autosave_timer.stop()
        self._elapsed_timer.stop()

        async def _do_stop() -> None:
            await session.stop()

        future = self._submit(_do_stop(), "Failed to stop recording")
        if future is not None:
            future.add_done_callback(lambda _f: self._after_stop())

    def _after_stop(self) -> None:
        self.act_start.setEnabled(True)
        self.act_pause.setEnabled(False)
        self.act_pause.setText("Pause")
        self.mic_combo.setEnabled(True)
        self._state_changed.emit("Stopped")
        self.status.set_message("Recording stopped")
        self.transcript_view.set_pending(False)

    @Slot()
    def _on_clear(self) -> None:
        session = self._require_session()
        if session is None:
            return
        self._autosave_timer.stop()
        self._elapsed_timer.stop()

        async def _do_clear() -> None:
            await session.clear_session()

        future = self._submit(_do_clear(), "Failed to clear session")
        if future is None:
            return

        def _done(_fut) -> None:
            self.transcript_view.clear()
            self.note_editor.clear()
            self.history_dropdown.clear()
            self.command_bar.clear_inputs()
            self.status.set_elapsed(0.0)
            self.status.set_level(0.0)
            self.act_start.setEnabled(True)
            self.act_pause.setEnabled(False)
            self.act_pause.setText("Pause")
            self.act_stop.setEnabled(False)
            self.mic_combo.setEnabled(True)
            self._state_changed.emit("Idle")
            self.status.set_message("Session cleared")

        future.add_done_callback(_done)

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
    def _on_elapsed_tick(self) -> None:
        session = self._session
        if session is None:
            return
        self.status.set_elapsed(session.elapsed_s())

    # ----- mic + VAD -----------------------------------------------------

    def _populate_mic_combo(self) -> None:
        """Populate the mic combo from sounddevice; preselect the configured device."""
        # Block signals so populating doesn't fire ``_on_mic_changed``.
        self.mic_combo.blockSignals(True)
        try:
            self.mic_combo.clear()
            self.mic_combo.addItem("System default", userData=None)
            for dev in list_input_devices():
                label = dev["name"]
                if dev.get("default"):
                    label += "  (system default)"
                self.mic_combo.addItem(label, userData=dev["index"])

            # Preselect from settings.
            current = self.settings.audio.input_device
            for i in range(self.mic_combo.count()):
                value = self.mic_combo.itemData(i)
                if (
                    value == current
                    or (isinstance(current, str) and isinstance(value, int)
                        and current == self.mic_combo.itemText(i))
                ):
                    self.mic_combo.setCurrentIndex(i)
                    break
        finally:
            self.mic_combo.blockSignals(False)

    @Slot(int)
    def _on_mic_changed(self, _index: int) -> None:
        session = self._session
        if session is None:
            return
        device = self.mic_combo.currentData()
        session.set_input_device(device)
        label = self.mic_combo.currentText()
        self.status.set_message(f"Microphone: {label}", timeout_ms=3000)

    @Slot(bool)
    def _on_vad_toggled(self, enabled: bool) -> None:
        session = self._session
        if session is None:
            return
        session.set_vad_enabled(enabled)
        self.status.set_message(
            f"VAD {'enabled' if enabled else 'disabled'} "
            f"(threshold {session.vad_threshold:.3f})",
            timeout_ms=3000,
        )

    # ----- multi-server picker ------------------------------------------

    def _populate_server_combo(self) -> None:
        """Populate the toolbar server combo from ``settings.llm_servers``."""
        self.server_combo.blockSignals(True)
        try:
            self.server_combo.clear()
            servers = self.settings.llm_servers
            if not servers:
                # Show the legacy single server but keep the combo disabled —
                # there's nothing to switch to.
                self.server_combo.addItem(
                    f"(single) {self.settings.lmstudio.base_url}",
                    userData=-1,
                )
                self.server_combo.setEnabled(False)
                return
            for idx, srv in enumerate(servers):
                label = srv.name or srv.base_url
                self.server_combo.addItem(label, userData=idx)
            active = self.settings.llm.active_server
            if 0 <= active < self.server_combo.count():
                self.server_combo.setCurrentIndex(active)
            self.server_combo.setEnabled(True)
        finally:
            self.server_combo.blockSignals(False)

    @Slot(int)
    def _on_server_changed(self, index: int) -> None:
        session = self._session
        if session is None:
            return
        target = self.server_combo.itemData(index)
        if not isinstance(target, int) or target < 0:
            return  # the "single server" fallback row
        if session.state.value in ("recording", "paused"):
            self._show_error(
                "Server switch deferred — stop the current recording first "
                "(toolbar Stop, or Ctrl+R) so the new server isn't asked to "
                "transcribe a stream it didn't see the start of."
            )
            # Roll the combo back to whatever's actually live.
            self._populate_server_combo()
            return

        async def _do() -> None:
            await session.select_server(target)

        future = self._submit(_do(), "Could not switch server")
        if future is None:
            return

        def _done(fut) -> None:
            try:
                fut.result()
            except Exception as exc:
                self._error_raised.emit(f"Could not switch server: {exc}")
                return
            new = self.settings.model_copy(deep=True)
            new.llm.active_server = target
            self._settings_applied.emit(new)
            self.settings = new
            self.status.set_message(
                f"Switched LM Studio → {new.active_server_label()}",
                timeout_ms=4000,
            )

        future.add_done_callback(_done)

    # ----- help / about --------------------------------------------------

    @Slot()
    def _on_about(self) -> None:
        """Show a small About dialog with version + active providers."""
        session = self._session
        asr = session.transcription_provider.provider_key if session else "—"
        llm = session.llm_provider.provider_key if session else "—"
        active_server = self.settings.active_server_label()
        body = (
            f"<b>Notekeeper {__version__}</b><br>"
            "Voice-driven notebook with live transcription and "
            "LLM-assisted note processing.<br><br>"
            f"<b>Transcription:</b> {asr}<br>"
            f"<b>LLM:</b> {llm}<br>"
            f"<b>Active server:</b> {active_server}<br>"
            f"<b>Config:</b> ~/.notekeeper/config.toml<br>"
        )
        QMessageBox.about(self, "About Notekeeper", body)

    # ----- export & clipboard -------------------------------------------

    _EXPORT_FILTERS: dict[str, str] = {
        "md": "Markdown (*.md)",
        "txt": "Plain text (*.txt)",
        "json": "JSON (*.json)",
        "pdf": "PDF (*.pdf)",
    }

    _EXPORT_SUFFIXES: dict[str, str] = {
        "md": ".md",
        "txt": ".txt",
        "json": ".json",
        "pdf": ".pdf",
    }

    def _on_export(self, fmt: str) -> None:
        """Common entry point for the four export formats."""
        session = self._require_session()
        if session is None:
            return
        payload: ExportPayload | None = session.export_payload()
        if payload is None:
            self._show_error(
                "Save the current note first — there's nothing to export yet."
            )
            return

        suffix = self._EXPORT_SUFFIXES[fmt]
        default_name = safe_filename(payload.note.title, suffix=suffix)
        path_str, _selected = QFileDialog.getSaveFileName(
            self,
            f"Export {fmt.upper()}",
            default_name,
            self._EXPORT_FILTERS[fmt],
        )
        if not path_str:
            return  # user cancelled
        path = Path(path_str)
        if path.suffix.lower() != suffix:
            # Add the suffix the user implicitly accepted via the filter.
            path = path.with_suffix(suffix)

        try:
            write_export(path, payload, fmt)
        except Exception as exc:
            self._show_error(f"Could not write {path}: {exc}")
            return
        self.status.set_message(
            f"Exported to {path}", timeout_ms=5000
        )

    @Slot()
    def _on_copy_processed(self) -> None:
        text = self.note_editor.text()
        self._set_clipboard(text, label="processed note")

    @Slot()
    def _on_copy_transcript(self) -> None:
        text = self.transcript_view.text()
        self._set_clipboard(text, label="raw transcript")

    @Slot()
    def _on_copy_selection(self) -> None:
        # Prefer the focused widget's selection when it has one; fall back
        # to whichever editor has *any* selection.
        focus = QApplication.focusWidget()
        text = ""
        for src in (focus, self.transcript_view, self.note_editor):
            if src is None:
                continue
            getter = getattr(src, "selected_text", None)
            if callable(getter):
                value = getter()
                if value and value.strip():
                    text = value
                    break
        if not text:
            self.status.set_message(
                "Nothing selected — drag a range in a panel first.",
                timeout_ms=3000,
            )
            return
        self._set_clipboard(text, label="selection")

    def _set_clipboard(self, text: str, *, label: str) -> None:
        if not text:
            self.status.set_message(f"No {label} to copy.", timeout_ms=3000)
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        char_count = len(text)
        self.status.set_message(
            f"Copied {label} ({char_count:,} char{'s' if char_count != 1 else ''})",
            timeout_ms=3000,
        )

    # ----- diagnostics dialog -------------------------------------------

    @Slot()
    def _on_open_diagnostics(self) -> None:
        if self._session is None:
            self._show_error("Session not initialized yet — try again in a moment.")
            return
        dialog = DiagnosticsDialog(
            self.settings, run_async=self._dialog_run_async, parent=self
        )
        dialog.show()
        if not hasattr(self, "_open_dialogs"):
            self._open_dialogs = []
        self._open_dialogs.append(dialog)
        dialog.finished.connect(lambda _r: self._open_dialogs.remove(dialog))

    # ----- chunk-event indicator ----------------------------------------

    @Slot(str, float, float)
    def _on_chunk_event(self, kind: str, timestamp: float, _rms: float) -> None:
        if kind == "received":
            self.transcript_view.set_pending(True, timestamp)
        elif kind == "transcribed":
            self.transcript_view.set_pending(False)
        elif kind == "skipped":
            self.transcript_view.mark_skipped(timestamp)
            # Auto-clear after a brief moment so the placeholder doesn't linger.
            QTimer.singleShot(800, lambda: self.transcript_view.set_pending(False))

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
        self._refresh_history_async()

    @Slot(object)
    def _on_note_saved(self, _note: Note) -> None:
        self._reload_notes_async()
        self._refresh_history_async()

    # ----- text source for actions --------------------------------------

    def _collect_action_text(self) -> str:
        """Pick the best text to feed an action.

        Priority: transcript-view selection → note-editor selection →
        whole transcript view contents.
        """
        for source in (
            self.transcript_view.selected_text(),
            self.note_editor.selected_text(),
        ):
            if source and source.strip():
                return source.strip()
        return self.transcript_view.text().strip()

    def _run_action_with_text(
        self,
        action: str,
        *,
        text: str,
        instruction: str = "",
        label: Optional[str] = None,
    ) -> None:
        """Shared streaming-action driver used by toolbar buttons + commands."""
        session = self._require_session()
        if session is None:
            return
        if not text:
            self._show_error("Nothing to process — record, type, or select some text first.")
            return

        label = label or self._TASK_LABELS.get(action, action.replace("_", " ").title())
        self._set_tasks_enabled(False)
        self.command_bar.set_busy(True)
        self._stream_started.emit()
        self.status.set_message(f"{label}…", timeout_ms=0)

        async def _drive() -> tuple[str, list[str]]:
            title = ""
            tags: list[str] = []
            async for event in session.stream_action(
                action, text=text, instruction=instruction or None
            ):
                if isinstance(event, StreamDelta):
                    self._stream_delta.emit(event.text)
                elif isinstance(event, StreamFinal):
                    title = event.output.title
                    tags = list(event.output.tags)
            return title, tags

        future = self._submit(_drive(), f"{action} failed")
        if future is None:
            self._set_tasks_enabled(True)
            self.command_bar.set_busy(False)
            return

        def _done(fut) -> None:
            self._set_tasks_enabled(True)
            self.command_bar.set_busy(False)
            try:
                title, tags = fut.result()
            except Exception as exc:
                self._error_raised.emit(f"{label} failed: {exc}")
                return
            self._stream_finished.emit(title, tags)
            self.status.set_message(f"{label} done", timeout_ms=4000)

        future.add_done_callback(_done)

    def _on_run_action(self, action: str) -> None:
        """Toolbar entry point: run ``action`` over the selection or full transcript."""
        self._run_action_with_text(action, text=self._collect_action_text())

    @Slot(str, str)
    def _on_command_submitted(self, command_text: str, instruction: str) -> None:
        """Resolve a slash command and dispatch it."""
        if not is_command(command_text):
            self._show_error(
                f"Commands must start with '/'. Try one of:\n\n{command_help()}"
            )
            return
        try:
            resolved = resolve_command(command_text)
        except CommandError as exc:
            self._show_error(f"{exc}\n\n{command_help()}")
            return

        merged = merge_instructions(resolved.instruction, instruction)
        text = self._collect_action_text()
        label = f"/{resolved.verb}" + (f" {resolved.arg}" if resolved.arg else "")
        self._run_action_with_text(
            resolved.template, text=text, instruction=merged, label=label
        )

    @Slot()
    def _on_toggle_record(self) -> None:
        """Ctrl+R: start or stop recording depending on session state."""
        if self.act_stop.isEnabled():
            self._on_stop()
        else:
            self._on_start()

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

    # ----- settings dialog ------------------------------------------------

    @Slot()
    def _on_open_settings(self) -> None:
        session = self._require_session()
        if session is None:
            return

        dialog = SettingsDialog(
            self.settings, run_async=self._dialog_run_async, parent=self
        )
        dialog.settings_saved.connect(self._on_settings_saved)
        dialog.show()
        # Hold a reference so non-modal dialogs aren't garbage-collected.
        if not hasattr(self, "_open_dialogs"):
            self._open_dialogs = []
        self._open_dialogs.append(dialog)
        dialog.finished.connect(lambda _r: self._open_dialogs.remove(dialog))

    def _dialog_run_async(self, coro, on_done) -> None:
        """Bridge the dialog's :class:`RunAsync` contract onto the worker loop."""
        future = self._submit(coro, "Settings test failed")
        if future is None:
            on_done(RuntimeError("Worker not ready"))
            return

        def _bounce(fut) -> None:
            try:
                result = fut.result()
            except Exception as exc:
                # Marshal back to the GUI thread before touching widgets.
                self._dialog_result.emit(on_done, exc)
                return
            self._dialog_result.emit(on_done, result)

        future.add_done_callback(_bounce)

    @Slot(object, object)
    def _deliver_dialog_result(self, on_done, result) -> None:
        on_done(result)

    @Slot(object)
    def _on_settings_saved(self, new_settings) -> None:
        """Apply the new settings to the live session."""
        session = self._session
        if session is None:
            return
        if session.state.value in ("recording", "paused"):
            self._show_error(
                "Settings not applied — stop the current recording first "
                "(toolbar Stop, or Ctrl+R). Live audio settings can't safely "
                "change while the pipeline is consuming chunks."
            )
            return

        async def _do() -> None:
            await session.apply_settings(new_settings)

        future = self._submit(_do(), "Could not apply settings")
        if future is None:
            return

        def _done(fut) -> None:
            try:
                fut.result()
            except Exception as exc:
                self._error_raised.emit(f"Could not apply settings: {exc}")
                return
            self.settings = new_settings
            # Refresh derived UI bits that don't go through listeners.
            self._settings_applied.emit(new_settings)

        future.add_done_callback(_done)

    @Slot(object)
    def _on_settings_applied(self, new_settings) -> None:
        self.status.set_providers(
            new_settings.transcription.provider, new_settings.llm.provider
        )
        self.vad_checkbox.blockSignals(True)
        try:
            self.vad_checkbox.setChecked(new_settings.audio.vad_enabled)
        finally:
            self.vad_checkbox.blockSignals(False)
        self._populate_mic_combo()
        self._populate_server_combo()
        self.status.set_message("Settings applied.", timeout_ms=4000)

    # ----- processing-runs history --------------------------------------

    def _refresh_history_async(self) -> None:
        session = self._session
        if session is None:
            return

        async def _do() -> list[ProcessingRun]:
            return session.processing_runs()

        future = self._submit(_do(), "Could not load processing history")
        if future is None:
            return

        def _done(fut) -> None:
            try:
                runs = fut.result()
            except Exception as exc:
                self._error_raised.emit(f"Could not load processing history: {exc}")
                return
            self.history_dropdown.set_runs(runs)

        future.add_done_callback(_done)

    @Slot(object)
    def _show_processing_run(self, run: ProcessingRun) -> None:
        """Open a non-modal dialog showing the run's prompt + output."""
        dialog = QDialog(self)
        dialog.setWindowTitle(
            f"Run #{run.id} — {run.task} ({run.created_at:%Y-%m-%d %H:%M:%S})"
        )
        dialog.resize(720, 520)

        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel(f"<b>Provider:</b> {run.provider}  ·  "
                                f"<b>Model:</b> {run.model}"))

        prompt_label = QLabel("Prompt")
        prompt_label.setStyleSheet("font-weight: 600; margin-top: 6px;")
        layout.addWidget(prompt_label)
        prompt_view = QPlainTextEdit(run.prompt)
        prompt_view.setReadOnly(True)
        layout.addWidget(prompt_view, stretch=1)

        output_label = QLabel("Output")
        output_label.setStyleSheet("font-weight: 600; margin-top: 6px;")
        layout.addWidget(output_label)
        output_view = QPlainTextEdit(run.output)
        output_view.setReadOnly(True)
        layout.addWidget(output_view, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.close)
        buttons.accepted.connect(dialog.close)
        layout.addWidget(buttons)

        dialog.show()
        # Hold a reference so the dialog stays alive after this slot returns.
        if not hasattr(self, "_open_dialogs"):
            self._open_dialogs: list[QDialog] = []
        self._open_dialogs.append(dialog)
        dialog.finished.connect(lambda _result: self._open_dialogs.remove(dialog))

    # ----- helpers ---------------------------------------------------------

    def _submit(self, coro, error_label: str):
        """Schedule ``coro`` on the worker loop, returning the future or None.

        Returning ``None`` means the operation could not be scheduled
        (no session yet, or the worker isn't accepting work). The
        coroutine is closed in that case so it doesn't leak a
        warning.
        """
        if self._session is None:
            self._show_error(
                "Notekeeper hasn't finished starting yet — wait a moment "
                "and try again."
            )
            coro.close()
            return None
        try:
            return self._worker.submit(coro)
        except Exception as exc:
            self._error_raised.emit(f"{error_label}: {exc}")
            log.exception("Worker submission failed for %s", error_label)
            return None

    def _run_on_worker(
        self,
        coro,
        *,
        on_success=None,
        error_label: str = "Operation failed",
        on_finally=None,
    ):
        """Schedule ``coro`` and wire a uniform done-callback.

        ``on_success`` and ``on_finally`` run on the worker thread when
        the future completes — they should emit Qt signals (not touch
        widgets directly) if they need to update the UI. Failures are
        funnelled through ``_error_raised`` with ``error_label`` as the
        prefix, so every async path produces the same shape of error
        dialog.

        Currently used by new call sites; the legacy hand-rolled
        callbacks scattered through ``_on_*`` slots can migrate to this
        helper one at a time without changing behaviour.
        """
        future = self._submit(coro, error_label)
        if future is None:
            if on_finally is not None:
                try:
                    on_finally()
                except Exception:
                    log.exception("on_finally raised for %s", error_label)
            return None

        def _done(fut) -> None:
            if on_finally is not None:
                try:
                    on_finally()
                except Exception:
                    log.exception("on_finally raised for %s", error_label)
            try:
                result = fut.result()
            except Exception as exc:
                self._error_raised.emit(f"{error_label}: {exc}")
                log.exception("Async error in %s", error_label)
                return
            if on_success is not None:
                try:
                    on_success(result)
                except Exception:
                    log.exception("on_success raised for %s", error_label)

        future.add_done_callback(_done)
        return future

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
        """Stop the worker thread and release every owned resource.

        Idempotent — ``closeEvent`` and an explicit ``shutdown()`` from a
        test both go through the same single-pass tear-down. Order:

        1. Stop the GUI-side timers so they don't fire mid-shutdown.
        2. ``await SessionManager.aclose()`` on the worker loop, which
           stops recording, drains the pipeline, halts the health
           monitor, and closes the LLM httpx client.
        3. Stop the worker loop and join its QThread.

        ``aclose`` itself is idempotent so a slow shutdown that times
        out doesn't leave the next call in an inconsistent state.
        """
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True

        # Quiet the GUI-thread timers first — anything they would have
        # tried to ``submit`` after this point would race the worker stop.
        try:
            self._autosave_timer.stop()
        except Exception:
            log.exception("Error stopping autosave timer")
        try:
            self._elapsed_timer.stop()
        except Exception:
            log.exception("Error stopping elapsed timer")

        if self._session is not None:
            try:
                future = self._worker.submit(self._session.aclose())
                # 5 s is enough to stop a recording, drain the pipeline,
                # and close httpx — but bounded so a stuck network call
                # can't pin the close button forever.
                future.result(timeout=5.0)
            except Exception:
                log.exception("Error closing session during shutdown")
            self._session = None

        try:
            self._worker.stop()
        except Exception:
            log.exception("Error stopping worker loop")
        self._worker_thread.quit()
        if not self._worker_thread.wait(3000):
            log.warning("Worker QThread did not exit within 3 s; terminating.")
            self._worker_thread.terminate()
            self._worker_thread.wait(1000)
        log.info("Shutdown complete")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.shutdown()
        super().closeEvent(event)


__all__ = ["MainWindow", "AsyncWorker"]


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
