"""Tabbed settings dialog.

Four tabs: Audio, Transcription, LLM, Connection tests. The dialog reads
the current :class:`AppSettings`, lets the user edit a curated subset, and
on Save:

1. Validates the new shape via pydantic (a re-construction of
   :class:`AppSettings`).
2. Writes the merged config to ``~/.notekeeper/config.toml`` via
   :func:`app.config.io.save_settings`.
3. Emits :pyattr:`settings_saved` so the main window can call
   :meth:`SessionManager.apply_settings` on the worker loop.

API keys are **never** displayed. Each provider section that uses
``api_key_env`` lets the user edit only the variable's *name* and shows a
``(set)`` / ``(not set)`` hint resolved from the current process
environment. The Connection-tests tab calls real probes through whatever
``run_async`` callable the parent passed in — that's the only way the
dialog reaches the worker thread.
"""

from __future__ import annotations

import asyncio
import copy
import os
from typing import Any, Awaitable, Callable, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.audio.recorder import list_input_devices
from app.config.io import save_settings
from app.config.settings import AppSettings, TranscriptionProviderName
from app.services.connection_tester import (
    ProbeResult,
    probe_anthropic,
    probe_lmstudio,
    probe_openai,
    probe_transcription,
)
from app.utils.logging import get_logger


log = get_logger(__name__)

#: Callable the dialog uses to schedule a coroutine on the main worker
#: loop. Signature: ``run_async(coro, on_done=callable)``. ``on_done``
#: receives the awaited result back on the GUI thread.
RunAsync = Callable[[Awaitable[Any], Callable[[Any], None]], None]

_TRANSCRIPTION_PROVIDERS: tuple[TranscriptionProviderName, ...] = (
    "faster_whisper",
    "openai_audio",
    "lmstudio_audio",
)
_LLM_PROVIDERS = ("lmstudio", "openai", "anthropic")
_WHISPER_DEVICES = ("cuda", "cpu", "auto")
_WHISPER_COMPUTE_TYPES = ("float16", "int8_float16", "int8", "float32")
_WHISPER_MODEL_HINTS = (
    "tiny", "base", "small", "medium", "large-v2", "large-v3",
)


# --------------------------------------------------------------------------- #
# Tab widgets                                                                 #
# --------------------------------------------------------------------------- #


class _AudioTab(QWidget):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)

        self.input_device = QComboBox()
        self.input_device.addItem("System default", userData=None)
        for dev in list_input_devices():
            label = dev["name"] + ("  (system default)" if dev.get("default") else "")
            self.input_device.addItem(label, userData=dev["index"])
        self._select_input_device(settings.audio.input_device)

        self.sample_rate = QSpinBox()
        self.sample_rate.setRange(8000, 48000)
        self.sample_rate.setSingleStep(1000)
        self.sample_rate.setSuffix(" Hz")
        self.sample_rate.setValue(settings.transcription.sample_rate)

        self.chunk_seconds = QSpinBox()
        self.chunk_seconds.setRange(1, 60)
        self.chunk_seconds.setSuffix(" s")
        self.chunk_seconds.setValue(settings.transcription.chunk_seconds)

        self.vad_enabled = QCheckBox("Enable VAD (skip silent chunks)")
        self.vad_enabled.setChecked(settings.audio.vad_enabled)

        self.vad_threshold = QDoubleSpinBox()
        self.vad_threshold.setRange(0.0, 1.0)
        self.vad_threshold.setSingleStep(0.005)
        self.vad_threshold.setDecimals(3)
        self.vad_threshold.setValue(settings.audio.vad_threshold)

        form = QFormLayout(self)
        form.addRow("Input device:", self.input_device)
        form.addRow("Sample rate:", self.sample_rate)
        form.addRow("Chunk seconds:", self.chunk_seconds)
        form.addRow(self.vad_enabled)
        form.addRow("VAD threshold (RMS):", self.vad_threshold)

    def _select_input_device(self, value: Any) -> None:
        for i in range(self.input_device.count()):
            data = self.input_device.itemData(i)
            if data == value or (
                isinstance(value, str)
                and value == self.input_device.itemText(i)
            ):
                self.input_device.setCurrentIndex(i)
                return
        self.input_device.setCurrentIndex(0)

    def collect(self) -> dict[str, dict[str, Any]]:
        return {
            "audio": {
                "input_device": self.input_device.currentData(),
                "vad_enabled": self.vad_enabled.isChecked(),
                "vad_threshold": float(self.vad_threshold.value()),
            },
            "transcription": {
                "sample_rate": int(self.sample_rate.value()),
                "chunk_seconds": int(self.chunk_seconds.value()),
            },
        }


class _TranscriptionTab(QWidget):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)

        self.provider = QComboBox()
        for name in _TRANSCRIPTION_PROVIDERS:
            self.provider.addItem(name)
        idx = self.provider.findText(settings.transcription.provider)
        if idx >= 0:
            self.provider.setCurrentIndex(idx)

        self.language = QLineEdit(settings.transcription.language)
        self.language.setPlaceholderText("auto, en, fr, de, …")

        # faster-whisper specific knobs.
        self.fw_model = QComboBox()
        self.fw_model.setEditable(True)
        for hint in _WHISPER_MODEL_HINTS:
            self.fw_model.addItem(hint)
        self.fw_model.setCurrentText(settings.faster_whisper.model)

        self.fw_device = QComboBox()
        for d in _WHISPER_DEVICES:
            self.fw_device.addItem(d)
        idx = self.fw_device.findText(settings.faster_whisper.device)
        if idx >= 0:
            self.fw_device.setCurrentIndex(idx)

        self.fw_compute = QComboBox()
        for c in _WHISPER_COMPUTE_TYPES:
            self.fw_compute.addItem(c)
        idx = self.fw_compute.findText(settings.faster_whisper.compute_type)
        if idx >= 0:
            self.fw_compute.setCurrentIndex(idx)

        self.fw_fallback = QCheckBox("Allow CPU fallback if CUDA fails")
        self.fw_fallback.setChecked(settings.faster_whisper.allow_cpu_fallback)

        common = QFormLayout()
        common.addRow("Provider:", self.provider)
        common.addRow("Language:", self.language)

        fw_box = QGroupBox("faster-whisper")
        fw_form = QFormLayout(fw_box)
        fw_form.addRow("Model:", self.fw_model)
        fw_form.addRow("Device:", self.fw_device)
        fw_form.addRow("Compute type:", self.fw_compute)
        fw_form.addRow(self.fw_fallback)

        layout = QVBoxLayout(self)
        layout.addLayout(common)
        layout.addWidget(fw_box)
        layout.addStretch(1)

    def collect(self) -> dict[str, dict[str, Any]]:
        return {
            "transcription": {
                "provider": self.provider.currentText(),
                "language": self.language.text().strip() or "auto",
            },
            "faster_whisper": {
                "model": self.fw_model.currentText().strip(),
                "device": self.fw_device.currentText(),
                "compute_type": self.fw_compute.currentText(),
                "allow_cpu_fallback": self.fw_fallback.isChecked(),
            },
        }


class _LLMTab(QWidget):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)

        self.provider = QComboBox()
        for name in _LLM_PROVIDERS:
            self.provider.addItem(name)
        idx = self.provider.findText(settings.llm.provider)
        if idx >= 0:
            self.provider.setCurrentIndex(idx)

        self.lm_base_url = QLineEdit(settings.lmstudio.base_url)
        self.lm_model = QLineEdit(settings.lmstudio.model)

        self.openai_env = QLineEdit(settings.openai.api_key_env)
        self.openai_status = QLabel(self._env_status(settings.openai.api_key_env))
        self.openai_env.editingFinished.connect(
            lambda: self.openai_status.setText(self._env_status(self.openai_env.text()))
        )
        openai_row = QHBoxLayout()
        openai_row.addWidget(self.openai_env, stretch=1)
        openai_row.addWidget(self.openai_status)

        self.anthropic_env = QLineEdit(settings.anthropic.api_key_env)
        self.anthropic_status = QLabel(self._env_status(settings.anthropic.api_key_env))
        self.anthropic_env.editingFinished.connect(
            lambda: self.anthropic_status.setText(self._env_status(self.anthropic_env.text()))
        )
        anthropic_row = QHBoxLayout()
        anthropic_row.addWidget(self.anthropic_env, stretch=1)
        anthropic_row.addWidget(self.anthropic_status)

        common = QFormLayout()
        common.addRow("Provider:", self.provider)

        lm_box = QGroupBox("LM Studio")
        lm_form = QFormLayout(lm_box)
        lm_form.addRow("Base URL:", self.lm_base_url)
        lm_form.addRow("Model ID:", self.lm_model)

        openai_box = QGroupBox("OpenAI")
        openai_form = QFormLayout(openai_box)
        openai_form.addRow("API key env:", openai_row)

        anthropic_box = QGroupBox("Anthropic")
        anthropic_form = QFormLayout(anthropic_box)
        anthropic_form.addRow("API key env:", anthropic_row)

        layout = QVBoxLayout(self)
        layout.addLayout(common)
        layout.addWidget(lm_box)
        layout.addWidget(openai_box)
        layout.addWidget(anthropic_box)
        layout.addWidget(QLabel(
            "<i>API keys are read from environment variables — Notekeeper "
            "never reads or stores the key value itself.</i>"
        ))
        layout.addStretch(1)

    @staticmethod
    def _env_status(name: str) -> str:
        name = (name or "").strip()
        if not name:
            return "(no env var configured)"
        value = os.environ.get(name, "")
        return "(set)" if value else "(not set)"

    def collect(self) -> dict[str, dict[str, Any]]:
        return {
            "llm": {
                "provider": self.provider.currentText(),
            },
            "lmstudio": {
                "base_url": self.lm_base_url.text().strip(),
                "model": self.lm_model.text().strip(),
            },
            "openai": {
                "api_key_env": self.openai_env.text().strip() or "OPENAI_API_KEY",
            },
            "anthropic": {
                "api_key_env": self.anthropic_env.text().strip() or "ANTHROPIC_API_KEY",
            },
        }


class _ConnectionTab(QWidget):
    """Four test buttons + status labels driven by a parent-supplied runner."""

    PROBES: tuple[tuple[str, str, Callable[[AppSettings], Awaitable[ProbeResult]]], ...] = (
        ("transcription", "Test transcription provider", probe_transcription),
        ("lmstudio", "Test LM Studio /v1/models", probe_lmstudio),
        ("openai", "Test OpenAI /v1/models", probe_openai),
        ("anthropic", "Test Anthropic /v1/models", probe_anthropic),
    )

    def __init__(
        self,
        *,
        get_pending_settings: Callable[[], AppSettings],
        run_async: RunAsync,
        parent=None,
    ):
        super().__init__(parent)
        self._get_pending_settings = get_pending_settings
        self._run_async = run_async

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<i>Tests run against the values currently in the dialog "
            "(unsaved). Use Save first if you want them persisted.</i>"
        ))

        self._labels: dict[str, QLabel] = {}
        self._buttons: dict[str, QPushButton] = {}
        for key, label, _probe in self.PROBES:
            row = QHBoxLayout()
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, k=key: self._run(k))
            status = QLabel("—")
            status.setWordWrap(True)
            status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row.addWidget(button)
            row.addWidget(status, stretch=1)
            layout.addLayout(row)
            self._labels[key] = status
            self._buttons[key] = button

        layout.addStretch(1)

    def _run(self, key: str) -> None:
        try:
            settings = self._get_pending_settings()
        except Exception as exc:
            self._labels[key].setText(f"⚠ Could not validate dialog values: {exc}")
            return

        probe = next(p for k, _l, p in self.PROBES if k == key)
        button = self._buttons[key]
        label = self._labels[key]
        button.setEnabled(False)
        label.setText("Testing…")

        def _on_done(result: Any) -> None:
            button.setEnabled(True)
            if isinstance(result, Exception):
                label.setText(f"⚠ {result}")
                return
            assert isinstance(result, ProbeResult)
            prefix = "✓" if result.ok else "✗"
            label.setText(f"{prefix} {result.message}")

        try:
            self._run_async(probe(settings), _on_done)
        except Exception as exc:
            button.setEnabled(True)
            label.setText(f"⚠ Could not schedule test: {exc}")


# --------------------------------------------------------------------------- #
# Dialog                                                                      #
# --------------------------------------------------------------------------- #


class SettingsDialog(QDialog):
    """Tabbed settings editor.

    ``run_async`` is the callable the dialog uses to fire connection tests
    on the main worker loop; ``save_to_disk`` defaults to
    :func:`app.config.io.save_settings` and is parametrized so tests can
    redirect writes to a temp path.
    """

    settings_saved = Signal(object)  # AppSettings

    def __init__(
        self,
        settings: AppSettings,
        *,
        run_async: Optional[RunAsync] = None,
        save_to_disk: Callable[[AppSettings], Any] = save_settings,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Notekeeper settings")
        self.resize(600, 540)

        self._initial_settings = settings
        self._save_to_disk = save_to_disk
        self._run_async = run_async or self._noop_runner

        self.audio_tab = _AudioTab(settings)
        self.transcription_tab = _TranscriptionTab(settings)
        self.llm_tab = _LLMTab(settings)
        self.connection_tab = _ConnectionTab(
            get_pending_settings=self.pending_settings,
            run_async=self._run_async,
        )

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.audio_tab, "Audio")
        self.tabs.addTab(self.transcription_tab, "Transcription")
        self.tabs.addTab(self.llm_tab, "LLM")
        self.tabs.addTab(self.connection_tab, "Connections")

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._on_save)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.buttons)

    # ----- public --------------------------------------------------------

    def collect_overrides(self) -> dict[str, dict[str, Any]]:
        """Merge the dialog's tab overrides into a single dict."""
        merged: dict[str, dict[str, Any]] = {}
        for tab in (self.audio_tab, self.transcription_tab, self.llm_tab):
            for section, fields in tab.collect().items():
                merged.setdefault(section, {}).update(fields)
        return merged

    def pending_settings(self) -> AppSettings:
        """Build (and validate) an :class:`AppSettings` from the current tab values.

        Used by the Connections tab so tests run against the state the user
        is about to save, not whatever's on disk.
        """
        base = self._initial_settings.model_dump()
        overrides = self.collect_overrides()
        for section, fields in overrides.items():
            base.setdefault(section, {}).update(fields)
        return AppSettings.model_validate(base)

    # ----- internals -----------------------------------------------------

    @Slot()
    def _on_save(self) -> None:
        try:
            new_settings = self.pending_settings()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid settings",
                f"Some settings are invalid:\n\n{exc}",
            )
            return

        try:
            self._save_to_disk(new_settings)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Could not save settings",
                f"Notekeeper could not write the config file:\n\n{exc}",
            )
            return

        self.settings_saved.emit(new_settings)
        self.accept()

    @staticmethod
    def _noop_runner(coro: Awaitable[Any], on_done: Callable[[Any], None]) -> None:
        """Default runner used in tests: cancel the coroutine, return an error."""
        if asyncio.iscoroutine(coro):
            coro.close()
        on_done(RuntimeError("No async runner attached to the dialog"))


# Keep the deepcopy helper around even if unused — saves a future foot-gun
# where a tab returns a mutable that gets stomped during merge.
_ = copy.deepcopy
