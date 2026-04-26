"""Command bar — slash command + custom-instruction inputs.

Sits above the processed-note panel. The command edit takes a slash command
(autocompletion is left to the user typing for now); the instruction edit
holds optional ad-hoc guidance the user wants prepended onto the prompt.

Submission paths:
* Pressing ``Enter`` (or ``Ctrl+Enter``) inside either edit fires
  :pyattr:`command_submitted`.
* Clicking the **Run** button fires the same signal.

The widget never talks to the LLM directly — it just packages the user's
intent and lets the main window dispatch.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class _CommandLineEdit(QLineEdit):
    """QLineEdit that emits ``return_pressed`` on Enter or Ctrl+Enter."""

    return_pressed = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt naming
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.return_pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class CommandBar(QWidget):
    """Two-row command + instruction input."""

    #: ``(command_text, instruction_text)``
    command_submitted = Signal(str, str)

    PLACEHOLDER_COMMAND = "/clean, /summarize, /rewrite clear, /tasks …"
    PLACEHOLDER_INSTRUCTION = "Optional extra guidance for this run…"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.command_edit = _CommandLineEdit(self)
        self.command_edit.setPlaceholderText(self.PLACEHOLDER_COMMAND)
        self.command_edit.setClearButtonEnabled(True)
        self.command_edit.return_pressed.connect(self.submit)
        self.command_edit.setToolTip(
            "Type a slash command and press Enter (or Ctrl+Enter). "
            "Examples: /summarize, /tasks, /rewrite elegant."
        )

        self.instruction_edit = _CommandLineEdit(self)
        self.instruction_edit.setPlaceholderText(self.PLACEHOLDER_INSTRUCTION)
        self.instruction_edit.setClearButtonEnabled(True)
        self.instruction_edit.return_pressed.connect(self.submit)

        self.run_button = QPushButton("Run", self)
        self.run_button.setDefault(False)
        self.run_button.setAutoDefault(False)
        self.run_button.clicked.connect(self.submit)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.command_edit, stretch=2)
        row.addWidget(self.instruction_edit, stretch=3)
        row.addWidget(self.run_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)

    # ----- public API ----------------------------------------------------

    def set_busy(self, busy: bool) -> None:
        """Disable inputs while a run is in flight."""
        self.command_edit.setEnabled(not busy)
        self.instruction_edit.setEnabled(not busy)
        self.run_button.setEnabled(not busy)

    @Slot(str)
    def fill_command(self, text: str) -> None:
        """Programmatically populate the command field (e.g. from a shortcut)."""
        self.command_edit.setText(text)
        self.command_edit.setFocus()

    def command_text(self) -> str:
        return self.command_edit.text().strip()

    def instruction_text(self) -> str:
        return self.instruction_edit.text().strip()

    def clear_inputs(self) -> None:
        self.command_edit.clear()
        self.instruction_edit.clear()

    @Slot()
    def submit(self) -> None:
        """Emit ``command_submitted`` if the command field is non-empty.

        Public so the main window's ``Ctrl+Enter`` action can fire the same
        path as Enter on a focused edit / clicking *Run*.
        """
        cmd = self.command_text()
        if not cmd:
            return
        self.command_submitted.emit(cmd, self.instruction_text())
