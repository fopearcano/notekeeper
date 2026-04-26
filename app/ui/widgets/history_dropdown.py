"""Dropdown of past LLM processing runs for the current note.

A small QComboBox + a "View" button. Selecting an entry stores it as the
current selection; clicking "View" emits :pyattr:`run_selected` with the
:class:`ProcessingRun` so the main window can pop a non-modal dialog
showing the run's prompt and output.
"""

from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from app.notes.models import ProcessingRun


class HistoryDropdown(QWidget):
    """Compact "history" picker."""

    #: Emitted when the user clicks View on a selected run.
    run_selected = Signal(object)

    PLACEHOLDER = "(no processing runs yet)"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._runs: list[ProcessingRun] = []

        self._label = QLabel("History:")

        self._combo = QComboBox(self)
        self._combo.setMinimumContentsLength(28)
        self._combo.addItem(self.PLACEHOLDER, userData=None)
        self._combo.setEnabled(False)

        self._view_button = QPushButton("View", self)
        self._view_button.setEnabled(False)
        self._view_button.clicked.connect(self._on_view_clicked)

        self._combo.currentIndexChanged.connect(self._on_combo_changed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        layout.addWidget(self._combo, stretch=1)
        layout.addWidget(self._view_button)

    # ----- public API ----------------------------------------------------

    @Slot(list)
    def set_runs(self, runs: Sequence[ProcessingRun]) -> None:
        """Replace the dropdown contents. ``runs`` is newest-first."""
        self._runs = list(runs)
        self._combo.blockSignals(True)
        try:
            self._combo.clear()
            if not self._runs:
                self._combo.addItem(self.PLACEHOLDER, userData=None)
                self._combo.setEnabled(False)
                self._view_button.setEnabled(False)
                return

            for idx, run in enumerate(self._runs):
                stamp = run.created_at.strftime("%H:%M:%S")
                self._combo.addItem(f"{stamp}  {run.task}", userData=idx)
            self._combo.setEnabled(True)
            self._view_button.setEnabled(True)
        finally:
            self._combo.blockSignals(False)

    def current_run(self) -> Optional[ProcessingRun]:
        idx = self._combo.currentData()
        if not isinstance(idx, int):
            return None
        if 0 <= idx < len(self._runs):
            return self._runs[idx]
        return None

    def clear(self) -> None:
        self.set_runs([])

    # ----- internal ------------------------------------------------------

    @Slot(int)
    def _on_combo_changed(self, _index: int) -> None:
        # Keep the View button in sync with whether there's a real run selected.
        self._view_button.setEnabled(self.current_run() is not None)

    @Slot()
    def _on_view_clicked(self) -> None:
        run = self.current_run()
        if run is not None:
            self.run_selected.emit(run)
