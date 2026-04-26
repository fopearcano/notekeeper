"""Application status bar."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel, QStatusBar


class NotekeeperStatusBar(QStatusBar):
    """Status bar with a permanent provider/state indicator on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._indicator = QLabel("Idle")
        self.addPermanentWidget(self._indicator)
        self.showMessage("Ready")

    @Slot(str)
    def set_state(self, state: str) -> None:
        self._indicator.setText(state)

    @Slot(str)
    def set_message(self, message: str, timeout_ms: int = 5000) -> None:
        self.showMessage(message, timeout_ms)
