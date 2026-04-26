"""Application status bar."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QFrame, QLabel, QStatusBar


class NotekeeperStatusBar(QStatusBar):
    """Status bar with permanent indicators for state + active providers.

    Layout (right side):

        [ ASR: faster_whisper | LLM: lmstudio ]   [ Idle ]
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._providers = QLabel("ASR: — | LLM: —")
        self._providers.setToolTip("Active transcription and LLM providers")

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)

        self._state = QLabel("Idle")

        self.addPermanentWidget(self._providers)
        self.addPermanentWidget(sep)
        self.addPermanentWidget(self._state)

        self.showMessage("Ready")

    @Slot(str)
    def set_state(self, state: str) -> None:
        self._state.setText(state)

    @Slot(str, str)
    def set_providers(self, transcription_key: str, llm_key: str) -> None:
        self._providers.setText(f"ASR: {transcription_key} | LLM: {llm_key}")

    @Slot(str)
    def set_message(self, message: str, timeout_ms: int = 5000) -> None:
        self.showMessage(message, timeout_ms)
