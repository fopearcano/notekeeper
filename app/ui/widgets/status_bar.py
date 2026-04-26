"""Application status bar."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QFrame, QLabel, QProgressBar, QStatusBar


class NotekeeperStatusBar(QStatusBar):
    """Status bar with permanent indicators for state, providers, and audio level.

    Layout (right side):

        [ ▮▮▮▯▯ ]  [ ASR: faster_whisper | LLM: lmstudio ]  | [ Idle ]
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._level = QProgressBar()
        self._level.setRange(0, 100)
        self._level.setValue(0)
        self._level.setFixedWidth(96)
        self._level.setTextVisible(False)
        self._level.setToolTip("Microphone level")

        self._providers = QLabel("ASR: — | LLM: —")
        self._providers.setToolTip("Active transcription and LLM providers")

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)

        self._state = QLabel("Idle")

        self.addPermanentWidget(self._level)
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

    @Slot(float)
    def set_level(self, level: float) -> None:
        """Set the level meter from a 0.0–1.0 RMS reading."""
        clamped = max(0.0, min(1.0, level))
        self._level.setValue(int(clamped * 100))

    @Slot(str)
    def set_message(self, message: str, timeout_ms: int = 5000) -> None:
        self.showMessage(message, timeout_ms)
