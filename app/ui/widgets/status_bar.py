"""Application status bar."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QFrame, QLabel, QProgressBar, QStatusBar


def _format_elapsed(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        seconds = 0.0
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class NotekeeperStatusBar(QStatusBar):
    """Status bar with permanent indicators for state, providers, level, and elapsed.

    Layout (right side):

        [⏱ 00:42] [▮▮▮▯▯] [ASR: faster_whisper | LLM: lmstudio] | [Idle]
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._elapsed = QLabel("00:00")
        self._elapsed.setToolTip("Active recording time (paused intervals excluded)")
        self._elapsed.setStyleSheet("font-family: monospace;")

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

        self.addPermanentWidget(self._elapsed)
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
        clamped = max(0.0, min(1.0, level))
        self._level.setValue(int(clamped * 100))

    @Slot(float)
    def set_elapsed(self, seconds: float) -> None:
        self._elapsed.setText(_format_elapsed(seconds))

    @Slot(str)
    def set_message(self, message: str, timeout_ms: int = 5000) -> None:
        self.showMessage(message, timeout_ms)
