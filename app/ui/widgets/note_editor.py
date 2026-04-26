"""Editor for the processed note (the result of an LLM transformation)."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget


class NoteEditor(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._title = QLabel("Processed note", self)
        self._title.setStyleSheet("font-weight: 600;")

        self._editor = QTextEdit(self)
        self._editor.setPlaceholderText(
            "LLM output (Summarize / Organize / Format) will appear here…"
        )
        self._editor.setAcceptRichText(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._editor)

    @Slot(str)
    def set_text(self, text: str) -> None:
        self._editor.setPlainText(text)

    def text(self) -> str:
        return self._editor.toPlainText()

    def clear(self) -> None:
        self._editor.clear()
