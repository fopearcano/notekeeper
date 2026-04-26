"""Editor for the processed note (the result of an LLM transformation).

Supports both atomic ``set_text`` updates and streaming via
``begin_stream`` / ``append_stream`` / ``end_stream`` so the LLM output
flows token-by-token into the panel without freezing the UI.
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget


class NoteEditor(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._title = QLabel("Processed note", self)
        self._title.setStyleSheet("font-weight: 600;")
        self._meta = QLabel("", self)
        self._meta.setStyleSheet("color: gray; font-size: 11px;")
        self._meta.setWordWrap(True)
        self._meta.hide()

        self._editor = QTextEdit(self)
        self._editor.setPlaceholderText(
            "LLM output will appear here as it streams from the model…"
        )
        self._editor.setAcceptRichText(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title)
        layout.addWidget(self._meta)
        layout.addWidget(self._editor)

    # ----- atomic updates -----------------------------------------------

    @Slot(str)
    def set_text(self, text: str) -> None:
        self._editor.setPlainText(text)

    def text(self) -> str:
        return self._editor.toPlainText()

    def selected_text(self) -> str:
        """Return whatever the user has selected in the editor, or ``""``."""
        cursor = self._editor.textCursor()
        if not cursor.hasSelection():
            return ""
        # ``selectedText`` swaps newlines for U+2029; normalise back.
        return cursor.selectedText().replace(" ", "\n")

    def clear(self) -> None:
        self._editor.clear()
        self._meta.clear()
        self._meta.hide()

    # ----- streaming ----------------------------------------------------

    @Slot()
    def begin_stream(self) -> None:
        """Reset to a clean state before a new streaming run."""
        self.clear()

    @Slot(str)
    def append_stream(self, text: str) -> None:
        """Append a delta from the model and keep the cursor at the end."""
        if not text:
            return
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._editor.setTextCursor(cursor)
        self._editor.ensureCursorVisible()

    @Slot(str, list)
    def end_stream(self, title: str, tags: Iterable[str]) -> None:
        """Show the suggested title / tags below the heading once the stream ends."""
        self._show_meta(title=title, tags=tags, label="Suggested title")

    def load_note(self, *, processed_text: str, title: str, tags: Iterable[str]) -> None:
        """Populate the panel from a saved note (no streaming animation)."""
        self.set_text(processed_text)
        self._show_meta(title=title, tags=tags, label="Title")

    def _show_meta(self, *, title: str, tags: Iterable[str], label: str) -> None:
        bits: list[str] = []
        if title:
            bits.append(f"<b>{label}:</b> {title}")
        tag_list = list(tags)
        if tag_list:
            tag_str = ", ".join(f"#{t}" for t in tag_list)
            bits.append(f"<b>Tags:</b> {tag_str}")
        if bits:
            self._meta.setText(" &nbsp; · &nbsp; ".join(bits))
            self._meta.show()
        else:
            self._meta.clear()
            self._meta.hide()
