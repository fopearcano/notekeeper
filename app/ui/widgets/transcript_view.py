"""Live transcript display.

A read-only text area that appends finalized segments and shows the most
recent partial in a muted style. The widget owns no business logic — the
session manager pushes segments to it via ``append_segment``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from app.transcription.base import TranscriptSegment

# Qt uses U+2029 (PARAGRAPH SEPARATOR) instead of '\n' inside ``selectedText``.
_QT_PARAGRAPH_SEP = " "


class TranscriptView(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._editor = QPlainTextEdit(self)
        self._editor.setReadOnly(True)
        self._editor.setPlaceholderText("Live transcript will appear here…")
        self._editor.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._editor)

    @Slot(object)
    def append_segment(self, segment: TranscriptSegment) -> None:
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        prefix = "" if not self._editor.toPlainText() else " "
        fmt = QTextCharFormat()
        if not segment.is_final:
            fmt.setFontItalic(True)
        cursor.insertText(prefix + segment.text, fmt)
        self._editor.setTextCursor(cursor)
        self._editor.ensureCursorVisible()

    def clear(self) -> None:
        self._editor.clear()

    @Slot(str)
    def set_text(self, text: str) -> None:
        """Replace the visible transcript with ``text`` (used when loading a note)."""
        self._editor.setPlainText(text)

    def text(self) -> str:
        return self._editor.toPlainText()

    def selected_text(self) -> str:
        cursor = self._editor.textCursor()
        return cursor.selectedText().replace(_QT_PARAGRAPH_SEP, "\n")
