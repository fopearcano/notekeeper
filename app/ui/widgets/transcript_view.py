"""Live transcript display.

Top area: a read-only :class:`QPlainTextEdit` of finalized segments, each
prefixed with a session-relative ``[mm:ss]`` timestamp.

Bottom area: a single-line label that shows the current chunk's
"transcribing…" placeholder while a chunk is in flight (set via
:meth:`set_pending`). Hidden when nothing is pending.

Listeners that drive this widget:
  * :meth:`append_segment` — call when a finalized :class:`TranscriptSegment`
    arrives.
  * :meth:`set_pending` — call with ``(active, timestamp_s)``; ``active=True``
    while the pipeline is waiting for the provider, ``False`` once a segment
    arrives or the chunk is VAD-skipped.
  * :meth:`set_text` — replace the whole panel when loading a saved note.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from app.transcription.base import TranscriptSegment

# Qt uses U+2029 (PARAGRAPH SEPARATOR) instead of ``\n`` inside ``selectedText``.
_QT_PARAGRAPH_SEP = " "


def format_timestamp(seconds: float) -> str:
    """Render seconds as ``[mm:ss]`` (or ``[hh:mm:ss]`` past 1 h)."""
    if seconds < 0 or seconds != seconds:  # negative or NaN
        seconds = 0.0
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"[{hours:d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def _segment_offset(segment: TranscriptSegment) -> float:
    """Pick the most useful timestamp for display.

    Prefer the pipeline-stamped ``received_s`` (session-relative); fall back
    to ``start_s`` for tests / providers that yield raw segments.
    """
    meta = getattr(segment, "metadata", None) or {}
    if "received_s" in meta and isinstance(meta["received_s"], (int, float)):
        return float(meta["received_s"])
    return float(segment.start_s)


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

        self._pending = QLabel("", self)
        self._pending.setStyleSheet("color: gray; font-style: italic;")
        self._pending.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._editor)
        layout.addWidget(self._pending)

        self._has_segments = False

    # ----- segment append ------------------------------------------------

    @Slot(object)
    def append_segment(self, segment: TranscriptSegment) -> None:
        """Append a finalized segment as ``[mm:ss] text`` on its own line."""
        text = (segment.text or "").strip()
        if not text:
            return
        line = f"{format_timestamp(_segment_offset(segment))} {text}"

        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        prefix = "" if not self._has_segments else "\n"
        fmt = QTextCharFormat()
        if not segment.is_final:
            fmt.setFontItalic(True)
        cursor.insertText(prefix + line, fmt)
        self._editor.setTextCursor(cursor)
        self._editor.ensureCursorVisible()
        self._has_segments = True

    # ----- pending indicator --------------------------------------------

    @Slot(bool, float)
    def set_pending(self, active: bool, timestamp: Optional[float] = None) -> None:
        if active:
            stamp = format_timestamp(timestamp if timestamp is not None else 0.0)
            self._pending.setText(f"{stamp} transcribing…")
            self._pending.show()
        else:
            self._pending.hide()
            self._pending.clear()

    @Slot(float)
    def mark_skipped(self, timestamp: float) -> None:
        """Briefly indicate a chunk was VAD-skipped, then hide."""
        self._pending.setText(
            f"{format_timestamp(timestamp)} (silence — skipped)"
        )
        self._pending.show()

    # ----- bulk operations ----------------------------------------------

    def clear(self) -> None:
        self._editor.clear()
        self._has_segments = False
        self.set_pending(False)

    @Slot(str)
    def set_text(self, text: str) -> None:
        """Replace the visible transcript (used when loading a note)."""
        self._editor.setPlainText(text)
        self._has_segments = bool(text)
        self.set_pending(False)

    def text(self) -> str:
        return self._editor.toPlainText()

    def selected_text(self) -> str:
        cursor = self._editor.textCursor()
        return cursor.selectedText().replace(_QT_PARAGRAPH_SEP, "\n")
