"""TranscriptView: timestamped append + pending indicator."""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.transcription.base import TranscriptSegment  # noqa: E402
from app.ui.widgets.transcript_view import (  # noqa: E402
    TranscriptView,
    format_timestamp,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_format_timestamp_under_one_hour():
    assert format_timestamp(0.0) == "[00:00]"
    assert format_timestamp(5.0) == "[00:05]"
    assert format_timestamp(65.7) == "[01:05]"


def test_format_timestamp_past_one_hour():
    assert format_timestamp(3661) == "[1:01:01]"


def test_format_timestamp_handles_negative_and_nan():
    import math

    assert format_timestamp(-1.0) == "[00:00]"
    assert format_timestamp(float("nan")) == "[00:00]"
    assert math  # quiet linter


def test_append_segment_uses_received_s_when_present(qapp):
    view = TranscriptView()
    seg = TranscriptSegment(
        text="hello world",
        is_final=True,
        start_s=0.0,
        end_s=1.0,
        metadata={"received_s": 7.5},
    )
    view.append_segment(seg)
    assert "[00:07] hello world" in view.text()


def test_append_segment_falls_back_to_start_s(qapp):
    view = TranscriptView()
    seg = TranscriptSegment(text="bare", is_final=True, start_s=12.0, end_s=13.0)
    view.append_segment(seg)
    assert "[00:12] bare" in view.text()


def test_multiple_segments_append_on_separate_lines(qapp):
    view = TranscriptView()
    for ts, text in [(0.5, "first"), (5.5, "second")]:
        view.append_segment(
            TranscriptSegment(
                text=text,
                is_final=True,
                start_s=ts,
                end_s=ts + 0.5,
                metadata={"received_s": ts},
            )
        )
    body = view.text()
    assert "[00:00] first" in body
    assert "[00:05] second" in body
    assert body.count("\n") == 1


def test_set_pending_shows_and_hides(qapp):
    view = TranscriptView()
    view.set_pending(True, 12.0)
    assert not view._pending.isHidden()
    assert "[00:12]" in view._pending.text()
    assert "transcribing" in view._pending.text()

    view.set_pending(False)
    assert view._pending.isHidden()


def test_mark_skipped_shows_silence_label(qapp):
    view = TranscriptView()
    view.mark_skipped(20.0)
    assert not view._pending.isHidden()
    assert "silence" in view._pending.text().lower()


def test_set_text_replaces_and_clears_pending(qapp):
    view = TranscriptView()
    view.append_segment(TranscriptSegment(text="x", is_final=True))
    view.set_pending(True, 1.0)

    view.set_text("loaded note body")

    assert view.text() == "loaded note body"
    assert view._pending.isHidden()


def test_clear_resets_state(qapp):
    view = TranscriptView()
    view.append_segment(TranscriptSegment(text="x", is_final=True))
    view.set_pending(True, 1.0)

    view.clear()

    assert view.text() == ""
    assert view._pending.isHidden()
    # The next append shouldn't carry a leading newline from the prior session.
    view.append_segment(
        TranscriptSegment(
            text="y", is_final=True, start_s=0.0, end_s=1.0, metadata={"received_s": 0.0}
        )
    )
    assert view.text() == "[00:00] y"
