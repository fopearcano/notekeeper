"""HistoryDropdown widget."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.notes.models import ProcessingRun  # noqa: E402
from app.ui.widgets.history_dropdown import HistoryDropdown  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication(sys.argv)


def _run(run_id: int, task: str, hour: int = 12) -> ProcessingRun:
    return ProcessingRun(
        id=run_id,
        note_id=1,
        provider="lmstudio",
        model="local",
        task=task,
        prompt="P",
        output="O",
        created_at=datetime(2026, 4, 26, hour, 30, 0, tzinfo=timezone.utc),
    )


def test_empty_state_shows_placeholder_and_disables_view(qapp):
    dd = HistoryDropdown()
    assert dd._combo.count() == 1
    assert dd._combo.currentText() == HistoryDropdown.PLACEHOLDER
    assert not dd._combo.isEnabled()
    assert not dd._view_button.isEnabled()


def test_set_runs_populates_combo_with_timestamp_and_task(qapp):
    dd = HistoryDropdown()
    dd.set_runs([_run(1, "summarize", hour=12), _run(2, "clean", hour=9)])

    assert dd._combo.count() == 2
    # Newest-first preserved.
    first = dd._combo.itemText(0)
    assert "12:30" in first and "summarize" in first
    second = dd._combo.itemText(1)
    assert "09:30" in second and "clean" in second
    assert dd._combo.isEnabled()
    assert dd._view_button.isEnabled()


def test_view_button_emits_run_selected(qapp):
    dd = HistoryDropdown()
    runs = [_run(1, "summarize"), _run(2, "clean")]
    dd.set_runs(runs)

    captured: list[ProcessingRun] = []
    dd.run_selected.connect(captured.append)

    dd._combo.setCurrentIndex(1)
    dd._view_button.click()

    assert captured == [runs[1]]


def test_clear_returns_to_empty_state(qapp):
    dd = HistoryDropdown()
    dd.set_runs([_run(1, "summarize")])
    dd.clear()
    assert dd._combo.count() == 1
    assert dd._combo.currentText() == HistoryDropdown.PLACEHOLDER
    assert not dd._view_button.isEnabled()


def test_current_run_returns_selected(qapp):
    dd = HistoryDropdown()
    runs = [_run(1, "a"), _run(2, "b")]
    dd.set_runs(runs)
    dd._combo.setCurrentIndex(0)
    assert dd.current_run() == runs[0]
    dd._combo.setCurrentIndex(1)
    assert dd.current_run() == runs[1]
