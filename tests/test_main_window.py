"""Smoke test the GUI: construct the main window without starting an event loop."""

from __future__ import annotations

import os
import sys

import pytest

# Use an offscreen Qt platform so the test runs in headless CI containers.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config.settings import load_settings  # noqa: E402
from app.notes.database import Database  # noqa: E402
from app.notes.repository import NoteRepository  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_main_window_constructs(qapp):
    settings = load_settings()
    db = Database(":memory:")
    try:
        repo = NoteRepository(db)
        window = MainWindow(settings, repo)
        try:
            assert window.windowTitle() == settings.ui.window_title
            assert window.act_start.isEnabled()
            assert not window.act_stop.isEnabled()
            assert window.transcript_view is not None
            assert window.note_editor is not None
        finally:
            window.shutdown()
            window.close()
    finally:
        db.close()
