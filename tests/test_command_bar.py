"""CommandBar widget."""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui.widgets.command_bar import CommandBar  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication(sys.argv)


def _press_enter(widget) -> None:
    """Synthesize a press of the Return key on ``widget``."""
    event = QKeyEvent(
        QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier
    )
    widget.keyPressEvent(event)


def test_clicking_run_emits_command_submitted(qapp):
    bar = CommandBar()
    captured: list[tuple[str, str]] = []
    bar.command_submitted.connect(lambda cmd, instr: captured.append((cmd, instr)))

    bar.command_edit.setText("/clean")
    bar.instruction_edit.setText("be terse")
    bar.run_button.click()

    assert captured == [("/clean", "be terse")]


def test_enter_in_command_edit_submits(qapp):
    bar = CommandBar()
    captured: list[tuple[str, str]] = []
    bar.command_submitted.connect(lambda cmd, instr: captured.append((cmd, instr)))

    bar.command_edit.setText("/summarize")
    _press_enter(bar.command_edit)

    assert captured == [("/summarize", "")]


def test_enter_in_instruction_edit_submits(qapp):
    bar = CommandBar()
    captured: list[tuple[str, str]] = []
    bar.command_submitted.connect(lambda cmd, instr: captured.append((cmd, instr)))

    bar.command_edit.setText("/rewrite clear")
    bar.instruction_edit.setText("aim for two sentences")
    _press_enter(bar.instruction_edit)

    assert captured == [("/rewrite clear", "aim for two sentences")]


def test_empty_submit_is_ignored(qapp):
    bar = CommandBar()
    captured: list[tuple[str, str]] = []
    bar.command_submitted.connect(lambda cmd, instr: captured.append((cmd, instr)))

    bar.run_button.click()
    bar.submit()

    assert captured == []


def test_set_busy_disables_inputs(qapp):
    bar = CommandBar()
    bar.set_busy(True)
    assert not bar.command_edit.isEnabled()
    assert not bar.instruction_edit.isEnabled()
    assert not bar.run_button.isEnabled()
    bar.set_busy(False)
    assert bar.command_edit.isEnabled()
    assert bar.instruction_edit.isEnabled()
    assert bar.run_button.isEnabled()


def test_fill_command_populates_and_focuses(qapp):
    bar = CommandBar()
    bar.fill_command("/tags")
    assert bar.command_edit.text() == "/tags"


def test_clear_inputs_resets_both_fields(qapp):
    bar = CommandBar()
    bar.command_edit.setText("/x")
    bar.instruction_edit.setText("y")
    bar.clear_inputs()
    assert bar.command_edit.text() == ""
    assert bar.instruction_edit.text() == ""
