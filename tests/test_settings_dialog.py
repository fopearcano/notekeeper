"""SettingsDialog smoke tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config.settings import load_settings  # noqa: E402
from app.ui.dialogs.settings_dialog import SettingsDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication(sys.argv)


def _build(settings, **kwargs):
    return SettingsDialog(settings, **kwargs)


# ---------- construction --------------------------------------------------


def test_dialog_constructs_with_four_tabs(qapp):
    settings = load_settings(bootstrap=False)
    dlg = _build(settings)
    try:
        labels = [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())]
        assert labels == ["Audio", "Transcription", "LLM", "Connections"]
    finally:
        dlg.close()


def test_audio_tab_populates_from_settings(qapp):
    settings = load_settings(bootstrap=False, overrides={
        "audio": {"vad_enabled": False, "vad_threshold": 0.05},
        "transcription": {"sample_rate": 22050, "chunk_seconds": 7},
    })
    dlg = _build(settings)
    try:
        assert dlg.audio_tab.vad_enabled.isChecked() is False
        assert dlg.audio_tab.vad_threshold.value() == pytest.approx(0.05)
        assert dlg.audio_tab.sample_rate.value() == 22050
        assert dlg.audio_tab.chunk_seconds.value() == 7
    finally:
        dlg.close()


def test_transcription_tab_populates_from_settings(qapp):
    settings = load_settings(bootstrap=False, overrides={
        "transcription": {"provider": "openai_audio", "language": "en"},
        "faster_whisper": {
            "model": "medium", "device": "cpu", "compute_type": "int8",
            "allow_cpu_fallback": False,
        },
    })
    dlg = _build(settings)
    try:
        assert dlg.transcription_tab.provider.currentText() == "openai_audio"
        assert dlg.transcription_tab.language.text() == "en"
        assert dlg.transcription_tab.fw_model.currentText() == "medium"
        assert dlg.transcription_tab.fw_device.currentText() == "cpu"
        assert dlg.transcription_tab.fw_compute.currentText() == "int8"
        assert dlg.transcription_tab.fw_fallback.isChecked() is False
    finally:
        dlg.close()


def test_llm_tab_never_displays_api_key(qapp, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-do-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-secret-do-not-leak")
    settings = load_settings(bootstrap=False)
    dlg = _build(settings)
    try:
        # The status labels show "(set)" but never the actual key value.
        assert dlg.llm_tab.openai_status.text() == "(set)"
        assert dlg.llm_tab.anthropic_status.text() == "(set)"
        # Walk every visible widget text to be sure.
        for widget in dlg.findChildren(object):
            text = getattr(widget, "text", lambda: "")
            try:
                value = text()
            except TypeError:
                continue
            if isinstance(value, str):
                assert "sk-secret-do-not-leak" not in value
                assert "ak-secret-do-not-leak" not in value
    finally:
        dlg.close()


def test_llm_tab_status_updates_when_env_var_changes(qapp, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MY_OWN_KEY", "x")
    settings = load_settings(bootstrap=False)
    dlg = _build(settings)
    try:
        assert dlg.llm_tab.openai_status.text() == "(not set)"
        dlg.llm_tab.openai_env.setText("MY_OWN_KEY")
        dlg.llm_tab.openai_env.editingFinished.emit()
        assert dlg.llm_tab.openai_status.text() == "(set)"
    finally:
        dlg.close()


# ---------- collect / save -------------------------------------------------


def test_pending_settings_validates_overrides(qapp):
    settings = load_settings(bootstrap=False)
    dlg = _build(settings)
    try:
        dlg.audio_tab.sample_rate.setValue(22050)
        dlg.transcription_tab.fw_model.setCurrentText("tiny")
        dlg.llm_tab.lm_base_url.setText("http://localhost:1234/v1")
        new = dlg.pending_settings()
        assert new.transcription.sample_rate == 22050
        assert new.faster_whisper.model == "tiny"
        assert new.lmstudio.base_url == "http://localhost:1234/v1"
    finally:
        dlg.close()


def test_save_writes_to_disk_and_emits_signal(qapp, tmp_path: Path):
    settings = load_settings(bootstrap=False)
    captured: list = []
    written: list = []

    def _save_to_disk(s):
        written.append(s)
        return tmp_path / "fake.toml"

    dlg = SettingsDialog(settings, save_to_disk=_save_to_disk)
    dlg.settings_saved.connect(captured.append)

    try:
        dlg.audio_tab.sample_rate.setValue(22050)
        dlg._on_save()
        assert len(captured) == 1
        assert captured[0].transcription.sample_rate == 22050
        assert len(written) == 1
        # The dialog accepted (closed itself).
        assert dlg.result() == dlg.DialogCode.Accepted.value
    finally:
        dlg.close()


def test_save_failure_keeps_dialog_open(qapp, monkeypatch):
    settings = load_settings(bootstrap=False)

    def _boom(_settings):
        raise OSError("disk full")

    dlg = SettingsDialog(settings, save_to_disk=_boom)
    captured: list = []
    dlg.settings_saved.connect(captured.append)

    # Suppress the error dialog popup.
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: None)

    dlg._on_save()
    assert captured == []  # signal must not fire on failed save
    assert dlg.result() != dlg.DialogCode.Accepted.value
    dlg.close()


# ---------- connection tab -----------------------------------------------


def test_connection_tab_runs_probe_through_runner(qapp):
    """Clicking a test button calls run_async with a coroutine."""
    settings = load_settings(bootstrap=False)
    captured = {"calls": []}

    def runner(coro, on_done):
        captured["calls"].append(("scheduled", type(coro).__name__))
        # Complete immediately with a cooked result so the label updates.
        from app.services.connection_tester import ProbeResult

        coro.close()  # drop the unawaited coroutine
        on_done(ProbeResult(True, "all good"))

    dlg = SettingsDialog(settings, run_async=runner)
    try:
        dlg.connection_tab._buttons["lmstudio"].click()
        assert any("scheduled" in c for c, _ in captured["calls"])
        assert "all good" in dlg.connection_tab._labels["lmstudio"].text()
        assert "✓" in dlg.connection_tab._labels["lmstudio"].text()
    finally:
        dlg.close()


def test_connection_tab_displays_failure_marker(qapp):
    settings = load_settings(bootstrap=False)

    def runner(coro, on_done):
        from app.services.connection_tester import ProbeResult

        coro.close()
        on_done(ProbeResult(False, "auth failed"))

    dlg = SettingsDialog(settings, run_async=runner)
    try:
        dlg.connection_tab._buttons["openai"].click()
        text = dlg.connection_tab._labels["openai"].text()
        assert "✗" in text
        assert "auth failed" in text
    finally:
        dlg.close()
