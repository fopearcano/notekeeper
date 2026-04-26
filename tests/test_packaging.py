"""Packaging metadata + entry-point + bundled assets."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------- pyproject.toml -------------------------------------------------


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_console_script_points_to_main(pyproject):
    scripts = pyproject["project"]["scripts"]
    assert scripts.get("notekeeper") == "app.main:main"


def test_pyproject_version_matches_app_dunder(pyproject):
    from app import __version__

    assert pyproject["project"]["version"] == __version__


def test_required_extras_are_declared(pyproject):
    extras = pyproject["project"]["optional-dependencies"]
    for key in ("dev", "faster-whisper", "packaging", "full"):
        assert key in extras, f"missing extra: {key}"


def test_required_runtime_dependencies_are_pinned(pyproject):
    deps = " ".join(pyproject["project"]["dependencies"])
    for needed in ("PySide6", "pydantic", "httpx", "sounddevice", "numpy"):
        assert needed in deps, f"runtime dep missing: {needed}"


def test_package_data_bundles_default_config_and_icon(pyproject):
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert "default_config.toml" in package_data["app.config"]
    # The asset glob covers SVG + PNG so PyInstaller and pip both pick them up.
    assert any(p.startswith("*.svg") or p.endswith(".svg") for p in package_data["app.assets"])


def test_classifiers_mention_each_supported_os(pyproject):
    classifiers = pyproject["project"]["classifiers"]
    text = " ".join(classifiers)
    assert "Linux" in text
    assert "MacOS" in text
    assert "Windows" in text


# ---------- bundled assets / templates -------------------------------------


def test_default_config_ships_in_package():
    bundled = REPO_ROOT / "app" / "config" / "default_config.toml"
    assert bundled.exists()
    # Sanity: parses as TOML and has the sections the loader expects.
    with bundled.open("rb") as fh:
        data = tomllib.load(fh)
    for section in ("app", "audio", "transcription", "llm", "lmstudio", "storage"):
        assert section in data


def test_app_icon_present_and_valid_svg():
    from app.assets import APP_ICON

    assert APP_ICON.exists()
    head = APP_ICON.read_text(encoding="utf-8").lstrip()
    assert head.startswith("<?xml") or head.startswith("<svg")


def test_env_example_lives_at_repo_root():
    target = REPO_ROOT / ".env.example"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in text
    assert "ANTHROPIC_API_KEY" in text


def test_config_example_lives_at_repo_root():
    target = REPO_ROOT / "config.example.toml"
    assert target.exists()
    # It's a valid TOML file the loader can ingest with no overrides.
    with target.open("rb") as fh:
        tomllib.load(fh)


def test_pyinstaller_spec_lives_at_repo_root():
    target = REPO_ROOT / "notekeeper.spec"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "Analysis(" in text
    assert "default_config.toml" in text  # bundled data file


# ---------- entry-point script ---------------------------------------------


def test_app_main_exposes_main_callable():
    from app.main import main

    assert callable(main)


def test_module_form_entry_point_is_executable(monkeypatch, tmp_path):
    """``python -m app.main`` boots the app and exits cleanly when the Qt
    event loop is short-circuited.

    We don't shell out — that would require a display backend even with
    ``offscreen`` and pull a multi-second start cost into the unit suite.
    Instead we patch ``QApplication.exec`` to schedule a quit, then call
    ``main()`` directly.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Redirect persistent state into a temp dir so the test never touches
    # ``~/.notekeeper`` on the developer's machine.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    from PySide6.QtWidgets import QApplication, QMainWindow

    # Replace the event loop with a no-op that returns 0. ``QApplication.exec``
    # is a bound method without parameters in PySide6; the simplest reliable
    # patch is to swap it for a function that just returns the success code.
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)

    # Tear the window down inside ``main()`` so the AsyncWorker QThread isn't
    # left running after this test returns. Patch ``MainWindow.show`` to
    # also schedule a shutdown — by the time exec() (the no-op) returns, the
    # window is already disposable.
    from app.ui.main_window import MainWindow as _MW

    captured: dict = {"window": None}
    real_show = _MW.show

    def _show_and_capture(self):
        captured["window"] = self
        return real_show(self)

    monkeypatch.setattr(_MW, "show", _show_and_capture)

    from app.main import main as app_main

    rc = app_main([sys.argv[0]])
    assert rc == 0
    if captured["window"] is not None:
        captured["window"].shutdown()


# ---------- version surfaced through the UI --------------------------------


def test_app_version_is_a_pep440_string():
    from app import __version__

    assert isinstance(__version__, str)
    # Loose PEP 440: a series of digit groups separated by dots.
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(p[0].isdigit() for p in parts)


def test_main_window_title_includes_version():
    """The window title carries the running version for screenshot diagnosis."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from app import __version__
    from app.config.settings import load_settings
    from app.notes.database import Database
    from app.notes.repository import NoteRepository
    from app.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    settings = load_settings(bootstrap=False)
    db = Database(":memory:")
    try:
        window = MainWindow(settings, NoteRepository(db))
        try:
            assert __version__ in window.windowTitle()
            assert settings.ui.window_title in window.windowTitle()
        finally:
            window.shutdown()
            window.close()
    finally:
        db.close()
        # Defensive: keep the app instance alive for sibling tests.
        del app
