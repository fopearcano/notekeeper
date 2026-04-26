"""TOML writer + ``save_settings`` round-trip."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from app.config.io import dumps_toml, save_settings, settings_to_toml_dict
from app.config.settings import AppSettings, load_settings


# ---------- dumps_toml -----------------------------------------------------


def test_dumps_toml_emits_section_headers_and_keys():
    data = {
        "section_a": {"name": "Hello", "count": 3, "ratio": 0.5, "flag": True},
        "section_b": {"items": ["one", "two", "three"]},
    }
    text = dumps_toml(data)
    assert "[section_a]" in text
    assert "[section_b]" in text
    assert 'name = "Hello"' in text
    assert "count = 3" in text
    assert "flag = true" in text
    assert 'items = ["one", "two", "three"]' in text


def test_dumps_toml_skips_none_values_with_a_comment():
    text = dumps_toml({"audio": {"input_device": None, "channels": 1}})
    assert "channels = 1" in text
    # We emit a placeholder comment so users see the key when they edit by hand.
    assert "# input_device = null" in text


def test_dumps_toml_round_trips_through_tomllib():
    data = {
        "a": {"s": 'with "quotes" and \\ slashes', "i": 42, "f": 3.14, "b": False},
        "b": {"empty": [], "ints": [1, 2, 3]},
    }
    parsed = tomllib.loads(dumps_toml(data))
    assert parsed == data


def test_dumps_toml_escapes_control_chars():
    text = dumps_toml({"x": {"s": "tab\tnew\nline"}})
    parsed = tomllib.loads(text)
    assert parsed == {"x": {"s": "tab\tnew\nline"}}


def test_dumps_toml_rejects_top_level_scalar():
    with pytest.raises(TypeError):
        dumps_toml({"x": "not a section"})


# ---------- settings_to_toml_dict ------------------------------------------


def test_settings_to_toml_dict_contains_every_section():
    settings = load_settings(bootstrap=False)
    data = settings_to_toml_dict(settings)
    expected = {
        "app",
        "ui",
        "audio",
        "transcription",
        "faster_whisper",
        "openai_audio",
        "lmstudio_audio",
        "llm",
        "lmstudio",
        "openai",
        "anthropic",
        "storage",
    }
    assert expected.issubset(data.keys())


# ---------- save_settings --------------------------------------------------


def test_save_settings_round_trip(tmp_path: Path):
    """Saved settings reload as the same AppSettings."""
    settings = load_settings(
        bootstrap=False,
        overrides={
            "ui": {"window_width": 1600},
            "llm": {"provider": "openai"},
            "openai": {"model": "gpt-4o-mini"},
        },
    )
    path = tmp_path / "config.toml"
    save_settings(settings, path=path)

    assert path.exists()
    reloaded = load_settings(user_config=path, bootstrap=False)

    assert reloaded.ui.window_width == 1600
    assert reloaded.llm.provider == "openai"
    assert reloaded.openai.model == "gpt-4o-mini"
    # Untouched fields retain their original defaults.
    assert reloaded.audio.vad_enabled is True


def test_save_settings_is_atomic(tmp_path: Path, monkeypatch):
    """A failed write must leave the prior file intact."""
    path = tmp_path / "config.toml"
    settings = load_settings(bootstrap=False)
    save_settings(settings, path=path)
    original = path.read_bytes()

    # Force an error during the move by monkeypatching Path.replace.
    real_replace = Path.replace

    def _boom(self, target):  # noqa: ARG001
        raise OSError("simulated disk failure")

    monkeypatch.setattr(Path, "replace", _boom)
    with pytest.raises(OSError):
        save_settings(settings, path=path)
    monkeypatch.setattr(Path, "replace", real_replace)

    # Original file is untouched and no leftover .tmp turdles in the dir.
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_save_settings_creates_parent_directories(tmp_path: Path):
    target = tmp_path / "nested" / "dir" / "config.toml"
    save_settings(load_settings(bootstrap=False), path=target)
    assert target.exists()


def test_save_settings_input_device_none_is_commented_out(tmp_path: Path):
    settings = load_settings(bootstrap=False)
    settings.audio.input_device = None
    path = tmp_path / "config.toml"
    save_settings(settings, path=path)
    assert "# input_device = null" in path.read_text(encoding="utf-8")


def test_save_settings_input_device_int_round_trips(tmp_path: Path):
    settings = load_settings(bootstrap=False)
    settings.audio.input_device = 7
    path = tmp_path / "config.toml"
    save_settings(settings, path=path)
    reloaded = load_settings(user_config=path, bootstrap=False)
    assert reloaded.audio.input_device == 7


def test_save_settings_does_not_serialize_api_key_value(tmp_path: Path, monkeypatch):
    """The TOML output never contains the resolved API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-do-not-leak")
    settings = load_settings(bootstrap=False)
    path = tmp_path / "config.toml"
    save_settings(settings, path=path)
    text = path.read_text(encoding="utf-8")
    assert "sk-secret-do-not-leak" not in text
    # The variable name is fine to persist; that's the user-edited part.
    assert "OPENAI_API_KEY" in text
