from pathlib import Path

import pytest

from app.config.settings import (
    AppSettings,
    DEFAULT_CONFIG_PATH,
    default_user_config_path,
    ensure_user_config,
    load_settings,
)


def test_load_settings_returns_app_settings():
    settings = load_settings(bootstrap=False)
    assert isinstance(settings, AppSettings)
    assert settings.app.name == "Notekeeper"
    assert settings.transcription.provider == "faster_whisper"
    assert settings.transcription.language == "auto"
    assert settings.transcription.chunk_seconds == 5
    assert settings.transcription.sample_rate == 16000


def test_default_provider_subsections():
    settings = load_settings(bootstrap=False)
    assert settings.faster_whisper.model == "small"
    assert settings.faster_whisper.device == "cuda"
    assert settings.faster_whisper.compute_type == "float16"
    assert settings.openai_audio.api_key_env == "OPENAI_API_KEY"
    assert settings.openai_audio.model == "whisper-1"
    assert settings.lmstudio_audio.enabled is False
    assert settings.llm.provider == "lmstudio"
    assert settings.llm.default_task == "clean"
    assert settings.lmstudio.api_key == "lm-studio"
    assert settings.lmstudio.timeout_seconds == 120
    assert settings.openai.model == "gpt-4.1-mini"
    assert settings.anthropic.model == "claude-sonnet-4-5"


def test_overrides_take_precedence():
    settings = load_settings(
        bootstrap=False, overrides={"ui": {"window_width": 1920}}
    )
    assert settings.ui.window_width == 1920


def test_log_level_normalized():
    settings = load_settings(
        bootstrap=False, overrides={"app": {"log_level": "debug"}}
    )
    assert settings.app.log_level == "DEBUG"


def test_default_user_config_path_under_dot_notekeeper(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    # ``Path.home()`` consults $HOME on POSIX; verify the canonical location.
    assert default_user_config_path() == tmp_path / ".notekeeper" / "config.toml"


def test_first_launch_bootstraps_user_config(tmp_path):
    target = tmp_path / "myconfig.toml"
    assert not target.exists()
    ensure_user_config(target)
    assert target.exists()
    # Bootstrapped content must equal the bundled defaults.
    assert target.read_bytes() == DEFAULT_CONFIG_PATH.read_bytes()


def test_load_settings_with_user_config_layered(tmp_path):
    user = tmp_path / "config.toml"
    user.write_text(
        "[llm]\nprovider = \"openai\"\n[openai]\nmodel = \"gpt-4o-mini\"\n",
        encoding="utf-8",
    )
    settings = load_settings(user_config=user, bootstrap=False)
    # User overrides applied
    assert settings.llm.provider == "openai"
    assert settings.openai.model == "gpt-4o-mini"
    # Untouched defaults preserved
    assert settings.lmstudio.api_key == "lm-studio"


def test_invalid_provider_rejected():
    with pytest.raises(Exception):
        load_settings(
            bootstrap=False,
            overrides={"transcription": {"provider": "bogus"}},
        )


def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xyz")
    settings = load_settings(bootstrap=False)
    assert settings.openai.resolve_api_key() == "sk-xyz"
    assert settings.openai_audio.resolve_api_key() == "sk-xyz"


def test_resolve_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = load_settings(bootstrap=False)
    assert settings.openai.resolve_api_key() == ""
