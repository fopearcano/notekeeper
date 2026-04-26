from app.config.settings import AppSettings, load_settings


def test_load_settings_returns_app_settings():
    settings = load_settings()
    assert isinstance(settings, AppSettings)
    assert settings.app.name == "Notekeeper"
    assert settings.audio.sample_rate >= 8000
    assert settings.transcription.provider in {"faster_whisper", "openai", "lmstudio_stub"}
    assert settings.llm.provider in {"lmstudio", "openai", "anthropic"}


def test_overrides_take_precedence():
    settings = load_settings(overrides={"ui": {"window_width": 1920}})
    assert settings.ui.window_width == 1920


def test_log_level_normalized():
    settings = load_settings(overrides={"app": {"log_level": "debug"}})
    assert settings.app.log_level == "DEBUG"
