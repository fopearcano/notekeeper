"""All packages should import cleanly without optional ML dependencies."""

import importlib

import pytest

PACKAGES = [
    "app",
    "app.audio",
    "app.audio.audio_buffer",
    "app.audio.recorder",
    "app.audio.vad",
    "app.config",
    "app.config.settings",
    "app.llm",
    "app.llm.base",
    "app.llm.factory",
    "app.llm.lmstudio_provider",
    "app.llm.openai_provider",
    "app.llm.anthropic_provider",
    "app.llm.prompt_templates",
    "app.notes",
    "app.notes.database",
    "app.notes.models",
    "app.notes.repository",
    "app.services",
    "app.services.note_processor",
    "app.services.session_manager",
    "app.services.transcript_pipeline",
    "app.transcription",
    "app.transcription.base",
    "app.transcription.factory",
    "app.transcription.faster_whisper_provider",
    "app.transcription.openai_audio_provider",
    "app.transcription.lmstudio_audio_provider_stub",
    "app.utils.logging",
]


@pytest.mark.parametrize("module_name", PACKAGES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
