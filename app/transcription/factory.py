"""Transcription provider factory."""

from __future__ import annotations

from app.config.settings import AppSettings
from app.transcription.base import TranscriptionProvider


def create_transcription_provider(settings: AppSettings) -> TranscriptionProvider:
    """Construct a transcription provider for ``settings.transcription.provider``.

    Heavy provider modules are imported lazily so the LLM-only code paths
    don't pull in optional ML dependencies.
    """
    name = settings.transcription.provider

    if name == "faster_whisper":
        from app.transcription.faster_whisper_provider import FasterWhisperProvider

        return FasterWhisperProvider(settings.transcription, settings.faster_whisper)

    if name == "openai_audio":
        from app.transcription.openai_audio_provider import OpenAIAudioProvider

        return OpenAIAudioProvider(settings.transcription, settings.openai_audio)

    if name == "lmstudio_audio":
        from app.transcription.lmstudio_audio_provider_stub import LMStudioAudioProviderStub

        return LMStudioAudioProviderStub(settings.transcription, settings.lmstudio_audio)

    raise ValueError(f"Unknown transcription provider: {name!r}")
