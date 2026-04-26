"""Provider factories should dispatch on configured provider names."""

import pytest

from app.config.settings import load_settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.factory import create_llm_provider
from app.llm.lmstudio_provider import LMStudioProvider
from app.llm.openai_provider import OpenAIProvider
from app.transcription.factory import create_transcription_provider
from app.transcription.faster_whisper_provider import FasterWhisperProvider
from app.transcription.lmstudio_audio_provider_stub import LMStudioAudioProviderStub
from app.transcription.openai_audio_provider import OpenAIAudioProvider


def _settings(overrides=None):
    return load_settings(bootstrap=False, overrides=overrides)


@pytest.mark.parametrize(
    "name, cls",
    [
        ("faster_whisper", FasterWhisperProvider),
        ("openai_audio", OpenAIAudioProvider),
        ("lmstudio_audio", LMStudioAudioProviderStub),
    ],
)
def test_transcription_factory_dispatch(name, cls):
    provider = create_transcription_provider(
        _settings({"transcription": {"provider": name}})
    )
    assert isinstance(provider, cls)
    assert provider.provider_key == name


@pytest.mark.parametrize(
    "name, cls",
    [
        ("lmstudio", LMStudioProvider),
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
    ],
)
def test_llm_factory_dispatch(name, cls):
    provider = create_llm_provider(_settings({"llm": {"provider": name}}))
    assert isinstance(provider, cls)
    assert provider.provider_key == name


def test_lmstudio_audio_stays_a_stub_when_streamed():
    """The stub must refuse to stream regardless of the ``enabled`` flag."""
    import asyncio

    settings = _settings({"transcription": {"provider": "lmstudio_audio"},
                          "lmstudio_audio": {"enabled": True}})
    provider = create_transcription_provider(settings)

    async def _run():
        await provider.start()

        async def empty_chunks():
            if False:
                yield  # pragma: no cover

        with pytest.raises(NotImplementedError):
            async for _seg in provider.stream(empty_chunks()):
                pass

    asyncio.run(_run())
