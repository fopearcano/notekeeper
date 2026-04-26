"""LLM provider factory."""

from __future__ import annotations

from app.config.settings import AppSettings
from app.llm.base import LLMProvider


def create_llm_provider(settings: AppSettings) -> LLMProvider:
    """Construct an LLM provider for ``settings.llm.provider``.

    When the user has configured ``[[llm_servers]]`` and the active server
    targets LM Studio, that entry overrides the legacy ``[lmstudio]``
    section so a toolbar/server switch survives a provider rebuild without
    re-editing the base config.
    """
    name = settings.llm.provider

    if name == "lmstudio":
        from app.llm.lmstudio_provider import LMStudioProvider

        # ``active_lmstudio_settings`` returns the legacy [lmstudio] block
        # when no [[llm_servers]] are defined or the active index is OOB.
        return LMStudioProvider(settings.active_lmstudio_settings())

    if name == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings.openai)

    if name == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings.anthropic)

    raise ValueError(f"Unknown LLM provider: {name!r}")
