"""LLM provider factory."""

from __future__ import annotations

from app.config.settings import AppSettings
from app.llm.base import LLMProvider


def create_llm_provider(settings: AppSettings) -> LLMProvider:
    """Construct an LLM provider for ``settings.llm.provider``."""
    name = settings.llm.provider

    if name == "lmstudio":
        from app.llm.lmstudio_provider import LMStudioProvider

        return LMStudioProvider(settings.lmstudio)

    if name == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings.openai)

    if name == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings.anthropic)

    raise ValueError(f"Unknown LLM provider: {name!r}")
