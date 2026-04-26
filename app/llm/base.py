"""LLM provider interface.

A provider takes a system prompt + user prompt and returns text. This is the
narrowest possible contract; streaming and tool-use can be added later
without breaking callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config.settings import LLMSettings


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: dict | None = None


class LLMProvider(ABC):
    def __init__(self, settings: LLMSettings):
        self.settings = settings

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def complete(self, *, system: str, user: str) -> LLMResponse:
        """Run a single completion request."""

    async def aclose(self) -> None:
        """Override to release any held resources (httpx clients, etc.)."""


def build_provider(settings: LLMSettings) -> LLMProvider:
    if settings.provider == "lmstudio":
        from app.llm.lmstudio_provider import LMStudioProvider

        return LMStudioProvider(settings)
    if settings.provider == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    if settings.provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    raise ValueError(f"Unknown LLM provider: {settings.provider!r}")
