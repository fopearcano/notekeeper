"""LLM provider interface.

A provider takes a system prompt + user prompt and returns text. This is the
narrowest possible contract; streaming and tool-use can be added later
without breaking callers.

Providers are constructed via :func:`app.llm.factory.create_llm_provider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: dict | None = None


class LLMProvider(ABC):
    #: Short identifier used for status display (e.g. ``"lmstudio"``).
    provider_key: str = "unknown"

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def complete(self, *, system: str, user: str) -> LLMResponse:
        """Run a single completion request."""

    async def aclose(self) -> None:
        """Override to release any held resources (httpx clients, etc.)."""
