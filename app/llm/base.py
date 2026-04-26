"""LLM provider interface.

Providers expose three operations:

* :meth:`LLMProvider.complete` — single non-streaming completion.
* :meth:`LLMProvider.stream_complete` — async iterator of text deltas. The
  default implementation buffers ``complete`` so subclasses only override it
  if they have a real server-sent-events path.
* :meth:`LLMProvider.list_models` — connection / inventory check used by the
  *Test LM Studio Connection* button.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


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

    async def stream_complete(self, *, system: str, user: str) -> AsyncIterator[str]:
        """Yield text deltas as they become available.

        Default fallback: produce the full :meth:`complete` body as one delta.
        Subclasses with a real streaming endpoint should override.
        """
        response = await self.complete(system=system, user=user)
        if response.text:
            yield response.text

    async def list_models(self) -> list[str]:
        """Return the list of available model ids (best-effort).

        Providers that don't expose a model index should return ``[]``. The
        UI uses this for the *Test Connection* check; failures bubble up so
        the dialog can show a clear error.
        """
        return []

    async def aclose(self) -> None:
        """Override to release any held resources (httpx clients, etc.)."""
