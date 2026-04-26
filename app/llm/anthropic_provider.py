"""Anthropic Messages API provider."""

from __future__ import annotations

from typing import Optional

import httpx

from app.llm.base import LLMProvider, LLMResponse
from app.utils.logging import get_logger

log = get_logger(__name__)

_ANTHROPIC_BASE = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    def __init__(self, settings):
        super().__init__(settings)
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            if not self.settings.api_key:
                raise RuntimeError(
                    "Anthropic provider requires an API key (set "
                    "ANTHROPIC_API_KEY or configure llm.api_key)."
                )
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url or _ANTHROPIC_BASE,
                timeout=self.settings.request_timeout_s,
                headers={
                    "x-api-key": self.settings.api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def complete(self, *, system: str, user: str) -> LLMResponse:
        client = self._get_client()
        payload = {
            "model": self.settings.model,
            "system": system,
            "max_tokens": self.settings.max_output_tokens,
            "temperature": self.settings.temperature,
            "messages": [{"role": "user", "content": user}],
        }
        resp = await client.post("/messages", json=payload)
        resp.raise_for_status()
        data = resp.json()
        # The Messages API returns a list of content blocks; concatenate text blocks.
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        return LLMResponse(
            text="".join(parts),
            model=data.get("model", self.settings.model),
            usage=data.get("usage"),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
