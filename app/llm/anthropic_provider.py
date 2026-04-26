"""Anthropic Messages API provider."""

from __future__ import annotations

from typing import Optional

import httpx

from app.config.settings import AnthropicLLMSettings
from app.llm.base import LLMProvider, LLMResponse
from app.utils.logging import get_logger

log = get_logger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    provider_key = "anthropic"

    def __init__(self, settings: AnthropicLLMSettings):
        self.settings = settings
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            api_key = self.settings.resolve_api_key()
            if not api_key:
                raise RuntimeError(
                    f"Anthropic provider requires an API key — set the "
                    f"${self.settings.api_key_env} environment variable."
                )
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_seconds,
                headers={
                    "x-api-key": api_key,
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
