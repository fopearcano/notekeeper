"""OpenAI Chat Completions provider."""

from __future__ import annotations

from typing import Optional

import httpx

from app.config.settings import OpenAILLMSettings
from app.llm.base import LLMProvider, LLMResponse
from app.utils.logging import get_logger

log = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    provider_key = "openai"

    def __init__(self, settings: OpenAILLMSettings):
        self.settings = settings
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            api_key = self.settings.resolve_api_key()
            if not api_key:
                raise RuntimeError(
                    f"OpenAI provider requires an API key — set the "
                    f"${self.settings.api_key_env} environment variable."
                )
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_seconds,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def complete(self, *, system: str, user: str) -> LLMResponse:
        client = self._get_client()
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_output_tokens,
        }
        resp = await client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(
            text=text,
            model=data.get("model", self.settings.model),
            usage=data.get("usage"),
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
