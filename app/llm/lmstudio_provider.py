"""LM Studio provider — talks to its OpenAI-compatible /v1/chat/completions API."""

from __future__ import annotations

from typing import Optional

import httpx

from app.llm.base import LLMProvider, LLMResponse
from app.utils.logging import get_logger

log = get_logger(__name__)


class LMStudioProvider(LLMProvider):
    """Default provider for fully-local development."""

    def __init__(self, settings):
        super().__init__(settings)
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=self.settings.request_timeout_s,
                headers={"Content-Type": "application/json"},
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
        log.debug("LM Studio request → %s/chat/completions", self.settings.base_url)
        resp = await client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, model=data.get("model", self.settings.model),
                           usage=data.get("usage"))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
