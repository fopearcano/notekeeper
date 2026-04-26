"""OpenAI Chat Completions provider."""

from __future__ import annotations

from typing import Optional

import httpx

from app.llm.base import LLMProvider, LLMResponse
from app.utils.logging import get_logger

log = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(self, settings):
        super().__init__(settings)
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            if not self.settings.api_key:
                raise RuntimeError(
                    "OpenAI provider requires an API key (set OPENAI_API_KEY or "
                    "configure llm.api_key in your config file)."
                )
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url or "https://api.openai.com/v1",
                timeout=self.settings.request_timeout_s,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
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
        return LLMResponse(text=text, model=data.get("model", self.settings.model),
                           usage=data.get("usage"))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
