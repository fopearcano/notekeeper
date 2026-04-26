"""LM Studio LLM provider — OpenAI-compatible chat/completions endpoint.

LM Studio exposes ``POST {base_url}/chat/completions`` with the standard
OpenAI request shape. The ``api_key`` is typically a placeholder string
(``lm-studio``) but is sent in the ``Authorization`` header anyway because
some LM Studio versions require it.
"""

from __future__ import annotations

from typing import Optional

import httpx

from app.config.settings import LMStudioLLMSettings
from app.llm.base import LLMProvider, LLMResponse
from app.utils.logging import get_logger

log = get_logger(__name__)


class LMStudioProvider(LLMProvider):
    provider_key = "lmstudio"

    def __init__(self, settings: LMStudioLLMSettings):
        self.settings = settings
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=self.settings.timeout_seconds,
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
        }
        log.debug("LM Studio request → %s/chat/completions", self.settings.base_url)
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
