"""LM Studio LLM provider — OpenAI-compatible chat/completions endpoint.

Three operations supported against ``base_url``:

* ``POST /chat/completions`` for both buffered and streamed completions.
  Streaming uses the OpenAI server-sent-events shape (``data: {…}\n\n``,
  terminated by ``data: [DONE]``).
* ``GET /models`` for the *Test Connection* check.

LM Studio accepts any non-empty ``api_key`` value — we still send it because
some versions reject anonymous requests outright.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from app.config.settings import LMStudioLLMSettings
from app.llm.base import LLMProvider, LLMResponse
from app.utils.logging import get_logger

log = get_logger(__name__)


class LMStudioProvider(LLMProvider):
    provider_key = "lmstudio"

    #: Allows tests to inject ``httpx.MockTransport`` without touching globals.
    _transport: Optional[Any] = None

    def __init__(self, settings: LMStudioLLMSettings):
        self.settings = settings
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            kwargs: dict[str, Any] = {
                "base_url": self.settings.base_url,
                "timeout": httpx.Timeout(self.settings.timeout_seconds, connect=10.0),
                "headers": {
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
            }
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    # ----- non-streaming -------------------------------------------------

    async def complete(self, *, system: str, user: str) -> LLMResponse:
        client = self._get_client()
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
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

    # ----- streaming -----------------------------------------------------

    async def stream_complete(
        self, *, system: str, user: str
    ) -> AsyncIterator[str]:
        client = self._get_client()
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
        }
        log.debug(
            "LM Studio stream request → %s/chat/completions",
            self.settings.base_url,
        )
        async with client.stream(
            "POST", "/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                delta = _parse_sse_delta(line)
                if delta is None:
                    continue
                if delta == _SENTINEL_DONE:
                    break
                if delta:
                    yield delta

    # ----- model index ---------------------------------------------------

    async def list_models(self) -> list[str]:
        client = self._get_client()
        resp = await client.get("/models")
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        out: list[str] = []
        for entry in models:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                out.append(entry["id"])
        return out

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# --------------------------------------------------------------------------- #
# SSE helpers                                                                 #
# --------------------------------------------------------------------------- #


_SENTINEL_DONE = object()


def _parse_sse_delta(line: str) -> Any:
    """Return the next delta string, ``_SENTINEL_DONE``, or ``None`` to skip.

    The OpenAI streaming shape sends ``data: {json}`` per event, with
    ``data: [DONE]`` to terminate the stream. We only forward the
    ``choices[0].delta.content`` field.
    """
    if not line:
        return None
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload:
        return None
    if payload == "[DONE]":
        return _SENTINEL_DONE
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None
    try:
        return chunk["choices"][0]["delta"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return None
