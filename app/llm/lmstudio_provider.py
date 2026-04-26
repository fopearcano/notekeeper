"""LM Studio LLM provider — OpenAI-compatible chat/completions endpoint.

Three operations supported against ``base_url``:

* ``POST /chat/completions`` for both buffered and streamed completions.
  Streaming uses the OpenAI server-sent-events shape (``data: {…}\n\n``,
  terminated by ``data: [DONE]``).
* ``GET /models`` for the *Test Connection* check.

LM Studio accepts any non-empty ``api_key`` value — we still send it because
some versions reject anonymous requests outright. The optional API token
flows through this same field; the bearer header is sent unconditionally.

Transient errors (timeouts, refused connections, 5xx) on the buffered
``complete()`` and the *opening* of the streaming POST are retried with
exponential backoff. Mid-stream disconnects are not retried — once token
deltas have been emitted to the UI, replaying would duplicate text.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from app.config.settings import LMStudioLLMSettings
from app.llm.base import LLMProvider, LLMResponse
from app.services.retry import (
    TRANSIENT_HTTPX_EXCEPTIONS,
    is_transient_status,
    with_retry,
)
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

        async def _attempt() -> httpx.Response:
            resp = await client.post("/chat/completions", json=payload)
            if is_transient_status(resp.status_code):
                # Coerce a transient HTTP error into a transient exception so
                # ``with_retry`` reschedules instead of raising upstream.
                raise httpx.RemoteProtocolError(
                    f"transient {resp.status_code} from LM Studio",
                    request=resp.request,
                )
            return resp

        resp = await with_retry(_attempt)
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

        # Retry the *open* of the streaming POST on transient failures, but
        # never replay a stream that has already started yielding deltas —
        # the UI has shown those tokens and a retry would duplicate them.
        async for delta in self._streaming_iterator(client, payload):
            yield delta

    async def _streaming_iterator(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        import asyncio

        max_attempts = 3
        delay = 0.5
        max_delay = 8.0
        backoff = 2.0
        last_exc: BaseException | None = None

        for attempt in range(1, max_attempts + 1):
            cm = client.stream("POST", "/chat/completions", json=payload)
            try:
                resp = await cm.__aenter__()
            except TRANSIENT_HTTPX_EXCEPTIONS as exc:
                last_exc = exc
                if attempt == max_attempts:
                    raise
                log.warning("LM Studio stream open retry %d/%d: %s",
                            attempt, max_attempts, exc)
                await asyncio.sleep(delay)
                delay = min(delay * backoff, max_delay)
                continue

            try:
                if is_transient_status(resp.status_code):
                    last_exc = httpx.RemoteProtocolError(
                        f"transient {resp.status_code} from LM Studio",
                        request=resp.request,
                    )
                    if attempt == max_attempts:
                        resp.raise_for_status()
                else:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        delta = _parse_sse_delta(line)
                        if delta is None:
                            continue
                        if delta == _SENTINEL_DONE:
                            return
                        if delta:
                            yield delta
                    return
            finally:
                await cm.__aexit__(None, None, None)

            # Got a transient status code without the resp.raise above.
            await asyncio.sleep(delay)
            delay = min(delay * backoff, max_delay)

        assert last_exc is not None
        raise last_exc

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
