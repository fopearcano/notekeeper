"""Exponential-backoff retry for transient failures.

Scoped to the small set of conditions that are actually retryable on a
LAN: connection refused, read timeouts, transport-level disconnects, and
5xx responses from servers that briefly hiccup. Non-transient outcomes —
401, 403, 404 — bubble up immediately so we don't paper over a
misconfiguration.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

import httpx

log = logging.getLogger(__name__)


T = TypeVar("T")

#: Default classifier shared by the LLM provider's complete + stream-open paths.
TRANSIENT_HTTPX_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
)


def is_transient_status(status: int) -> bool:
    """5xx (except 501) is treated as transient.

    501 *Not Implemented* is configuration, not a hiccup, so it's left to
    the caller. We deliberately don't include 408 (Request Timeout) on the
    server side because real LM Studio servers don't emit it.
    """
    return 500 <= status < 600 and status != 501


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 8.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.15,
    transient_exceptions: tuple[type[BaseException], ...] = TRANSIENT_HTTPX_EXCEPTIONS,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call ``fn`` with exponential-backoff retry on transient errors.

    Sleeps ``initial_delay × backoff_factor**(attempt-1)`` capped at
    ``max_delay``, with up to ``±jitter`` proportional jitter applied to
    each delay so multiple clients reconnecting after the same outage
    don't synchronize. ``max_attempts`` includes the first attempt.

    ``on_retry(attempt, exc, sleep_s)`` lets callers thread retry events
    onto a UI status channel without coupling this helper to Qt.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if initial_delay < 0 or max_delay < 0:
        raise ValueError("delays must be non-negative")

    delay = initial_delay
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except transient_exceptions as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            sleep_s = _jittered(delay, jitter)
            log.warning(
                "Retry %d/%d after %s: %s; sleeping %.2fs",
                attempt, max_attempts, type(exc).__name__, exc, sleep_s,
            )
            if on_retry is not None:
                try:
                    on_retry(attempt, exc, sleep_s)
                except Exception:
                    log.exception("on_retry callback raised")
            await asyncio.sleep(sleep_s)
            delay = min(delay * backoff_factor, max_delay)

    assert last_exc is not None
    raise last_exc


def _jittered(base: float, jitter: float) -> float:
    if base <= 0 or jitter <= 0:
        return max(0.0, base)
    spread = base * jitter
    return max(0.0, base + random.uniform(-spread, spread))
