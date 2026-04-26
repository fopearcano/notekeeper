"""Exponential-backoff retry helper."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.services.retry import (
    TRANSIENT_HTTPX_EXCEPTIONS,
    is_transient_status,
    with_retry,
)


def _patch_sleep(monkeypatch) -> list[float]:
    """Replace ``asyncio.sleep`` with a no-op that records each delay."""
    calls: list[float] = []

    async def _fast_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr("app.services.retry.asyncio.sleep", _fast_sleep)
    return calls


def test_succeeds_first_try(monkeypatch):
    sleeps = _patch_sleep(monkeypatch)
    calls = []

    async def fn():
        calls.append(1)
        return "ok"

    result = asyncio.run(with_retry(fn))
    assert result == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_retries_transient_exceptions_then_succeeds(monkeypatch):
    sleeps = _patch_sleep(monkeypatch)
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("refused", request=None)
        return "ok"

    result = asyncio.run(
        with_retry(fn, max_attempts=4, initial_delay=0.5, backoff_factor=2.0, jitter=0.0)
    )
    assert result == "ok"
    assert attempts["n"] == 3
    # Two sleeps: 0.5 then 1.0 (jitter disabled).
    assert sleeps == [pytest.approx(0.5), pytest.approx(1.0)]


def test_gives_up_after_max_attempts(monkeypatch):
    _patch_sleep(monkeypatch)
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        raise httpx.TimeoutException("slow", request=None)

    with pytest.raises(httpx.TimeoutException):
        asyncio.run(with_retry(fn, max_attempts=3, jitter=0.0))
    assert attempts["n"] == 3


def test_does_not_retry_non_transient_errors(monkeypatch):
    _patch_sleep(monkeypatch)
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        asyncio.run(with_retry(fn))
    assert attempts["n"] == 1


def test_caps_delay_at_max(monkeypatch):
    sleeps = _patch_sleep(monkeypatch)
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] < 5:
            raise httpx.ReadError("flap", request=None)
        return "ok"

    asyncio.run(
        with_retry(
            fn,
            max_attempts=5,
            initial_delay=1.0,
            backoff_factor=10.0,  # would explode without the cap
            max_delay=2.5,
            jitter=0.0,
        )
    )
    # Delays: 1.0, then capped to 2.5 (× 3 more retries).
    assert sleeps[0] == pytest.approx(1.0)
    for s in sleeps[1:]:
        assert s == pytest.approx(2.5)


def test_jitter_perturbs_delay_within_bounds(monkeypatch):
    sleeps = _patch_sleep(monkeypatch)
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("x", request=None)
        return "ok"

    asyncio.run(
        with_retry(
            fn,
            max_attempts=3,
            initial_delay=1.0,
            backoff_factor=1.0,  # delay stays at 1.0
            jitter=0.2,
        )
    )
    for s in sleeps:
        assert 0.8 <= s <= 1.2


def test_on_retry_callback_fires_per_attempt(monkeypatch):
    _patch_sleep(monkeypatch)
    captured: list[tuple[int, str, float]] = []

    def on_retry(attempt, exc, sleep_s):
        captured.append((attempt, type(exc).__name__, sleep_s))

    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.PoolTimeout("queued", request=None)
        return None

    asyncio.run(with_retry(fn, max_attempts=3, jitter=0.0, on_retry=on_retry))
    assert [c[0] for c in captured] == [1, 2]
    assert all(c[1] == "PoolTimeout" for c in captured)


def test_invalid_arguments_rejected():
    async def fn():  # pragma: no cover - never called
        return None

    with pytest.raises(ValueError):
        asyncio.run(with_retry(fn, max_attempts=0))
    with pytest.raises(ValueError):
        asyncio.run(with_retry(fn, initial_delay=-1))


def test_is_transient_status_classification():
    assert is_transient_status(500) is True
    assert is_transient_status(502) is True
    assert is_transient_status(503) is True
    assert is_transient_status(504) is True
    assert is_transient_status(501) is False  # 'Not Implemented' is config, not flap
    assert is_transient_status(400) is False
    assert is_transient_status(404) is False
    assert is_transient_status(200) is False


def test_transient_exception_set_includes_common_lan_failures():
    expected = {
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
    }
    assert expected.issubset(set(TRANSIENT_HTTPX_EXCEPTIONS))
