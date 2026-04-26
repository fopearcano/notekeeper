"""HealthMonitor / LLMServerStatus."""

from __future__ import annotations

import asyncio
from typing import Callable

import httpx
import pytest

from app.config.settings import LMStudioLLMSettings
from app.services.health_monitor import (
    HealthMonitor,
    HealthSnapshot,
    LLMServerStatus,
)


def _settings(**overrides) -> LMStudioLLMSettings:
    base = {
        "base_url": "http://192.168.1.100:1234/v1",
        "api_key": "lm-studio",
        "model": "local-model",
        "timeout_seconds": 30,
    }
    base.update(overrides)
    return LMStudioLLMSettings(**base)


def _monitor(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> HealthMonitor:
    settings = kwargs.pop("settings", _settings())

    def factory():
        return httpx.AsyncClient(
            base_url=settings.base_url,
            transport=httpx.MockTransport(handler),
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )

    return HealthMonitor(settings, client_factory=factory, **kwargs)


# ---------- one-shot probe behaviour ---------------------------------------


def test_probe_online_when_model_in_list():
    def handler(_req):
        return httpx.Response(200, json={"data": [{"id": "local-model"}, {"id": "other"}]})

    monitor = _monitor(handler)
    snapshot = asyncio.run(monitor.probe_once())

    assert snapshot.status == LLMServerStatus.ONLINE
    assert snapshot.model == "local-model"
    assert "local-model" in snapshot.available_models
    assert snapshot.error == ""
    assert snapshot.latency_ms > 0


def test_probe_model_unavailable_when_missing_from_list():
    def handler(_req):
        return httpx.Response(200, json={"data": [{"id": "another-model"}]})

    monitor = _monitor(handler)
    snapshot = asyncio.run(monitor.probe_once())

    assert snapshot.status == LLMServerStatus.MODEL_UNAVAILABLE
    assert "not loaded" in snapshot.error.lower()


def test_probe_offline_on_connection_error():
    def handler(req):
        raise httpx.ConnectError("refused", request=req)

    monitor = _monitor(handler)
    snapshot = asyncio.run(monitor.probe_once())

    assert snapshot.status == LLMServerStatus.OFFLINE
    assert "ConnectError" in snapshot.error


def test_probe_error_on_auth_failure():
    def handler(_req):
        return httpx.Response(401, json={"error": "no auth"})

    monitor = _monitor(handler)
    snapshot = asyncio.run(monitor.probe_once())

    assert snapshot.status == LLMServerStatus.ERROR
    assert "auth" in snapshot.error.lower()


def test_probe_error_on_5xx():
    def handler(_req):
        return httpx.Response(503, text="busy")

    monitor = _monitor(handler)
    snapshot = asyncio.run(monitor.probe_once())

    assert snapshot.status == LLMServerStatus.ERROR
    assert "503" in snapshot.error


def test_probe_error_on_non_json():
    def handler(_req):
        return httpx.Response(200, text="not json")

    monitor = _monitor(handler)
    snapshot = asyncio.run(monitor.probe_once())

    assert snapshot.status == LLMServerStatus.ERROR
    assert "non-JSON" in snapshot.error


def test_probe_online_when_model_field_is_empty():
    """An unconfigured model field shouldn't tank the status — anything goes."""

    def handler(_req):
        return httpx.Response(200, json={"data": [{"id": "anything"}]})

    monitor = _monitor(handler, settings=_settings(model=""))
    snapshot = asyncio.run(monitor.probe_once())

    assert snapshot.status == LLMServerStatus.ONLINE


# ---------- listeners + overrides ------------------------------------------


def test_listener_receives_initial_snapshot_on_attach():
    monitor = _monitor(lambda _r: httpx.Response(200, json={"data": []}))
    received: list[HealthSnapshot] = []
    monitor.add_listener(received.append)
    # First snapshot is UNKNOWN — listeners should get it before any probe.
    assert received[0].status == LLMServerStatus.UNKNOWN


def test_listener_fires_on_probe():
    def handler(_req):
        return httpx.Response(200, json={"data": [{"id": "local-model"}]})

    monitor = _monitor(handler)
    received: list[HealthSnapshot] = []
    monitor.add_listener(received.append)

    asyncio.run(monitor.probe_once())

    # Initial UNKNOWN + probe ONLINE.
    assert [s.status for s in received] == [
        LLMServerStatus.UNKNOWN,
        LLMServerStatus.ONLINE,
    ]


def test_set_status_override_updates_snapshot_and_notifies():
    monitor = _monitor(lambda _r: httpx.Response(200, json={"data": []}))
    received: list[HealthSnapshot] = []
    monitor.add_listener(received.append)

    monitor.set_status_override(LLMServerStatus.GENERATING)

    assert monitor.snapshot.status == LLMServerStatus.GENERATING
    assert any(s.status == LLMServerStatus.GENERATING for s in received)


def test_update_settings_resets_to_unknown_and_notifies():
    monitor = _monitor(lambda _r: httpx.Response(200, json={"data": []}))
    received: list[HealthSnapshot] = []
    monitor.add_listener(received.append)

    monitor.update_settings(_settings(base_url="http://10.0.0.5:1234/v1"))

    assert monitor.snapshot.status == LLMServerStatus.UNKNOWN
    assert monitor.snapshot.base_url == "http://10.0.0.5:1234/v1"
    assert received[-1].base_url == "http://10.0.0.5:1234/v1"


def test_remove_listener_stops_emissions():
    monitor = _monitor(lambda _r: httpx.Response(200, json={"data": []}))
    received: list[HealthSnapshot] = []
    monitor.add_listener(received.append)
    initial = len(received)

    monitor.remove_listener(received.append)
    monitor.set_status_override(LLMServerStatus.GENERATING)

    assert len(received) == initial  # no new emission after removal


# ---------- background loop ------------------------------------------------


def test_run_forever_probes_then_stops_promptly():
    calls = {"n": 0}

    def handler(_req):
        calls["n"] += 1
        return httpx.Response(200, json={"data": [{"id": "local-model"}]})

    monitor = _monitor(handler, interval_s=0.05)

    async def _drive():
        task = monitor.start()
        await asyncio.sleep(0.18)  # roughly 3-4 ticks
        await monitor.stop()
        if not task.done():
            task.cancel()

    asyncio.run(_drive())
    assert calls["n"] >= 2


# ---------- bookkeeping ----------------------------------------------------


def test_status_label_strings():
    assert LLMServerStatus.ONLINE.label == "online"
    assert LLMServerStatus.MODEL_UNAVAILABLE.label == "model unavailable"
    assert LLMServerStatus.GENERATING.label == "generating"
    assert LLMServerStatus.OFFLINE.label == "offline"


def test_health_snapshot_is_frozen():
    snapshot = HealthSnapshot(
        status=LLMServerStatus.ONLINE, base_url="x", model="y"
    )
    with pytest.raises(Exception):
        snapshot.error = "tampered"  # type: ignore[misc]


def test_health_snapshot_is_healthy_only_when_online():
    base = dict(base_url="x", model="y")
    assert HealthSnapshot(status=LLMServerStatus.ONLINE, **base).is_healthy()
    for s in (
        LLMServerStatus.UNKNOWN,
        LLMServerStatus.OFFLINE,
        LLMServerStatus.MODEL_UNAVAILABLE,
        LLMServerStatus.GENERATING,
        LLMServerStatus.ERROR,
    ):
        assert not HealthSnapshot(status=s, **base).is_healthy()
