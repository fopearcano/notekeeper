"""LM Studio health monitor.

Periodically probes the active LM Studio server's ``/models`` endpoint
and emits :class:`HealthSnapshot` events to subscribed listeners. The
monitor runs as an asyncio task on the existing worker loop so the GUI
thread never blocks on a network call.

Status transitions
------------------

* ``UNKNOWN`` — initial value, before the first probe completes.
* ``OFFLINE`` — last probe could not reach the server (timeout / connect
  refused / DNS).
* ``ERROR`` — last probe completed but returned an unexpected status or
  malformed body.
* ``MODEL_UNAVAILABLE`` — server responded but the configured model id
  isn't in the returned list.
* ``ONLINE`` — server reachable and configured model is loaded.
* ``GENERATING`` — set externally by :class:`SessionManager` while a
  request is in flight; the next probe restores the underlying state.

The ``GENERATING`` state is **never** set by the monitor itself; the
session manager flips to it before each LLM call and back to the
last-known status afterwards. That way a long stream isn't constantly
fighting with a "ONLINE" override on the next periodic probe.
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx

from app.config.settings import LMStudioLLMSettings
from app.utils.logging import get_logger

log = get_logger(__name__)


class LLMServerStatus(enum.Enum):
    UNKNOWN = "unknown"
    OFFLINE = "offline"
    ONLINE = "online"
    MODEL_UNAVAILABLE = "model_unavailable"
    GENERATING = "generating"
    ERROR = "error"

    @property
    def label(self) -> str:
        """Human-friendly label for the UI."""
        return {
            LLMServerStatus.UNKNOWN: "unknown",
            LLMServerStatus.OFFLINE: "offline",
            LLMServerStatus.ONLINE: "online",
            LLMServerStatus.MODEL_UNAVAILABLE: "model unavailable",
            LLMServerStatus.GENERATING: "generating",
            LLMServerStatus.ERROR: "error",
        }[self]


@dataclass(frozen=True)
class HealthSnapshot:
    status: LLMServerStatus
    base_url: str
    model: str
    available_models: tuple[str, ...] = ()
    latency_ms: float = 0.0
    error: str = ""
    checked_at: float = field(default_factory=time.time)

    def is_healthy(self) -> bool:
        return self.status == LLMServerStatus.ONLINE


HealthListener = Callable[[HealthSnapshot], None]


class HealthMonitor:
    """Periodic ``/models`` probe."""

    def __init__(
        self,
        settings: LMStudioLLMSettings,
        *,
        interval_s: float = 30.0,
        timeout_s: float = 5.0,
        client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
    ):
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.settings = settings
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self._client_factory = client_factory or self._default_client
        self._listeners: list[HealthListener] = []
        self._snapshot = HealthSnapshot(
            status=LLMServerStatus.UNKNOWN,
            base_url=settings.base_url,
            model=settings.model,
        )
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def _default_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.settings.base_url,
            timeout=httpx.Timeout(self.timeout_s, connect=min(self.timeout_s, 3.0)),
            headers={"Authorization": f"Bearer {self.settings.api_key}"},
        )

    @property
    def snapshot(self) -> HealthSnapshot:
        return self._snapshot

    def add_listener(self, listener: HealthListener) -> None:
        self._listeners.append(listener)
        # Immediately fire current snapshot so newly-attached UI bits don't
        # have to wait one full ``interval_s`` for the first paint.
        try:
            listener(self._snapshot)
        except Exception:
            log.exception("Health listener raised on attach")

    def remove_listener(self, listener: HealthListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def update_settings(self, settings: LMStudioLLMSettings) -> None:
        """Switch the monitor to a different server in place."""
        self.settings = settings
        self._snapshot = HealthSnapshot(
            status=LLMServerStatus.UNKNOWN,
            base_url=settings.base_url,
            model=settings.model,
        )
        self._notify()

    def set_status_override(
        self, status: LLMServerStatus, *, error: str = ""
    ) -> None:
        """Force a transient status (used for ``GENERATING`` / ``ERROR``).

        The next periodic probe overwrites this with whatever it observes.
        """
        self._snapshot = HealthSnapshot(
            status=status,
            base_url=self.settings.base_url,
            model=self.settings.model,
            available_models=self._snapshot.available_models,
            latency_ms=self._snapshot.latency_ms,
            error=error,
        )
        self._notify()

    async def probe_once(self) -> HealthSnapshot:
        """Run a single probe; update the snapshot and notify listeners."""
        client = self._client_factory()
        t0 = time.monotonic()
        try:
            try:
                resp = await client.get("/models")
            except httpx.HTTPError as exc:
                snapshot = HealthSnapshot(
                    status=LLMServerStatus.OFFLINE,
                    base_url=self.settings.base_url,
                    model=self.settings.model,
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._snapshot = snapshot
                self._notify()
                return snapshot

            latency_ms = (time.monotonic() - t0) * 1000.0
            if resp.status_code in (401, 403):
                snapshot = HealthSnapshot(
                    status=LLMServerStatus.ERROR,
                    base_url=self.settings.base_url,
                    model=self.settings.model,
                    latency_ms=latency_ms,
                    error=f"authentication failed ({resp.status_code})",
                )
                self._snapshot = snapshot
                self._notify()
                return snapshot
            if resp.status_code >= 400:
                snapshot = HealthSnapshot(
                    status=LLMServerStatus.ERROR,
                    base_url=self.settings.base_url,
                    model=self.settings.model,
                    latency_ms=latency_ms,
                    error=f"HTTP {resp.status_code}",
                )
                self._snapshot = snapshot
                self._notify()
                return snapshot

            try:
                payload = resp.json()
            except ValueError:
                snapshot = HealthSnapshot(
                    status=LLMServerStatus.ERROR,
                    base_url=self.settings.base_url,
                    model=self.settings.model,
                    latency_ms=latency_ms,
                    error="non-JSON response from /models",
                )
                self._snapshot = snapshot
                self._notify()
                return snapshot

            raw = payload.get("data") if isinstance(payload, dict) else None
            models: tuple[str, ...] = tuple(
                m["id"]
                for m in (raw or [])
                if isinstance(m, dict) and isinstance(m.get("id"), str)
            )
            wanted = self.settings.model
            if not wanted or wanted in models:
                status = LLMServerStatus.ONLINE
                error = ""
            else:
                status = LLMServerStatus.MODEL_UNAVAILABLE
                error = (
                    f"configured model {wanted!r} not loaded on the server "
                    f"(available: {', '.join(models[:3]) or 'none'})"
                )

            snapshot = HealthSnapshot(
                status=status,
                base_url=self.settings.base_url,
                model=wanted,
                available_models=models,
                latency_ms=latency_ms,
                error=error,
            )
            self._snapshot = snapshot
            self._notify()
            return snapshot
        finally:
            try:
                await client.aclose()
            except Exception:
                log.exception("aclose raised on health-probe client")

    async def run_forever(self) -> None:
        """Probe periodically until :meth:`stop` is called."""
        while not self._stop.is_set():
            await self.probe_once()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
                return
            except asyncio.TimeoutError:
                continue

    def start(self) -> asyncio.Task:
        if self._task is not None and not self._task.done():
            return self._task
        self._stop.clear()
        self._task = asyncio.create_task(self.run_forever())
        return self._task

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

    def _notify(self) -> None:
        for listener in list(self._listeners):
            try:
                listener(self._snapshot)
            except Exception:
                log.exception("Health listener raised")
