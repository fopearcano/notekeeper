"""Graceful-shutdown contracts for SessionManager + MainWindow."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config.settings import load_settings  # noqa: E402
from app.notes.database import Database  # noqa: E402
from app.notes.repository import NoteRepository  # noqa: E402
from app.services.health_monitor import LLMServerStatus  # noqa: E402
from app.services.session_manager import SessionManager, SessionState  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402


# --------------------------------------------------------------------------- #
# Recorder fixtures                                                            #
# --------------------------------------------------------------------------- #


class _FakeStream:
    """Records ``stop`` / ``close`` calls so tests can assert tear-down."""

    instances: list["_FakeStream"] = []

    def __init__(self, *, samplerate, channels, dtype, device, callback):
        self.started = False
        self.stopped = False
        self.closed = False
        type(self).instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _patch_stream(monkeypatch) -> None:
    import sounddevice as sd

    _FakeStream.instances = []
    monkeypatch.setattr(sd, "InputStream", _FakeStream)


def _quiet_health(sm: SessionManager) -> None:
    """Stub the health probe so aclose doesn't try to reach the LAN."""

    async def _noop():
        return sm.health_monitor.snapshot

    sm.health_monitor.probe_once = _noop  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# SessionManager.aclose                                                        #
# --------------------------------------------------------------------------- #


def _build_session(monkeypatch) -> SessionManager:
    _patch_stream(monkeypatch)
    settings = load_settings(bootstrap=False)
    db = Database(":memory:")
    sm = SessionManager(settings, NoteRepository(db))
    _quiet_health(sm)
    return sm


def test_aclose_stops_recording_in_progress(monkeypatch):
    sm = _build_session(monkeypatch)

    async def _do():
        await sm.start()
        assert sm.state == SessionState.RECORDING
        await sm.aclose()

    asyncio.run(_do())

    assert sm.state == SessionState.STOPPED
    # The fake stream's stop() and close() both fired.
    assert _FakeStream.instances, "no stream was opened"
    assert _FakeStream.instances[-1].stopped is True
    assert _FakeStream.instances[-1].closed is True


def test_aclose_is_idempotent(monkeypatch):
    sm = _build_session(monkeypatch)

    async def _do():
        await sm.aclose()
        await sm.aclose()  # second call must be a no-op
        await sm.aclose()

    asyncio.run(_do())  # no exception, no extra side effects


def test_aclose_stops_health_monitor(monkeypatch):
    sm = _build_session(monkeypatch)

    async def _do():
        # Start the periodic monitor with a tiny interval so the test
        # observes whether ``aclose`` actually halts the loop.
        sm.health_monitor.interval_s = 0.05
        sm.health_monitor.start()
        # Hand the loop one tick so the task gets a chance to run.
        await asyncio.sleep(0.05)
        await sm.aclose()
        # The monitor's task is now either done or cancelled.
        assert sm.health_monitor._task is None or sm.health_monitor._task.done()

    asyncio.run(_do())


def test_aclose_survives_failures_in_underlying_components(monkeypatch):
    """A failing close on one piece must not prevent the others from running."""
    sm = _build_session(monkeypatch)

    closed: dict[str, bool] = {"llm": False}

    async def _boom_stop():
        raise RuntimeError("simulated health-monitor failure")

    async def _track_aclose():
        closed["llm"] = True

    sm.health_monitor.stop = _boom_stop  # type: ignore[assignment]
    sm.note_processor.aclose = _track_aclose  # type: ignore[assignment]

    asyncio.run(sm.aclose())

    # The LLM provider's close still ran even though the monitor's threw.
    assert closed["llm"] is True


# --------------------------------------------------------------------------- #
# MainWindow.shutdown                                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication(sys.argv)


def test_main_window_shutdown_is_idempotent(qapp, monkeypatch):
    _patch_stream(monkeypatch)
    settings = load_settings(bootstrap=False)
    db = Database(":memory:")
    repo = NoteRepository(db)
    try:
        window = MainWindow(settings, repo)
        try:
            # Force the worker loop to settle so ``_session`` exists.
            window._worker_thread.wait(50)
            window.shutdown()
            window.shutdown()  # second call must be a no-op
            assert getattr(window, "_shutdown_complete", False) is True
        finally:
            window.close()
    finally:
        db.close()


def test_main_window_shutdown_stops_timers(qapp, monkeypatch):
    _patch_stream(monkeypatch)
    settings = load_settings(bootstrap=False)
    db = Database(":memory:")
    repo = NoteRepository(db)
    try:
        window = MainWindow(settings, repo)
        try:
            window._autosave_timer.start()
            window._elapsed_timer.start()
            window.shutdown()
            assert not window._autosave_timer.isActive()
            assert not window._elapsed_timer.isActive()
        finally:
            window.close()
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Status transitions during requests (smoke)                                   #
# --------------------------------------------------------------------------- #


def test_health_status_override_propagates(monkeypatch):
    """``set_status_override`` reaches subscribed listeners."""
    sm = _build_session(monkeypatch)
    received: list[LLMServerStatus] = []
    sm.add_health_listener(lambda snap: received.append(snap.status))

    sm.health_monitor.set_status_override(LLMServerStatus.GENERATING)
    sm.health_monitor.set_status_override(
        LLMServerStatus.ERROR, error="upstream broke"
    )

    asyncio.run(sm.aclose())

    # The replay-on-attach plus two overrides → at least three signals.
    assert LLMServerStatus.GENERATING in received
    assert LLMServerStatus.ERROR in received
