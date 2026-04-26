"""SessionManager: pause, resume, clear_session, VAD toggle, device selection."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.config.settings import load_settings
from app.notes.database import Database
from app.notes.repository import NoteRepository
from app.services.session_manager import SessionManager, SessionState


class _FakeStream:
    """No-op InputStream so .start()/.pause()/.resume() don't open a real device."""

    def __init__(self, *, samplerate, channels, dtype, device, callback):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.callback = callback

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def session(monkeypatch):
    """SessionManager backed by an in-memory SQLite repo and a fake mic stream."""
    import sounddevice as sd

    monkeypatch.setattr(sd, "InputStream", _FakeStream)
    settings = load_settings(bootstrap=False)
    db = Database(":memory:")
    repo = NoteRepository(db)
    sm = SessionManager(settings, repo)
    yield sm
    db.close()


def _run_lifecycle(session: SessionManager, *steps: str) -> None:
    """Run a sequence of lifecycle methods inside a single event loop.

    The pipeline binds asyncio tasks to the loop in which ``start()`` ran,
    so subsequent ``stop()`` etc. must use the same loop — calling
    ``asyncio.run`` once per step would break that invariant.
    """

    async def _do() -> None:
        for step in steps:
            await getattr(session, step)()

    asyncio.run(_do())


# ---------- state transitions ----------------------------------------------


def test_start_pause_resume_stop_state_transitions(session):
    states: list[SessionState] = []

    async def _do():
        await session.start()
        states.append(session.state)
        await session.pause()
        states.append(session.state)
        await session.resume()
        states.append(session.state)
        await session.stop()
        states.append(session.state)

    asyncio.run(_do())
    assert states == [
        SessionState.RECORDING,
        SessionState.PAUSED,
        SessionState.RECORDING,
        SessionState.STOPPED,
    ]


def test_pause_no_op_when_not_recording(session):
    asyncio.run(session.pause())
    assert session.state == SessionState.IDLE


def test_resume_no_op_when_not_paused(session):
    async def _do():
        await session.start()
        await session.resume()  # already recording → no-op
        await session.stop()

    asyncio.run(_do())
    assert session.state == SessionState.STOPPED


def test_clear_session_resets_state(session):
    from app.transcription.base import TranscriptSegment

    saved_id: dict[str, int] = {}

    async def _do():
        # All lifecycle calls must share one event loop because the pipeline
        # binds asyncio tasks to whichever loop ``start()`` ran on.
        await session.start()
        session._on_segment(TranscriptSegment(text="hi", is_final=True))
        saved = session.save_note(title="WIP")
        saved_id["id"] = saved.id
        assert session.current_note is not None

        await session.clear_session()

    asyncio.run(_do())

    assert session.state == SessionState.IDLE
    assert session.transcript_text() == ""
    assert session.current_note is None
    # The persisted note still exists on disk.
    assert session.repository.get_note(saved_id["id"]).title == "WIP"


def test_clear_session_stops_active_recording(session):
    async def _do():
        await session.start()
        assert session.state == SessionState.RECORDING
        await session.clear_session()

    asyncio.run(_do())
    assert session.state == SessionState.IDLE


# ---------- VAD ------------------------------------------------------------


def test_vad_toggle_proxies_to_detector(session):
    assert session.vad_enabled is True
    session.set_vad_enabled(False)
    assert session.vad_enabled is False
    assert session.vad.enabled is False

    session.set_vad_threshold(0.25)
    assert session.vad_threshold == pytest.approx(0.25)
    assert session.vad.threshold == pytest.approx(0.25)


# ---------- input device --------------------------------------------------


def test_set_input_device_blocked_while_recording(session):
    warnings: list[str] = []
    session.add_warning_listener(warnings.append)

    session.set_input_device(3)
    assert session.settings.audio.input_device == 3

    async def _do():
        await session.start()
        session.set_input_device(7)  # ignored
        await session.stop()

    asyncio.run(_do())

    assert session.settings.audio.input_device == 3
    assert any("cannot be changed while recording" in w for w in warnings)


def test_set_input_device_applies_when_idle(session):
    session.set_input_device("USB Mic")
    assert session.settings.audio.input_device == "USB Mic"
    # The recorder reads from settings.audio so the next start uses the new value.
    assert session.recorder.settings is session.settings.audio
