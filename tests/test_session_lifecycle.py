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
    # Tests don't have an LM Studio reachable on the LAN — short-circuit the
    # post-action and post-apply probes so they don't burn 3s on httpx
    # default timeouts.
    async def _fake_probe():
        return sm.health_monitor.snapshot

    sm.health_monitor.probe_once = _fake_probe  # type: ignore[assignment]
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


# ---------- apply_settings -------------------------------------------------


def test_apply_settings_swaps_providers(session):
    """Changing transcription/llm providers rebuilds the manager's internals."""
    new_settings = load_settings(
        bootstrap=False,
        overrides={
            "transcription": {"provider": "openai_audio"},
            "llm": {"provider": "openai"},
        },
    )

    asyncio.run(session.apply_settings(new_settings))

    assert session.transcription_provider.provider_key == "openai_audio"
    assert session.llm_provider.provider_key == "openai"
    # SessionManager kept the same recorder instance — only its settings changed.
    assert session.recorder.settings is new_settings.audio


def test_apply_settings_refuses_while_recording(session):
    new_settings = load_settings(bootstrap=False)

    async def _do():
        await session.start()
        with pytest.raises(RuntimeError, match="Stop the recording"):
            await session.apply_settings(new_settings)
        await session.stop()

    asyncio.run(_do())


def test_apply_settings_updates_vad_in_place(session):
    """VAD is runtime-mutable; apply_settings should reach into the existing detector."""
    detector_before = session.vad
    new_settings = load_settings(
        bootstrap=False,
        overrides={"audio": {"vad_enabled": False, "vad_threshold": 0.5}},
    )

    asyncio.run(session.apply_settings(new_settings))

    # Same detector instance — runtime knobs flipped, not rebuilt.
    assert session.vad is detector_before
    assert session.vad.enabled is False
    assert session.vad.threshold == pytest.approx(0.5)


def test_apply_settings_preserves_external_listeners(session):
    """Listeners added on the manager survive the rebuild."""
    captured: list = []
    session.add_warning_listener(captured.append)

    new_settings = load_settings(
        bootstrap=False, overrides={"transcription": {"provider": "openai_audio"}}
    )
    asyncio.run(session.apply_settings(new_settings))

    # Trigger a warning through the new pipeline's listener path.
    session._on_warning("hello after reload")
    assert captured == ["hello after reload"]


def test_apply_settings_picks_up_new_sample_rate(session):
    new_settings = load_settings(
        bootstrap=False, overrides={"transcription": {"sample_rate": 22050}}
    )
    asyncio.run(session.apply_settings(new_settings))
    assert session.recorder.sample_rate == 22050


# ---------- multi-server ---------------------------------------------------


def test_select_server_switches_active_index(session):
    """Calling select_server applies the new index and rebuilds the provider."""
    new_settings = load_settings(
        bootstrap=False,
        overrides={
            "llm_servers": [
                {"name": "A", "base_url": "http://10.0.0.1/v1", "provider": "lmstudio"},
                {"name": "B", "base_url": "http://10.0.0.2/v1", "provider": "lmstudio"},
            ],
        },
    )
    asyncio.run(session.apply_settings(new_settings))
    assert session.llm_provider.settings.base_url == "http://10.0.0.1/v1"

    asyncio.run(session.select_server(1))

    assert session.settings.llm.active_server == 1
    assert session.llm_provider.settings.base_url == "http://10.0.0.2/v1"


def test_select_server_rejects_out_of_range(session):
    new_settings = load_settings(
        bootstrap=False,
        overrides={
            "llm_servers": [
                {"name": "A", "base_url": "http://10.0.0.1/v1", "provider": "lmstudio"},
            ],
        },
    )
    asyncio.run(session.apply_settings(new_settings))
    with pytest.raises(IndexError):
        asyncio.run(session.select_server(5))
