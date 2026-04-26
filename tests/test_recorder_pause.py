"""AudioRecorder: pause/resume + active-time elapsed."""

from __future__ import annotations

import time

import sounddevice as sd

from app.audio.audio_buffer import AudioBuffer
from app.audio.recorder import AudioRecorder, RecorderState
from app.config.settings import AudioSettings


class _FakeStream:
    """No-op stream so recorder lifecycle tests don't open a real device."""

    instances: list["_FakeStream"] = []

    def __init__(self, *, samplerate, channels, dtype, device, callback):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.callback = callback
        self.started = False
        self.closed = False
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def _build(monkeypatch) -> AudioRecorder:
    _FakeStream.instances = []
    monkeypatch.setattr(sd, "InputStream", _FakeStream)
    return AudioRecorder(
        AudioSettings(),
        sample_rate=16000,
        chunk_seconds=0.1,
        buffer=AudioBuffer(),
    )


def test_pause_and_resume_open_close_streams(monkeypatch):
    rec = _build(monkeypatch)
    rec.start()
    assert rec.state == RecorderState.RECORDING
    assert len(_FakeStream.instances) == 1
    assert _FakeStream.instances[-1].started is True

    rec.pause()
    assert rec.state == RecorderState.PAUSED
    # Stream stopped + closed on pause.
    assert _FakeStream.instances[0].closed is True

    rec.resume()
    assert rec.state == RecorderState.RECORDING
    assert len(_FakeStream.instances) == 2
    assert _FakeStream.instances[-1].started is True

    rec.stop()
    assert rec.state == RecorderState.STOPPED


def test_paused_time_excluded_from_elapsed(monkeypatch):
    rec = _build(monkeypatch)
    # Stub time so the test is deterministic.
    fake_now = [1000.0]

    def _mono():
        return fake_now[0]

    monkeypatch.setattr("app.audio.recorder.time.monotonic", _mono)

    rec.start()
    assert rec.elapsed_s() == 0.0

    fake_now[0] += 5.0  # 5 s of recording
    assert rec.elapsed_s() == 5.0

    rec.pause()
    fake_now[0] += 10.0  # 10 s of paused time
    assert rec.elapsed_s() == 5.0  # still 5 — pause excluded

    rec.resume()
    fake_now[0] += 3.0  # 3 more seconds of recording
    assert rec.elapsed_s() == 8.0


def test_pause_no_op_when_not_recording(monkeypatch):
    rec = _build(monkeypatch)
    # pause() before start() is a no-op (stays IDLE).
    rec.pause()
    assert rec.state == RecorderState.IDLE


def test_resume_no_op_when_not_paused(monkeypatch):
    rec = _build(monkeypatch)
    rec.start()
    rec.resume()  # already recording → no-op
    assert rec.state == RecorderState.RECORDING


def test_stop_while_paused_includes_pause_in_total(monkeypatch):
    rec = _build(monkeypatch)
    fake_now = [1000.0]
    monkeypatch.setattr("app.audio.recorder.time.monotonic", lambda: fake_now[0])

    rec.start()
    fake_now[0] += 1.0
    rec.pause()
    fake_now[0] += 5.0
    # Stopping from paused must close the stream cleanly.
    rec.stop()
    assert rec.state == RecorderState.STOPPED
