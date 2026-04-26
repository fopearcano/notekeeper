"""Recorder tests that don't open a real audio device."""

from __future__ import annotations

import time

import pytest

from app.audio.audio_buffer import AudioBuffer
from app.audio.recorder import AudioRecorder, RecorderState
from app.config.settings import AudioSettings


def _recorder(buffer=None) -> AudioRecorder:
    return AudioRecorder(
        AudioSettings(),
        sample_rate=16000,
        chunk_seconds=0.5,
        buffer=buffer or AudioBuffer(),
    )


def test_recorder_construction_does_not_touch_audio_backend():
    rec = _recorder()
    assert rec.state == RecorderState.IDLE
    assert rec.elapsed_s() == 0.0


def test_recorder_start_failure_emits_error_and_sets_state(monkeypatch):
    """If sounddevice can't open a device, the recorder reports ERROR via callback."""
    import sounddevice as sd

    class _BoomStream:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no audio device")

    monkeypatch.setattr(sd, "InputStream", _BoomStream)

    rec = _recorder()
    errors: list[str] = []
    rec.set_error_callback(errors.append)

    rec.start()

    assert rec.state == RecorderState.ERROR
    assert errors and "no audio device" in errors[0]


def test_recorder_callback_pushes_chunks(monkeypatch):
    """Drive the audio callback directly with float32 frames and verify chunking."""
    import numpy as np
    import sounddevice as sd

    captured_callback = {}

    class _FakeStream:
        def __init__(self, *, samplerate, channels, dtype, device, callback):
            captured_callback["fn"] = callback
            captured_callback["samplerate"] = samplerate
            captured_callback["channels"] = channels

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(sd, "InputStream", _FakeStream)

    buf = AudioBuffer()
    rec = AudioRecorder(
        AudioSettings(),
        sample_rate=16000,
        chunk_seconds=0.1,  # 1600 frames per chunk
        buffer=buf,
    )

    levels: list[float] = []
    rec.set_level_callback(levels.append)

    rec.start()
    assert rec.state == RecorderState.RECORDING

    cb = captured_callback["fn"]
    # 0.25 s of fake audio at full scale → expect 2 emitted chunks (0.1 s each)
    # with a 0.05 s remainder still in the assembler.
    block = np.full((4000, 1), 0.5, dtype=np.float32)
    cb(block, 4000, None, None)

    rec.stop()
    assert rec.state == RecorderState.STOPPED

    # Two complete 0.1s chunks of int16 mono = 3200 bytes each.
    chunks = []
    while True:
        chunk = buf.pop(timeout=0.0)
        if chunk is None:
            break
        chunks.append(chunk)

    assert len(chunks) >= 2
    for c in chunks[:2]:
        assert len(c.data) == 3200
        assert c.sample_rate == 16000
        assert c.channels == 1
    # A final flushed remainder chunk may also appear after stop().
    if len(chunks) == 3:
        assert 0 < len(chunks[2].data) < 3200

    # The level callback fired at least once with a non-zero reading.
    assert levels
    assert max(levels) > 0.0
