"""Voice Activity Detection placeholder.

The real implementation will likely wrap WebRTC VAD or Silero. For now this
exposes the interface used by the rest of the app and a permissive default
that always reports speech, so the pipeline can be exercised end-to-end.
"""

from __future__ import annotations

from app.audio.audio_buffer import AudioChunk


class VoiceActivityDetector:
    """Reports whether a chunk contains speech.

    Parameters
    ----------
    aggressiveness:
        Mirrors WebRTC VAD's 0..3 scale for forward compatibility.
    """

    def __init__(self, aggressiveness: int = 2):
        if not 0 <= aggressiveness <= 3:
            raise ValueError("aggressiveness must be in [0, 3]")
        self.aggressiveness = aggressiveness

    def is_speech(self, chunk: AudioChunk) -> bool:  # noqa: ARG002 - placeholder
        # TODO: wire up webrtcvad / silero. For now treat every chunk as speech
        # so downstream code receives a continuous stream during development.
        return True
