"""Cloud / OpenAI-compatible audio transcription provider.

Sends each :class:`AudioChunk` as a small WAV file (in-memory, no temp file)
to ``POST {base_url}/audio/transcriptions`` and forwards the returned text.
Compatible with OpenAI's whisper-1 endpoint and any third-party server that
implements the same shape.

Behaviour notes:

* The HTTP client is built lazily; construction validates that the API key
  resolved from ``api_key_env`` is non-empty.
* A **404** is treated as a configuration mismatch: the user pointed
  ``transcription.provider = "openai_audio"`` at a server that doesn't
  expose ``/audio/transcriptions``. We emit a clear warning and bail out
  via ``NotImplementedError`` so the pipeline shuts down cleanly instead of
  spamming the same failure for every chunk.
* **401 / 403** are similarly bailed-out — auth errors won't fix themselves.
* **Timeouts and 5xx** are treated as transient: the chunk is dropped with a
  warning and the next one is attempted.
"""

from __future__ import annotations

import io
import time
import wave
from typing import Any, AsyncIterator, Optional

import httpx

from app.audio.audio_buffer import AudioChunk
from app.config.settings import OpenAIAudioSettings, TranscriptionSettings
from app.transcription.base import TranscriptionProvider, TranscriptSegment
from app.utils.logging import get_logger

log = get_logger(__name__)


def _chunk_to_wav_bytes(chunk: AudioChunk) -> bytes:
    """Wrap a chunk's raw int16 PCM in a minimal WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(chunk.channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(chunk.sample_rate)
        wf.writeframes(chunk.data)
    return buf.getvalue()


class _AudioEndpointMissing(RuntimeError):
    """Server returned 404 for /audio/transcriptions."""


class _AuthError(RuntimeError):
    """Server returned 401/403."""


class OpenAIAudioProvider(TranscriptionProvider):
    provider_key = "openai_audio"
    is_stub = False

    #: Allows tests to inject ``httpx.MockTransport`` without monkeypatching globals.
    _transport: Optional[Any] = None

    #: Connect / read timeout for the upload. Audio uploads are small but the
    #: server may take a few seconds for the actual transcription.
    _DEFAULT_TIMEOUT_S = 60.0

    def __init__(
        self,
        transcription: TranscriptionSettings,
        openai_audio: OpenAIAudioSettings,
    ):
        self.transcription = transcription
        self.openai_audio = openai_audio
        self._client: Optional[httpx.AsyncClient] = None

    # ----- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        log.info(
            "OpenAIAudioProvider.start (model=%s, base_url=%s)",
            self.openai_audio.model,
            self.openai_audio.base_url,
        )

    async def stop(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                log.exception("Error closing httpx client")
            self._client = None
        log.info("OpenAIAudioProvider.stop")

    # ----- internals -----------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            api_key = self.openai_audio.resolve_api_key()
            if not api_key:
                raise RuntimeError(
                    f"OpenAI audio requires an API key — set the "
                    f"${self.openai_audio.api_key_env} environment variable."
                )
            kwargs: dict[str, Any] = {
                "base_url": self.openai_audio.base_url,
                "timeout": httpx.Timeout(self._DEFAULT_TIMEOUT_S, connect=10.0),
                "headers": {"Authorization": f"Bearer {api_key}"},
            }
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    def _language_arg(self) -> Optional[str]:
        lang = self.transcription.language
        if lang in ("auto", "", None):
            return None
        return lang

    async def _transcribe(self, wav_bytes: bytes) -> str:
        client = self._get_client()
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data: dict[str, Any] = {
            "model": self.openai_audio.model,
            "response_format": "json",
        }
        lang = self._language_arg()
        if lang is not None:
            data["language"] = lang

        resp = await client.post("/audio/transcriptions", files=files, data=data)

        if resp.status_code == 404:
            raise _AudioEndpointMissing(
                f"{self.openai_audio.base_url}/audio/transcriptions returned 404. "
                "This endpoint does not support audio transcription."
            )
        if resp.status_code in (401, 403):
            raise _AuthError(
                f"Authentication failed ({resp.status_code}) when calling "
                f"{self.openai_audio.base_url}/audio/transcriptions"
            )
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError:
            return resp.text or ""
        # OpenAI: {"text": "..."} for response_format=json. Some compatible
        # servers also nest under "transcription"; try both.
        return (payload.get("text") or payload.get("transcription") or "").strip()

    # ----- streaming -----------------------------------------------------

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptSegment]:
        async for chunk in chunks:
            wav = _chunk_to_wav_bytes(chunk)
            t0 = time.monotonic()

            try:
                text = await self._transcribe(wav)
            except _AudioEndpointMissing as exc:
                self._emit_warning(str(exc))
                # Bail out of the stream — calling /audio/transcriptions on a
                # server that doesn't expose it will keep returning 404.
                raise NotImplementedError(str(exc)) from exc
            except _AuthError as exc:
                self._emit_warning(str(exc))
                raise NotImplementedError(str(exc)) from exc
            except httpx.TimeoutException as exc:
                self._emit_warning(
                    f"Transcription request timed out: {exc}; dropping chunk"
                )
                continue
            except httpx.HTTPError as exc:
                self._emit_warning(f"Transcription request failed: {exc}")
                continue

            latency_ms = (time.monotonic() - t0) * 1000.0
            self._emit_status(f"Transcription latency: {latency_ms:.0f} ms")

            if not text.strip():
                self._emit_warning(
                    f"No speech detected in {chunk.duration_s:.1f}s chunk"
                )
                continue

            yield TranscriptSegment(
                text=text.strip(),
                is_final=True,
                start_s=chunk.timestamp,
                end_s=chunk.timestamp + chunk.duration_s,
                metadata={
                    "latency_ms": latency_ms,
                    "model": self.openai_audio.model,
                },
            )
