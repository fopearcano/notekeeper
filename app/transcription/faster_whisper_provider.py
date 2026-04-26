"""Real `faster-whisper`-backed transcription provider.

Behaviour:

* The :class:`~faster_whisper.WhisperModel` is **lazily loaded** the first time
  a chunk arrives — construction is cheap and the GUI can open before any
  ML weights touch disk. Once loaded the model is held for the lifetime of
  the provider so subsequent chunks reuse it.
* Audio chunks come in as int16 PCM bytes from :class:`AudioChunk`. We convert
  to float32 numpy in [-1, 1] (the format ``WhisperModel.transcribe`` expects)
  before calling the model.
* CUDA failures (no driver, OOM, missing model files for the requested
  precision) trigger a CPU fallback when ``faster_whisper.allow_cpu_fallback``
  is set. The compute_type is coerced because CPU doesn't support ``float16``.
* Status messages ("Loading Whisper model…", per-chunk latency) and warnings
  ("no speech detected in chunk") are forwarded through the side channels on
  :class:`TranscriptionProvider`; the pipeline forwards both up to the UI.

Diarization is intentionally not implemented yet — segments come back without
a ``speaker`` field.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Optional

import numpy as np

from app.audio.audio_buffer import AudioChunk
from app.config.settings import FasterWhisperSettings, TranscriptionSettings
from app.transcription.base import TranscriptionProvider, TranscriptSegment
from app.utils.logging import get_logger

log = get_logger(__name__)


# Tests monkeypatch this to inject a fake model class without needing the
# real faster-whisper wheel installed.
def _import_whisper_model() -> Any:
    from faster_whisper import WhisperModel  # noqa: PLC0415 - lazy import

    return WhisperModel


# CPU compute_type fallback — CTranslate2 CPU does NOT support float16.
_CPU_COMPUTE_FALLBACK = {
    "float16": "int8",
    "int8_float16": "int8",
}


def _normalize_compute_type_for_cpu(compute_type: str) -> str:
    return _CPU_COMPUTE_FALLBACK.get(compute_type, compute_type)


def _chunk_to_float32(chunk: AudioChunk) -> np.ndarray:
    """Convert int16 PCM bytes (mono) to float32 numpy in [-1, 1]."""
    if chunk.channels != 1:
        # Defensive: the recorder is configured for mono. If a multi-channel
        # chunk ever shows up we down-mix here so the model still works.
        arr = np.frombuffer(chunk.data, dtype=np.int16).reshape(-1, chunk.channels)
        arr = arr.mean(axis=1)
    else:
        arr = np.frombuffer(chunk.data, dtype=np.int16)
    return arr.astype(np.float32) / 32768.0


class FasterWhisperProvider(TranscriptionProvider):
    provider_key = "faster_whisper"
    is_stub = False

    def __init__(
        self,
        transcription: TranscriptionSettings,
        faster_whisper: FasterWhisperSettings,
    ):
        self.transcription = transcription
        self.faster_whisper = faster_whisper
        self._model: Optional[Any] = None
        self._model_lock = asyncio.Lock()
        self._active_device: Optional[str] = None
        self._active_compute_type: Optional[str] = None

    # ----- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        # Model loading is deferred to the first chunk so the UI opens
        # promptly even when the requested model is large.
        log.info(
            "FasterWhisperProvider.start (model=%s, device=%s, compute_type=%s, "
            "language=%s, allow_cpu_fallback=%s) — model load deferred",
            self.faster_whisper.model,
            self.faster_whisper.device,
            self.faster_whisper.compute_type,
            self.transcription.language,
            self.faster_whisper.allow_cpu_fallback,
        )

    async def stop(self) -> None:
        # Drop the reference so the user's GPU memory is released as soon as
        # GC runs. Re-loading on a subsequent session is acceptable.
        self._model = None
        self._active_device = None
        self._active_compute_type = None
        log.info("FasterWhisperProvider.stop")

    # ----- model loading -------------------------------------------------

    def _build_model(self, *, device: str, compute_type: str) -> Any:
        cls = _import_whisper_model()
        return cls(self.faster_whisper.model, device=device, compute_type=compute_type)

    def _load_model_sync(self) -> Any:
        """Blocking model load — called from a worker thread.

        Tries the user's configured device first; on failure falls back to
        CPU when ``allow_cpu_fallback`` is set, coercing ``compute_type`` if
        needed. Failures are re-raised; the async wrapper turns them into
        warnings so the pipeline keeps running.
        """
        primary_device = self.faster_whisper.device
        primary_compute = self.faster_whisper.compute_type

        try:
            model = self._build_model(device=primary_device, compute_type=primary_compute)
            self._active_device = primary_device
            self._active_compute_type = primary_compute
            return model
        except Exception as exc:
            primary_error = exc
            log.warning(
                "Failed to load Whisper model on %s/%s: %s",
                primary_device, primary_compute, exc,
            )

        # Only fall back when explicitly allowed and we weren't already on CPU.
        if not self.faster_whisper.allow_cpu_fallback or primary_device == "cpu":
            raise primary_error

        cpu_compute = _normalize_compute_type_for_cpu(primary_compute)
        self._emit_status(
            f"CUDA load failed ({primary_error}); falling back to CPU/{cpu_compute}…"
        )
        try:
            model = self._build_model(device="cpu", compute_type=cpu_compute)
        except Exception as cpu_exc:
            log.error("CPU fallback also failed: %s", cpu_exc)
            raise cpu_exc from primary_error
        self._active_device = "cpu"
        self._active_compute_type = cpu_compute
        return model

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is not None:
                return self._model
            self._emit_status(f"Loading Whisper model {self.faster_whisper.model}…")
            t0 = time.monotonic()
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(None, self._load_model_sync)
            elapsed = time.monotonic() - t0
            self._emit_status(
                f"Whisper {self.faster_whisper.model} loaded "
                f"({self._active_device}/{self._active_compute_type}, "
                f"{elapsed:.1f}s)"
            )
            log.info(
                "Whisper model ready (model=%s, device=%s, compute_type=%s, took=%.2fs)",
                self.faster_whisper.model,
                self._active_device,
                self._active_compute_type,
                elapsed,
            )
            return self._model

    # ----- transcription -------------------------------------------------

    def _language_arg(self) -> Optional[str]:
        lang = self.transcription.language
        if lang in ("auto", "", None):
            return None
        return lang

    def _transcribe_sync(self, model: Any, audio: np.ndarray) -> tuple[list[Any], Any]:
        """Run the model and materialise the segment iterator.

        ``WhisperModel.transcribe`` returns a generator — iterating it is what
        actually does the work. We collect into a list inside the executor so
        the asyncio loop never sees the (potentially long) work.
        """
        segments_iter, info = model.transcribe(
            audio,
            language=self._language_arg(),
            beam_size=1,
            vad_filter=False,
        )
        return list(segments_iter), info

    async def stream(
        self, chunks: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[TranscriptSegment]:
        load_failed = False

        async for chunk in chunks:
            if load_failed:
                # Drop further chunks without churning — the pipeline already warned.
                continue

            try:
                model = await self._ensure_model()
            except Exception as exc:
                load_failed = True
                # Re-raise as NotImplementedError so the pipeline's existing
                # handler emits a clean warning instead of a crash.
                raise NotImplementedError(
                    f"Whisper model could not be loaded: {exc}"
                ) from exc

            audio = _chunk_to_float32(chunk)

            t0 = time.monotonic()
            try:
                loop = asyncio.get_running_loop()
                segments, info = await loop.run_in_executor(
                    None, self._transcribe_sync, model, audio
                )
            except Exception as exc:
                self._emit_warning(f"Transcription failed: {exc}")
                log.exception("Transcription failed for chunk t=%.2f", chunk.timestamp)
                continue
            latency_ms = (time.monotonic() - t0) * 1000.0
            self._emit_status(f"Transcription latency: {latency_ms:.0f} ms")

            text_blocks = [s for s in segments if (s.text or "").strip()]
            if not text_blocks:
                self._emit_warning(
                    f"No speech detected in {chunk.duration_s:.1f}s chunk"
                )
                continue

            for s in text_blocks:
                yield TranscriptSegment(
                    text=s.text.strip(),
                    is_final=True,
                    start_s=chunk.timestamp + float(s.start),
                    end_s=chunk.timestamp + float(s.end),
                    metadata={
                        "latency_ms": latency_ms,
                        "language": getattr(info, "language", None),
                        "language_probability": getattr(info, "language_probability", None),
                    },
                )
