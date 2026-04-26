"""Quick "is this provider reachable?" probes used by the Settings dialog.

Each test takes a snapshot of the relevant settings, builds an ephemeral
client, and returns ``ProbeResult(ok, message)``. The dialog renders the
message verbatim, so it should be one short line and never include secret
material — tests resolve API keys from the configured environment variable
but only display "(set)" or "(not set)", never the value itself.

Tests are coroutines so the dialog can run them on the existing async
worker thread; the dialog converts the result back into a Qt signal.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config.settings import (
    AnthropicLLMSettings,
    AppSettings,
    LMStudioLLMSettings,
    OpenAIAudioSettings,
    OpenAILLMSettings,
)
from app.llm.lmstudio_provider import LMStudioProvider
from app.transcription.factory import create_transcription_provider
from app.utils.logging import get_logger

log = get_logger(__name__)


_ANTHROPIC_VERSION = "2023-06-01"
_HTTP_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    message: str


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _key_from_env(env_name: str) -> Optional[str]:
    value = os.environ.get(env_name, "")
    return value or None


def _format_models(models: list[str]) -> str:
    if not models:
        return "no models reported"
    head = ", ".join(models[:3])
    extra = f" (+{len(models) - 3} more)" if len(models) > 3 else ""
    return f"{len(models)} model(s): {head}{extra}"


# --------------------------------------------------------------------------- #
# Individual probes                                                           #
# --------------------------------------------------------------------------- #


async def probe_transcription(settings: AppSettings) -> ProbeResult:
    """Probe the configured transcription provider.

    For ``faster_whisper`` we just import its model class without loading
    weights — that's enough to catch the common "package not installed"
    failure mode without paying the multi-second model-load cost. For the
    HTTP-shaped providers we issue a single GET against their ``/models``
    endpoint via the corresponding LLM helper since the audio routes don't
    have a cheap ping.
    """
    provider_key = settings.transcription.provider
    if provider_key == "faster_whisper":
        try:
            from app.transcription.faster_whisper_provider import (
                _import_whisper_model,
            )

            cls = _import_whisper_model()
        except Exception as exc:
            return ProbeResult(
                False,
                f"faster-whisper not importable ({type(exc).__name__}: {exc})",
            )
        return ProbeResult(
            True,
            f"faster-whisper {getattr(cls, '__name__', 'WhisperModel')} importable; "
            f"model load deferred to first chunk.",
        )

    if provider_key == "openai_audio":
        return await _test_openai_compatible(
            base_url=settings.openai_audio.base_url,
            api_key=settings.openai_audio.resolve_api_key(),
            api_key_env=settings.openai_audio.api_key_env,
            label="OpenAI audio",
        )

    if provider_key == "lmstudio_audio":
        return ProbeResult(
            False,
            "lmstudio_audio is a disabled experimental stub — pick "
            "faster_whisper or openai_audio for real transcription.",
        )

    # Defensive: pydantic should have validated this already.
    provider = create_transcription_provider(settings)
    return ProbeResult(True, f"{provider.provider_key} provider constructed.")


async def probe_lmstudio(settings: AppSettings) -> ProbeResult:
    """Hit ``GET {base_url}/models`` via the LM Studio provider."""
    return await _list_models_via(
        provider=LMStudioProvider(settings.lmstudio),
        label="LM Studio",
        base_url=settings.lmstudio.base_url,
    )


async def probe_openai(settings: AppSettings) -> ProbeResult:
    return await _test_openai_compatible(
        base_url=settings.openai.base_url,
        api_key=settings.openai.resolve_api_key(),
        api_key_env=settings.openai.api_key_env,
        label="OpenAI",
    )


async def probe_anthropic(settings: AppSettings) -> ProbeResult:
    api_key = settings.anthropic.resolve_api_key()
    if not api_key:
        return ProbeResult(
            False,
            f"Anthropic API key env var ${settings.anthropic.api_key_env} is not set.",
        )
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
    }
    try:
        async with httpx.AsyncClient(
            base_url=settings.anthropic.base_url,
            timeout=_HTTP_TIMEOUT,
            headers=headers,
        ) as client:
            resp = await client.get("/models")
        return _result_for_models_response(resp, label="Anthropic")
    except httpx.HTTPError as exc:
        return ProbeResult(False, f"Anthropic request failed: {exc}")


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


async def _test_openai_compatible(
    *,
    base_url: str,
    api_key: Optional[str],
    api_key_env: str,
    label: str,
) -> ProbeResult:
    if not api_key:
        return ProbeResult(
            False, f"{label} API key env var ${api_key_env} is not set."
        )
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(
            base_url=base_url, timeout=_HTTP_TIMEOUT, headers=headers
        ) as client:
            resp = await client.get("/models")
    except httpx.HTTPError as exc:
        return ProbeResult(False, f"{label} request failed: {exc}")
    return _result_for_models_response(resp, label=label)


def _result_for_models_response(
    resp: httpx.Response, *, label: str
) -> ProbeResult:
    if resp.status_code in (401, 403):
        return ProbeResult(
            False, f"{label} authentication failed ({resp.status_code})."
        )
    if resp.status_code == 404:
        return ProbeResult(
            False,
            f"{label} server returned 404 — endpoint does not expose /models.",
        )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return ProbeResult(False, f"{label} returned {exc.response.status_code}.")
    try:
        data = resp.json()
    except ValueError:
        return ProbeResult(False, f"{label}: response was not JSON.")
    raw = data.get("data") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return ProbeResult(False, f"{label}: unexpected response shape.")
    models = [m.get("id") for m in raw if isinstance(m, dict) and isinstance(m.get("id"), str)]
    return ProbeResult(True, f"{label} OK — {_format_models(models)}.")


async def _list_models_via(
    *, provider: LMStudioProvider, label: str, base_url: str
) -> ProbeResult:
    try:
        models = await provider.list_models()
    except httpx.HTTPError as exc:
        return ProbeResult(False, f"{label} request to {base_url} failed: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface as an error message
        return ProbeResult(False, f"{label} probe raised: {exc}")
    finally:
        try:
            await provider.aclose()
        except Exception:
            log.exception("aclose raised during %s probe", label)
    return ProbeResult(True, f"{label} OK — {_format_models(models)}.")
