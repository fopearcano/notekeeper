"""Pydantic-backed configuration models and TOML loader.

On first launch the bundled ``app/config/default_config.toml`` is copied to
``~/.notekeeper/config.toml``. Subsequent launches load that user file (with
the bundled defaults as a base layer so missing keys still validate).

Provider API keys are resolved from environment variables — each provider
section declares the variable name in ``api_key_env``. The exception is the
LM Studio LLM section which carries an inline ``api_key`` because LM Studio
uses a static placeholder string.
"""

from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.toml"

TranscriptionProviderName = Literal["faster_whisper", "openai_audio", "lmstudio_audio"]
LLMProviderName = Literal["lmstudio", "openai", "anthropic"]


# --------------------------------------------------------------------------- #
# Base + meta sections                                                        #
# --------------------------------------------------------------------------- #


class AppMeta(BaseModel):
    name: str = "Notekeeper"
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class UISettings(BaseModel):
    window_title: str = "Notekeeper"
    window_width: int = Field(default=1280, ge=640)
    window_height: int = Field(default=800, ge=480)


class AudioSettings(BaseModel):
    """Capture-side audio settings (sample rate lives on the transcription section).

    ``input_device`` may be ``None`` (system default), a device name string,
    or a numeric index. ``sounddevice.query_devices()`` accepts any of those.

    The energy-based VAD has only two knobs:

    * ``vad_enabled`` — when ``False`` every chunk is sent to transcription.
    * ``vad_threshold`` — RMS in [0, 1]; chunks below this are skipped at
      transcription time but still advance the timeline so timestamps stay
      anchored to wall-clock recording time.
    """

    channels: int = Field(default=1, ge=1, le=2)
    input_device: Optional[Any] = None  # None | str | int
    vad_enabled: bool = True
    vad_threshold: float = Field(default=0.01, ge=0.0, le=1.0)


class StorageSettings(BaseModel):
    database_path: str = "notekeeper.db"

    def resolved_path(self, data_dir: Path) -> Path:
        p = Path(self.database_path).expanduser()
        return p if p.is_absolute() else data_dir / p


# --------------------------------------------------------------------------- #
# Transcription                                                               #
# --------------------------------------------------------------------------- #


class TranscriptionSettings(BaseModel):
    provider: TranscriptionProviderName = "faster_whisper"
    language: str = "auto"
    chunk_seconds: int = Field(default=5, ge=1, le=60)
    sample_rate: int = Field(default=16000, ge=8000, le=48000)


class FasterWhisperSettings(BaseModel):
    model: str = "small"
    device: str = "cuda"  # "cuda", "cpu", "auto"
    compute_type: str = "float16"  # "int8", "int8_float16", "float16", "float32"
    allow_cpu_fallback: bool = True


class _APIKeyFromEnvMixin(BaseModel):
    api_key_env: str

    def resolve_api_key(self) -> str:
        """Return the API key from the configured environment variable.

        Returns an empty string when the variable is unset; providers should
        raise a clear error at request time rather than at import time.
        """
        return os.environ.get(self.api_key_env, "")


class OpenAIAudioSettings(_APIKeyFromEnvMixin):
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "whisper-1"


class LMStudioAudioSettings(BaseModel):
    base_url: str = "http://192.168.1.100:1234/v1"
    model: str = "whisper-local"
    enabled: bool = False


# --------------------------------------------------------------------------- #
# LLM                                                                         #
# --------------------------------------------------------------------------- #


class LLMSettings(BaseModel):
    provider: LLMProviderName = "lmstudio"
    default_task: str = "clean"
    #: Index into ``llm_servers`` selecting the active LM Studio server.
    #: Ignored when ``llm_servers`` is empty.
    active_server: int = Field(default=0, ge=0)


class LLMServer(BaseModel):
    """One LM Studio server (or compatible) on the LAN.

    A list of these lives at the top level under ``[[llm_servers]]``.
    When non-empty, the entry at ``llm.active_server`` overrides
    ``[lmstudio]`` for the LLM factory; setting an out-of-range index
    falls back to the legacy single-server config silently.
    """

    name: str
    base_url: str
    provider: Literal["lmstudio"] = "lmstudio"
    api_key: str = "lm-studio"
    #: ``""`` means "use the model from ``[lmstudio]``".
    model: str = ""
    #: ``0`` means "use the timeout from ``[lmstudio]``".
    timeout_seconds: int = Field(default=0, ge=0)


class LMStudioLLMSettings(BaseModel):
    """LM Studio uses an OpenAI-compatible chat/completions endpoint."""

    base_url: str = "http://192.168.1.100:1234/v1"
    api_key: str = "lm-studio"
    model: str = "local-model-name"
    timeout_seconds: int = Field(default=120, ge=1)


class OpenAILLMSettings(_APIKeyFromEnvMixin):
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4.1-mini"
    timeout_seconds: int = Field(default=120, ge=1)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1)


class AnthropicLLMSettings(_APIKeyFromEnvMixin):
    base_url: str = "https://api.anthropic.com/v1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-sonnet-4-5"
    timeout_seconds: int = Field(default=120, ge=1)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1)


# --------------------------------------------------------------------------- #
# Top-level                                                                   #
# --------------------------------------------------------------------------- #


class AppSettings(BaseModel):
    app: AppMeta = Field(default_factory=AppMeta)
    ui: UISettings = Field(default_factory=UISettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)

    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    faster_whisper: FasterWhisperSettings = Field(default_factory=FasterWhisperSettings)
    openai_audio: OpenAIAudioSettings = Field(default_factory=OpenAIAudioSettings)
    lmstudio_audio: LMStudioAudioSettings = Field(default_factory=LMStudioAudioSettings)

    llm: LLMSettings = Field(default_factory=LLMSettings)
    lmstudio: LMStudioLLMSettings = Field(default_factory=LMStudioLLMSettings)
    openai: OpenAILLMSettings = Field(default_factory=OpenAILLMSettings)
    anthropic: AnthropicLLMSettings = Field(default_factory=AnthropicLLMSettings)
    llm_servers: list[LLMServer] = Field(default_factory=list)

    def active_lmstudio_settings(self) -> LMStudioLLMSettings:
        """Return the effective LM Studio settings.

        When ``llm_servers`` is non-empty and ``llm.active_server`` is in
        range, the matching ``LLMServer`` overrides
        :pyattr:`AppSettings.lmstudio` (with empty / zero fields falling
        through to the base ``[lmstudio]`` section so users only set the
        fields they want to override per server).
        """
        base = self.lmstudio
        if not self.llm_servers:
            return base
        if not 0 <= self.llm.active_server < len(self.llm_servers):
            return base
        server = self.llm_servers[self.llm.active_server]
        return LMStudioLLMSettings(
            base_url=server.base_url,
            api_key=server.api_key or base.api_key,
            model=server.model or base.model,
            timeout_seconds=server.timeout_seconds or base.timeout_seconds,
        )

    def active_server_label(self) -> str:
        """Display label of the active server (falls back to base_url)."""
        if self.llm_servers and 0 <= self.llm.active_server < len(self.llm_servers):
            srv = self.llm_servers[self.llm.active_server]
            return srv.name or srv.base_url
        return self.lmstudio.base_url


# --------------------------------------------------------------------------- #
# Loader                                                                      #
# --------------------------------------------------------------------------- #


def default_user_config_path() -> Path:
    """The canonical user config location: ``~/.notekeeper/config.toml``."""
    return Path.home() / ".notekeeper" / "config.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def ensure_user_config(path: Path | None = None) -> Path:
    """Materialize the user config from defaults if missing, return its path."""
    target = path if path is not None else default_user_config_path()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEFAULT_CONFIG_PATH, target)
        log.info("Created user config at %s from bundled defaults", target)
    return target


def load_settings(
    *,
    user_config: Path | None = None,
    overrides: dict[str, Any] | None = None,
    bootstrap: bool = True,
) -> AppSettings:
    """Load and validate settings.

    Parameters
    ----------
    user_config:
        Override path. Defaults to ``~/.notekeeper/config.toml``.
    overrides:
        Final dict layered on top — useful from tests.
    bootstrap:
        When ``True`` (default) and the user config is missing, copy it from
        the bundled defaults. Tests pass ``False`` to keep the home dir clean.
    """
    data = _read_toml(DEFAULT_CONFIG_PATH)

    user_path = user_config if user_config is not None else default_user_config_path()
    if bootstrap:
        ensure_user_config(user_path)
    if user_path.exists():
        data = _deep_merge(data, _read_toml(user_path))

    if overrides:
        data = _deep_merge(data, overrides)

    return AppSettings.model_validate(data)


def user_data_dir() -> Path:
    """Per-user data directory used for the SQLite database, logs, etc."""
    base = Path.home() / ".notekeeper"
    path = base / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path
