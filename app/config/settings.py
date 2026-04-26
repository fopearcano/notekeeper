"""Pydantic-backed configuration models and TOML loader.

Settings are loaded by merging three layers (later wins):

1. ``app/config/default_config.toml`` shipped with the package.
2. ``~/.config/notekeeper/config.toml`` if it exists.
3. A small set of environment variables for secrets.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.toml"


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
    sample_rate: int = 16000
    channels: int = Field(default=1, ge=1, le=2)
    chunk_ms: int = Field(default=30, ge=10, le=200)
    device: str = "default"
    vad_aggressiveness: int = Field(default=2, ge=0, le=3)


TranscriptionProvider = Literal["faster_whisper", "openai", "lmstudio_stub"]


class TranscriptionSettings(BaseModel):
    provider: TranscriptionProvider = "faster_whisper"
    model: str = "base.en"
    language: str = "en"
    emit_interval_ms: int = Field(default=250, ge=50, le=2000)


LLMProvider = Literal["lmstudio", "openai", "anthropic"]


class LLMSettings(BaseModel):
    provider: LLMProvider = "lmstudio"
    model: str = "local-model"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = ""
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1)
    request_timeout_s: int = Field(default=60, ge=1)


class StorageSettings(BaseModel):
    database_path: str = "notekeeper.db"

    def resolved_path(self, data_dir: Path) -> Path:
        p = Path(self.database_path).expanduser()
        return p if p.is_absolute() else data_dir / p


class AppSettings(BaseModel):
    app: AppMeta = Field(default_factory=AppMeta)
    ui: UISettings = Field(default_factory=UISettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)


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


def _user_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "notekeeper" / "config.toml"


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    llm = data.setdefault("llm", {})
    provider = llm.get("provider", "lmstudio")
    if provider == "openai" and (key := os.environ.get("OPENAI_API_KEY")):
        llm["api_key"] = key
    if provider == "anthropic" and (key := os.environ.get("ANTHROPIC_API_KEY")):
        llm["api_key"] = key
    return data


def load_settings(
    *,
    user_config: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppSettings:
    """Load and validate settings.

    Parameters
    ----------
    user_config:
        Optional override path. Defaults to ``~/.config/notekeeper/config.toml``.
    overrides:
        Final dict layered on top — useful from tests.
    """
    data = _read_toml(DEFAULT_CONFIG_PATH)

    user_path = user_config if user_config is not None else _user_config_path()
    if user_path.exists():
        data = _deep_merge(data, _read_toml(user_path))

    data = _apply_env_overrides(data)

    if overrides:
        data = _deep_merge(data, overrides)

    return AppSettings.model_validate(data)


def user_data_dir() -> Path:
    """Best-effort cross-platform per-user data directory."""
    if (xdg := os.environ.get("XDG_DATA_HOME")):
        base = Path(xdg)
    else:
        base = Path.home() / ".local" / "share"
    path = base / "notekeeper"
    path.mkdir(parents=True, exist_ok=True)
    return path
