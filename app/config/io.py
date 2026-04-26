"""TOML serialization for the config file.

We only ever read TOML with the stdlib ``tomllib``; the writer here is a
small hand-rolled emitter scoped to the shapes our config actually uses
(top-level tables of scalar / list-of-scalar fields). Nesting beyond one
level, arrays of tables, inline tables, and datetime values are out of
scope — the loader's pydantic models won't produce them, and adding a
heavyweight dependency just for the write path isn't worth it.

``None`` values are skipped on write because TOML has no null. Pydantic
defaults fill the gap on the next load — every model field is either
``Optional[...] = None`` or has a defaulted scalar.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from app.config.settings import (
    AppSettings,
    DEFAULT_CONFIG_PATH,
    default_user_config_path,
)


__all__ = [
    "dumps_toml",
    "save_settings",
    "settings_to_toml_dict",
]


# --------------------------------------------------------------------------- #
# Encoders                                                                    #
# --------------------------------------------------------------------------- #


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _escape_string(s: str) -> str:
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif _CONTROL_CHAR_RE.match(ch):
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return f'"{"".join(out)}"'


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Avoid scientific notation for normal config values.
        if value != value:  # NaN
            return "nan"
        return repr(value)
    if isinstance(value, str):
        return _escape_string(value)
    raise TypeError(f"Cannot serialise {type(value).__name__!r} to TOML")


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[" + ", ".join(_format_scalar(v) for v in value) + "]"
    return _format_scalar(value)


def _format_section(name: str, fields: Mapping[str, Any]) -> list[str]:
    lines = [f"[{name}]"]
    for key, value in fields.items():
        if value is None:
            # TOML has no null — skip the key. The pydantic default fills in
            # on the next load.
            lines.append(f"# {key} = null")
            continue
        lines.append(f"{key} = {_format_value(value)}")
    return lines


def dumps_toml(data: Mapping[str, Mapping[str, Any]]) -> str:
    """Serialize a ``{section: {key: value}}`` mapping to a TOML string."""
    blocks: list[list[str]] = []
    for section, fields in data.items():
        if not isinstance(fields, Mapping):
            raise TypeError(
                f"Top-level value for {section!r} must be a mapping, "
                f"got {type(fields).__name__}"
            )
        blocks.append(_format_section(section, fields))
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


# --------------------------------------------------------------------------- #
# Settings-shaped helpers                                                     #
# --------------------------------------------------------------------------- #


def settings_to_toml_dict(settings: AppSettings) -> dict[str, dict[str, Any]]:
    """Project an :class:`AppSettings` into the layout expected by ``dumps_toml``.

    Field order matches ``default_config.toml`` so a ``git diff`` of
    ``~/.notekeeper/config.toml`` after a Settings-dialog save shows only
    the lines the user actually changed.
    """
    return {
        "app": settings.app.model_dump(),
        "ui": settings.ui.model_dump(),
        "audio": settings.audio.model_dump(),
        "transcription": settings.transcription.model_dump(),
        "faster_whisper": settings.faster_whisper.model_dump(),
        "openai_audio": settings.openai_audio.model_dump(),
        "lmstudio_audio": settings.lmstudio_audio.model_dump(),
        "llm": settings.llm.model_dump(),
        "lmstudio": settings.lmstudio.model_dump(),
        "openai": settings.openai.model_dump(),
        "anthropic": settings.anthropic.model_dump(),
        "storage": settings.storage.model_dump(),
    }


def save_settings(
    settings: AppSettings,
    *,
    path: Path | None = None,
) -> Path:
    """Persist ``settings`` to ``~/.notekeeper/config.toml`` (or ``path``).

    The write is atomic — we serialize into a temp file in the same
    directory and ``os.replace`` it into place so a crash mid-write can't
    leave the user with a half-written config.
    """
    target = path if path is not None else default_user_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    text = dumps_toml(settings_to_toml_dict(settings))

    # NamedTemporaryFile in the same directory ensures os.replace is atomic
    # on POSIX (cross-device move would not be).
    fd = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=target.parent,
        prefix=".notekeeper-",
        suffix=".toml.tmp",
    )
    try:
        fd.write(text)
        fd.flush()
        fd.close()
        Path(fd.name).replace(target)
    except Exception:
        # Best-effort cleanup; leaving a stale tmp file is better than raising
        # a different exception that masks the original.
        try:
            Path(fd.name).unlink()
        except OSError:
            pass
        raise
    return target


# Re-export so callers can ``from app.config.io import default_user_config_path``.
__all__.extend(("DEFAULT_CONFIG_PATH", "default_user_config_path"))
_ = (DEFAULT_CONFIG_PATH, default_user_config_path)  # silence "imported but unused"
