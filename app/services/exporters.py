"""Note exporters.

Four formats:

* **Markdown** — YAML-style frontmatter (title, dates, tags, source,
  language) followed by sections for the processed note, the raw
  transcript, and an optional processing-history section.
* **Plain text** — same content as the markdown export with the
  frontmatter rendered as ``Key: value`` lines and section breaks
  rendered as ``=== Heading ===`` markers. The processed-note body
  ships through unchanged because most users want their markdown
  preserved when they paste into a TXT consumer that understands it.
* **JSON** — full structured payload (note, segments, processing runs).
  Stable shape for tooling pipelines.
* **PDF** — markdown rendered via :class:`PySide6.QtGui.QTextDocument`
  + :class:`QPdfWriter`. Imports Qt lazily so the rest of the module
  is usable in headless / non-Qt contexts (tests, CLI scripts).

Filenames produced by :func:`safe_filename` are conservative — anything
the major OSes treat as special is replaced with ``-`` and the result is
trimmed to a sane length.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from app.notes.models import Note, ProcessingRun, Segment


# --------------------------------------------------------------------------- #
# Payload + filename helpers                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExportPayload:
    """Everything an export function needs in one bundle."""

    note: Note
    segments: Sequence[Segment] = field(default_factory=tuple)
    processing_runs: Sequence[ProcessingRun] = field(default_factory=tuple)


# Matches every non-whitespace char filesystems / shells choke on. ``\s``
# bytes are handled separately so a tab collapses to a space instead of a
# dash (otherwise ``"foo\tbar"`` would survive the bad-char pass as
# ``foo-bar`` and skip the whitespace-collapse step).
_FILENAME_BAD_CHARS = re.compile(r'[<>:"/\\|?*]')
_WS_RUN = re.compile(r"\s+")
_LEADING_TRAILING_TRIM = " .-"
#: Windows treats these as reserved device names regardless of extension.
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_filename(
    title: str,
    *,
    suffix: str = "",
    max_length: int = 100,
    fallback: str = "note",
) -> str:
    """Return a cross-platform-safe filename for ``title``.

    * Path separators, shell metacharacters, and control bytes → ``-``.
    * Whitespace runs collapse to a single space.
    * Leading/trailing dots and spaces stripped (Windows trims them anyway).
    * Truncated to ``max_length`` characters, including the optional ``suffix``.
    * Empty / reserved names → ``fallback`` (with a leading underscore for
      reserved device names so the user still sees what they typed).
    """
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    # 1. Drop control bytes outright — replacing them with '-' would survive
    #    the whitespace-collapse step below as a stray dash.
    cleaned = "".join(c for c in title if ord(c) >= 0x20 and c != "\x7f")
    # 2. Collapse whitespace runs (including \t, \n, \r) into single spaces
    #    BEFORE we replace shell metacharacters; otherwise a tab in the input
    #    would be substituted as '-' and never collapsed.
    cleaned = _WS_RUN.sub(" ", cleaned)
    # 3. Replace shell / path metacharacters with '-'.
    cleaned = _FILENAME_BAD_CHARS.sub("-", cleaned)
    # 4. Trim leading/trailing whitespace, dots, and dashes — all artifacts
    #    of the replacement passes. Repeat until stable so an input like
    #    "...---" reduces to empty.
    cleaned = cleaned.strip(_LEADING_TRAILING_TRIM)
    # 5. Fallback if the result is now empty or a string of only dashes
    #    (e.g. ``"???"`` → ``"---"`` → still nothing meaningful).
    if not cleaned or all(c == "-" for c in cleaned):
        cleaned = fallback

    upper = cleaned.upper()
    stem = upper.split(".", 1)[0]
    if stem in _WINDOWS_RESERVED:
        cleaned = "_" + cleaned

    # 6. Reserve room for the suffix so the full name fits in max_length.
    head_budget = max(1, max_length - len(suffix))
    if len(cleaned) > head_budget:
        cleaned = cleaned[:head_budget].rstrip(_LEADING_TRAILING_TRIM)
    if not cleaned:
        cleaned = fallback
    return cleaned + suffix


# --------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# --------------------------------------------------------------------------- #


def _isoformat(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def _format_tags(tags: Iterable[str]) -> str:
    items = [t for t in tags if t]
    if not items:
        return "[]"
    return "[" + ", ".join(items) + "]"


# --------------------------------------------------------------------------- #
# Markdown                                                                    #
# --------------------------------------------------------------------------- #


def to_markdown(
    payload: ExportPayload, *, include_history: bool = True
) -> str:
    """Render the payload as a single markdown document with YAML frontmatter."""
    note = payload.note
    lines: list[str] = ["---"]
    lines.append(f"title: {_yaml_scalar(note.title)}")
    lines.append(f"created: {_isoformat(note.created_at)}")
    lines.append(f"updated: {_isoformat(note.updated_at)}")
    lines.append(f"source: {_yaml_scalar(note.source)}")
    if note.language:
        lines.append(f"language: {_yaml_scalar(note.language)}")
    lines.append(f"tags: {_format_tags(note.tags)}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {note.title}")
    lines.append("")

    lines.append("## Processed Note")
    lines.append("")
    lines.append(note.processed_text.strip() or "_(empty)_")
    lines.append("")

    lines.append("## Raw Transcript")
    lines.append("")
    lines.append(note.raw_transcript.strip() or "_(empty)_")
    lines.append("")

    if include_history and payload.processing_runs:
        lines.append("## Processing History")
        lines.append("")
        for run in payload.processing_runs:
            stamp = _isoformat(run.created_at)
            lines.append(
                f"### {stamp} — `{run.task}` ({run.provider}/{run.model})"
            )
            lines.append("")
            lines.append("```")
            lines.append(run.output.strip() or "(no output)")
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _yaml_scalar(value: str) -> str:
    """Quote a YAML scalar only when necessary."""
    if value == "":
        return '""'
    if any(c in value for c in ":\n\"'#") or value.strip() != value:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


# --------------------------------------------------------------------------- #
# Plain text                                                                  #
# --------------------------------------------------------------------------- #


def to_text(
    payload: ExportPayload, *, include_history: bool = True
) -> str:
    note = payload.note
    lines: list[str] = []
    lines.append(f"Title: {note.title}")
    lines.append(f"Created: {_isoformat(note.created_at)}")
    lines.append(f"Updated: {_isoformat(note.updated_at)}")
    lines.append(f"Source: {note.source}")
    if note.language:
        lines.append(f"Language: {note.language}")
    if note.tags:
        lines.append("Tags: " + ", ".join(note.tags))
    lines.append("")
    lines.append("=== Processed Note ===")
    lines.append("")
    lines.append(note.processed_text.strip() or "(empty)")
    lines.append("")
    lines.append("=== Raw Transcript ===")
    lines.append("")
    lines.append(note.raw_transcript.strip() or "(empty)")
    lines.append("")

    if include_history and payload.processing_runs:
        lines.append("=== Processing History ===")
        lines.append("")
        for run in payload.processing_runs:
            stamp = _isoformat(run.created_at)
            lines.append(f"--- {stamp} — {run.task} ({run.provider}/{run.model}) ---")
            lines.append(run.output.strip() or "(no output)")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# JSON                                                                        #
# --------------------------------------------------------------------------- #


def to_json(
    payload: ExportPayload,
    *,
    include_history: bool = True,
    indent: Optional[int] = 2,
) -> str:
    note = payload.note
    out = {
        "id": note.id,
        "title": note.title,
        "created_at": _isoformat(note.created_at),
        "updated_at": _isoformat(note.updated_at),
        "source": note.source,
        "language": note.language,
        "tags": list(note.tags),
        "raw_transcript": note.raw_transcript,
        "processed_text": note.processed_text,
        "segments": [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
                "confidence": s.confidence,
                "created_at": _isoformat(s.created_at),
            }
            for s in payload.segments
        ],
    }
    if include_history:
        out["processing_runs"] = [
            {
                "id": r.id,
                "provider": r.provider,
                "model": r.model,
                "task": r.task,
                "prompt": r.prompt,
                "output": r.output,
                "created_at": _isoformat(r.created_at),
            }
            for r in payload.processing_runs
        ]
    # Force UTF-8 + stable key ordering inside dicts (insertion order) and
    # opt out of ASCII-escaping so unicode tag names round-trip readably.
    return json.dumps(out, indent=indent, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# PDF                                                                         #
# --------------------------------------------------------------------------- #


def to_pdf_bytes(
    payload: ExportPayload, *, include_history: bool = True
) -> bytes:
    """Render the markdown export to PDF using Qt.

    Requires a live ``QApplication`` (which Notekeeper always has).
    Imports Qt locally so headless callers / tests can use the rest of
    this module without a Qt dependency at module import time.
    """
    # Local imports: keep the whole module headless-compatible.
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QPageSize, QPdfWriter, QTextDocument
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        raise RuntimeError(
            "PDF export needs a QApplication — open Notekeeper first or "
            "construct one before calling to_pdf_bytes()."
        )

    document = QTextDocument()
    document.setMarkdown(to_markdown(payload, include_history=include_history))

    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.ReadWrite)
    writer = QPdfWriter(buffer)
    writer.setResolution(150)
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    document.print_(writer)
    data: QByteArray = buffer.data()
    buffer.close()
    return bytes(data)


# --------------------------------------------------------------------------- #
# Convenience: write to disk                                                  #
# --------------------------------------------------------------------------- #


def write_export(
    path: Path,
    payload: ExportPayload,
    fmt: str,
    *,
    include_history: bool = True,
) -> Path:
    """Render ``payload`` in ``fmt`` (``md|txt|json|pdf``) and write to ``path``."""
    fmt = fmt.lower()
    if fmt in {"md", "markdown"}:
        path.write_text(
            to_markdown(payload, include_history=include_history), encoding="utf-8"
        )
        return path
    if fmt in {"txt", "text"}:
        path.write_text(
            to_text(payload, include_history=include_history), encoding="utf-8"
        )
        return path
    if fmt == "json":
        path.write_text(
            to_json(payload, include_history=include_history), encoding="utf-8"
        )
        return path
    if fmt == "pdf":
        path.write_bytes(to_pdf_bytes(payload, include_history=include_history))
        return path
    raise ValueError(f"Unknown export format: {fmt!r}")
