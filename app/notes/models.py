"""Pydantic models for notes, transcript segments, and processing runs.

The persisted shape mirrors the SQLite schema; transient streaming types
(:class:`app.transcription.base.TranscriptSegment`,
:class:`app.llm.prompt_templates.StructuredOutput`) are kept separate so the
storage layer can evolve without touching the live-capture path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Notes                                                                       #
# --------------------------------------------------------------------------- #


class NoteDraft(BaseModel):
    """A note about to be persisted (no id yet)."""

    title: str = "Untitled note"
    raw_transcript: str = ""
    processed_text: str = ""
    source: str = "recording"  # "recording" | "imported" | "manual"
    language: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class Note(BaseModel):
    id: int
    title: str
    raw_transcript: str
    processed_text: str
    created_at: datetime
    updated_at: datetime = Field(default_factory=_utcnow)
    source: str = "recording"
    language: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class NoteSummary(BaseModel):
    """Compact projection used for the sidebar list."""

    id: int
    title: str
    updated_at: datetime
    source: str
    language: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Transcript segments                                                         #
# --------------------------------------------------------------------------- #


class SegmentDraft(BaseModel):
    """A transcript segment about to be persisted."""

    start_time: float = 0.0
    end_time: float = 0.0
    text: str
    confidence: Optional[float] = None


class Segment(BaseModel):
    id: int
    note_id: int
    start_time: float
    end_time: float
    text: str
    confidence: Optional[float] = None
    created_at: datetime


# --------------------------------------------------------------------------- #
# Processing runs                                                             #
# --------------------------------------------------------------------------- #


class ProcessingRunDraft(BaseModel):
    provider: str
    model: str
    task: str
    prompt: str
    output: str


class ProcessingRun(BaseModel):
    id: int
    note_id: int
    provider: str
    model: str
    task: str
    prompt: str
    output: str
    created_at: datetime
