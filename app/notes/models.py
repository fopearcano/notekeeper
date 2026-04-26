"""Pydantic models for notes and notebooks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Notebook(BaseModel):
    id: int
    name: str
    created_at: datetime


class NoteDraft(BaseModel):
    """A note about to be persisted (no id yet)."""

    notebook_id: Optional[int] = None
    title: str = "Untitled note"
    raw_transcript: str = ""
    processed_text: str = ""


class Note(BaseModel):
    id: int
    notebook_id: Optional[int] = None
    title: str
    raw_transcript: str
    processed_text: str
    created_at: datetime
    updated_at: datetime = Field(default_factory=_utcnow)
