"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when running from a fresh checkout.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.notes.database import Database  # noqa: E402
from app.notes.repository import NoteRepository  # noqa: E402


@pytest.fixture
def repository() -> NoteRepository:
    db = Database(":memory:")
    try:
        yield NoteRepository(db)
    finally:
        db.close()
