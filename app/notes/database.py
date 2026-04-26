"""SQLite wrapper.

Uses the standard library ``sqlite3`` module — no external ORM. The schema is
created idempotently the first time a connection is opened.

Three tables:

* ``notes`` — top-level note record.
* ``transcript_segments`` — one row per emitted :class:`TranscriptSegment`,
  keyed by ``note_id``. Replaced wholesale on autosave so the order on disk
  matches the order the model produced.
* ``processing_runs`` — one row per LLM task invocation: provider, model,
  task name, full prompt, and full output. Keeps an audit trail per note.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.utils.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL DEFAULT 'Untitled note',
    raw_transcript  TEXT    NOT NULL DEFAULT '',
    processed_text  TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    source          TEXT    NOT NULL DEFAULT 'recording',
    language        TEXT,
    tags_json       TEXT    NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id     INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    start_time  REAL    NOT NULL,
    end_time    REAL    NOT NULL,
    text        TEXT    NOT NULL,
    confidence  REAL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_transcript_segments_note
    ON transcript_segments(note_id);

CREATE TABLE IF NOT EXISTS processing_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id     INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    provider    TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    task        TEXT    NOT NULL,
    prompt      TEXT    NOT NULL,
    output      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_processing_runs_note
    ON processing_runs(note_id);
"""


class Database:
    """Connection holder responsible for schema setup."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()
        log.debug("Opened SQLite database at %s", self.path)

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()
