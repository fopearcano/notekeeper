"""Lightweight SQLite wrapper.

Uses the standard library ``sqlite3`` module — no external ORM. The schema is
created idempotently the first time a connection is opened.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.utils.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notebooks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    notebook_id     INTEGER REFERENCES notebooks(id) ON DELETE SET NULL,
    title           TEXT    NOT NULL DEFAULT 'Untitled note',
    raw_transcript  TEXT    NOT NULL DEFAULT '',
    processed_text  TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_notes_notebook ON notes(notebook_id);
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
