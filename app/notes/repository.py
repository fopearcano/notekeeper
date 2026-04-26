"""Data access layer for notebooks and notes."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.notes.database import Database
from app.notes.models import Note, NoteDraft, Notebook


def _parse_dt(value: str) -> datetime:
    # SQLite stores datetimes as ISO 8601 strings with default datetime('now').
    # ``fromisoformat`` handles both "YYYY-MM-DD HH:MM:SS" and ISO variants.
    return datetime.fromisoformat(value.replace(" ", "T"))


def _row_to_note(row) -> Note:
    return Note(
        id=row["id"],
        notebook_id=row["notebook_id"],
        title=row["title"],
        raw_transcript=row["raw_transcript"],
        processed_text=row["processed_text"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _row_to_notebook(row) -> Notebook:
    return Notebook(
        id=row["id"],
        name=row["name"],
        created_at=_parse_dt(row["created_at"]),
    )


class NoteRepository:
    """CRUD for ``notebooks`` and ``notes``."""

    def __init__(self, db: Database):
        self._db = db

    # ----- notebooks -----------------------------------------------------

    def list_notebooks(self) -> list[Notebook]:
        cur = self._db.connection.execute(
            "SELECT id, name, created_at FROM notebooks ORDER BY name"
        )
        return [_row_to_notebook(r) for r in cur.fetchall()]

    def create_notebook(self, name: str) -> Notebook:
        with self._db.transaction() as conn:
            cur = conn.execute("INSERT INTO notebooks(name) VALUES (?)", (name,))
            row = conn.execute(
                "SELECT id, name, created_at FROM notebooks WHERE id = ?",
                (cur.lastrowid,),
            ).fetchone()
        return _row_to_notebook(row)

    def get_or_create_notebook(self, name: str) -> Notebook:
        row = self._db.connection.execute(
            "SELECT id, name, created_at FROM notebooks WHERE name = ?", (name,)
        ).fetchone()
        if row is not None:
            return _row_to_notebook(row)
        return self.create_notebook(name)

    # ----- notes ---------------------------------------------------------

    def create_note(self, draft: NoteDraft) -> Note:
        with self._db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO notes(notebook_id, title, raw_transcript, processed_text)
                VALUES (?, ?, ?, ?)
                """,
                (draft.notebook_id, draft.title, draft.raw_transcript, draft.processed_text),
            )
            row = conn.execute(
                """
                SELECT id, notebook_id, title, raw_transcript, processed_text,
                       created_at, updated_at
                FROM notes WHERE id = ?
                """,
                (cur.lastrowid,),
            ).fetchone()
        return _row_to_note(row)

    def update_note(
        self,
        note_id: int,
        *,
        title: Optional[str] = None,
        raw_transcript: Optional[str] = None,
        processed_text: Optional[str] = None,
    ) -> Note:
        fields: list[str] = []
        values: list[object] = []
        if title is not None:
            fields.append("title = ?")
            values.append(title)
        if raw_transcript is not None:
            fields.append("raw_transcript = ?")
            values.append(raw_transcript)
        if processed_text is not None:
            fields.append("processed_text = ?")
            values.append(processed_text)
        if not fields:
            return self.get_note(note_id)

        fields.append("updated_at = datetime('now')")
        values.append(note_id)
        with self._db.transaction() as conn:
            conn.execute(f"UPDATE notes SET {', '.join(fields)} WHERE id = ?", values)
        return self.get_note(note_id)

    def get_note(self, note_id: int) -> Note:
        row = self._db.connection.execute(
            """
            SELECT id, notebook_id, title, raw_transcript, processed_text,
                   created_at, updated_at
            FROM notes WHERE id = ?
            """,
            (note_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Note {note_id} not found")
        return _row_to_note(row)

    def list_notes(self, notebook_id: Optional[int] = None) -> list[Note]:
        if notebook_id is None:
            cur = self._db.connection.execute(
                """
                SELECT id, notebook_id, title, raw_transcript, processed_text,
                       created_at, updated_at
                FROM notes ORDER BY updated_at DESC
                """
            )
        else:
            cur = self._db.connection.execute(
                """
                SELECT id, notebook_id, title, raw_transcript, processed_text,
                       created_at, updated_at
                FROM notes WHERE notebook_id = ? ORDER BY updated_at DESC
                """,
                (notebook_id,),
            )
        return [_row_to_note(r) for r in cur.fetchall()]

    def delete_note(self, note_id: int) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
