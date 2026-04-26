"""Data access layer for notes, segments, and processing runs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable, Optional, Sequence

from app.notes.database import Database
from app.notes.models import (
    Note,
    NoteDraft,
    NoteSummary,
    ProcessingRun,
    ProcessingRunDraft,
    Segment,
    SegmentDraft,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace(" ", "T"))


def _decode_tags(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [str(item) for item in decoded if isinstance(item, str)]


def _encode_tags(tags: Sequence[str]) -> str:
    return json.dumps(list(tags))


def _row_to_note(row) -> Note:
    return Note(
        id=row["id"],
        title=row["title"],
        raw_transcript=row["raw_transcript"],
        processed_text=row["processed_text"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        source=row["source"],
        language=row["language"],
        tags=_decode_tags(row["tags_json"]),
    )


def _row_to_summary(row) -> NoteSummary:
    return NoteSummary(
        id=row["id"],
        title=row["title"],
        updated_at=_parse_dt(row["updated_at"]),
        source=row["source"],
        language=row["language"],
        tags=_decode_tags(row["tags_json"]),
    )


def _row_to_segment(row) -> Segment:
    return Segment(
        id=row["id"],
        note_id=row["note_id"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        text=row["text"],
        confidence=row["confidence"],
        created_at=_parse_dt(row["created_at"]),
    )


def _row_to_run(row) -> ProcessingRun:
    return ProcessingRun(
        id=row["id"],
        note_id=row["note_id"],
        provider=row["provider"],
        model=row["model"],
        task=row["task"],
        prompt=row["prompt"],
        output=row["output"],
        created_at=_parse_dt(row["created_at"]),
    )


# --------------------------------------------------------------------------- #
# Repository                                                                  #
# --------------------------------------------------------------------------- #


_NOTE_COLUMNS = (
    "id, title, raw_transcript, processed_text, created_at, updated_at, "
    "source, language, tags_json"
)


class NoteRepository:
    """CRUD for notes plus their segment / processing-run children."""

    def __init__(self, db: Database):
        self._db = db

    # ----- notes ---------------------------------------------------------

    def create_note(self, draft: NoteDraft) -> Note:
        with self._db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO notes
                  (title, raw_transcript, processed_text, source, language, tags_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    draft.title,
                    draft.raw_transcript,
                    draft.processed_text,
                    draft.source,
                    draft.language,
                    _encode_tags(draft.tags),
                ),
            )
            row = conn.execute(
                f"SELECT {_NOTE_COLUMNS} FROM notes WHERE id = ?",
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
        source: Optional[str] = None,
        language: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
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
        if source is not None:
            fields.append("source = ?")
            values.append(source)
        if language is not None:
            fields.append("language = ?")
            values.append(language)
        if tags is not None:
            fields.append("tags_json = ?")
            values.append(_encode_tags(tags))

        if not fields:
            return self.get_note(note_id)

        fields.append("updated_at = datetime('now')")
        values.append(note_id)
        with self._db.transaction() as conn:
            conn.execute(
                f"UPDATE notes SET {', '.join(fields)} WHERE id = ?",
                values,
            )
        return self.get_note(note_id)

    def get_note(self, note_id: int) -> Note:
        row = self._db.connection.execute(
            f"SELECT {_NOTE_COLUMNS} FROM notes WHERE id = ?",
            (note_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Note {note_id} not found")
        return _row_to_note(row)

    def list_notes(self, *, limit: Optional[int] = None) -> list[NoteSummary]:
        # ``id DESC`` tiebreaker so two notes saved within the same second
        # have a stable order (datetime('now') is second-resolution).
        sql = (
            f"SELECT {_NOTE_COLUMNS} FROM notes "
            "ORDER BY datetime(updated_at) DESC, id DESC"
        )
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self._db.connection.execute(sql, params).fetchall()
        return [_row_to_summary(r) for r in rows]

    def delete_note(self, note_id: int) -> None:
        # Cascade deletes segments and processing runs (FK ON DELETE CASCADE).
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    # ----- segments ------------------------------------------------------

    def replace_segments(
        self, note_id: int, segments: Iterable[SegmentDraft]
    ) -> list[Segment]:
        """Replace all segments for ``note_id`` atomically."""
        items = list(segments)
        with self._db.transaction() as conn:
            conn.execute(
                "DELETE FROM transcript_segments WHERE note_id = ?", (note_id,)
            )
            if items:
                conn.executemany(
                    """
                    INSERT INTO transcript_segments
                      (note_id, start_time, end_time, text, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            note_id,
                            s.start_time,
                            s.end_time,
                            s.text,
                            s.confidence,
                        )
                        for s in items
                    ],
                )
        return self.list_segments(note_id)

    def add_segment(self, note_id: int, draft: SegmentDraft) -> Segment:
        with self._db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO transcript_segments
                  (note_id, start_time, end_time, text, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (note_id, draft.start_time, draft.end_time, draft.text, draft.confidence),
            )
            row = conn.execute(
                """
                SELECT id, note_id, start_time, end_time, text, confidence, created_at
                FROM transcript_segments WHERE id = ?
                """,
                (cur.lastrowid,),
            ).fetchone()
        return _row_to_segment(row)

    def list_segments(self, note_id: int) -> list[Segment]:
        rows = self._db.connection.execute(
            """
            SELECT id, note_id, start_time, end_time, text, confidence, created_at
            FROM transcript_segments WHERE note_id = ?
            ORDER BY start_time, id
            """,
            (note_id,),
        ).fetchall()
        return [_row_to_segment(r) for r in rows]

    # ----- processing runs -----------------------------------------------

    def add_processing_run(
        self, note_id: int, draft: ProcessingRunDraft
    ) -> ProcessingRun:
        with self._db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO processing_runs
                  (note_id, provider, model, task, prompt, output)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    note_id,
                    draft.provider,
                    draft.model,
                    draft.task,
                    draft.prompt,
                    draft.output,
                ),
            )
            row = conn.execute(
                """
                SELECT id, note_id, provider, model, task, prompt, output, created_at
                FROM processing_runs WHERE id = ?
                """,
                (cur.lastrowid,),
            ).fetchone()
        return _row_to_run(row)

    def list_processing_runs(self, note_id: int) -> list[ProcessingRun]:
        rows = self._db.connection.execute(
            """
            SELECT id, note_id, provider, model, task, prompt, output, created_at
            FROM processing_runs WHERE note_id = ?
            ORDER BY datetime(created_at), id
            """,
            (note_id,),
        ).fetchall()
        return [_row_to_run(r) for r in rows]
