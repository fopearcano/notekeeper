"""CRUD tests for ``NoteRepository`` (notes + segments + processing runs)."""

from __future__ import annotations

import pytest

from app.notes.models import (
    NoteDraft,
    ProcessingRunDraft,
    SegmentDraft,
)


# ---------- notes -----------------------------------------------------------


def test_create_note_round_trip(repository):
    draft = NoteDraft(
        title="Hello",
        raw_transcript="raw text",
        processed_text="cleaned text",
        source="recording",
        language="en",
        tags=["meeting", "Q4"],
    )
    note = repository.create_note(draft)

    assert note.id > 0
    fetched = repository.get_note(note.id)
    assert fetched.title == "Hello"
    assert fetched.raw_transcript == "raw text"
    assert fetched.processed_text == "cleaned text"
    assert fetched.source == "recording"
    assert fetched.language == "en"
    assert fetched.tags == ["meeting", "Q4"]


def test_create_note_defaults_are_sensible(repository):
    note = repository.create_note(NoteDraft(title="bare"))
    assert note.raw_transcript == ""
    assert note.processed_text == ""
    assert note.source == "recording"
    assert note.language is None
    assert note.tags == []


def test_update_note_partial(repository):
    note = repository.create_note(NoteDraft(title="orig"))
    updated = repository.update_note(
        note.id, title="renamed", tags=["a", "b"]
    )
    assert updated.title == "renamed"
    assert updated.tags == ["a", "b"]
    # Untouched fields stay put.
    assert updated.raw_transcript == ""

    again = repository.update_note(note.id, processed_text="cleaned")
    assert again.title == "renamed"
    assert again.processed_text == "cleaned"
    assert again.tags == ["a", "b"]


def test_update_note_with_no_fields_is_a_noop(repository):
    note = repository.create_note(NoteDraft(title="x"))
    same = repository.update_note(note.id)
    assert same.id == note.id
    assert same.title == "x"


def test_update_note_changes_updated_at(repository):
    import time

    note = repository.create_note(NoteDraft(title="x"))
    time.sleep(1.1)  # SQLite datetime('now') is second-resolution.
    updated = repository.update_note(note.id, title="y")
    assert updated.updated_at >= note.updated_at


def test_get_note_missing_raises(repository):
    with pytest.raises(KeyError):
        repository.get_note(9999)


def test_list_notes_orders_by_updated_at_desc(repository):
    import time

    a = repository.create_note(NoteDraft(title="A"))
    time.sleep(1.1)
    b = repository.create_note(NoteDraft(title="B"))

    summaries = repository.list_notes()
    assert [s.id for s in summaries] == [b.id, a.id]

    # Touching A bumps it to the top.
    time.sleep(1.1)
    repository.update_note(a.id, title="A2")
    summaries = repository.list_notes()
    assert summaries[0].id == a.id


def test_list_notes_respects_limit(repository):
    for i in range(5):
        repository.create_note(NoteDraft(title=f"n{i}"))
    assert len(repository.list_notes(limit=3)) == 3


def test_delete_note_cascades_segments_and_runs(repository):
    note = repository.create_note(NoteDraft(title="x"))
    repository.replace_segments(
        note.id,
        [SegmentDraft(start_time=0.0, end_time=1.0, text="hi")],
    )
    repository.add_processing_run(
        note.id,
        ProcessingRunDraft(
            provider="lmstudio",
            model="m",
            task="clean",
            prompt="p",
            output="o",
        ),
    )

    repository.delete_note(note.id)

    with pytest.raises(KeyError):
        repository.get_note(note.id)
    assert repository.list_segments(note.id) == []
    assert repository.list_processing_runs(note.id) == []


# ---------- tag round-trip --------------------------------------------------


def test_tags_round_trip_and_handle_special_characters(repository):
    note = repository.create_note(
        NoteDraft(title="t", tags=["alpha", "beta-gamma", "with space", "中文"])
    )
    fetched = repository.get_note(note.id)
    assert fetched.tags == ["alpha", "beta-gamma", "with space", "中文"]


def test_tags_can_be_cleared(repository):
    note = repository.create_note(NoteDraft(title="t", tags=["a"]))
    updated = repository.update_note(note.id, tags=[])
    assert updated.tags == []


# ---------- segments --------------------------------------------------------


def test_replace_segments_atomically(repository):
    note = repository.create_note(NoteDraft(title="x"))

    first = repository.replace_segments(
        note.id,
        [
            SegmentDraft(start_time=0.0, end_time=1.0, text="hello"),
            SegmentDraft(start_time=1.0, end_time=2.0, text="world", confidence=0.95),
        ],
    )
    assert [s.text for s in first] == ["hello", "world"]
    assert first[1].confidence == 0.95

    # Replacing wipes the previous set.
    second = repository.replace_segments(
        note.id,
        [SegmentDraft(start_time=0.0, end_time=0.5, text="just one")],
    )
    assert [s.text for s in second] == ["just one"]
    # And the underlying list endpoint agrees.
    assert [s.text for s in repository.list_segments(note.id)] == ["just one"]


def test_add_segment_appends(repository):
    note = repository.create_note(NoteDraft(title="x"))
    s1 = repository.add_segment(
        note.id, SegmentDraft(start_time=0.0, end_time=1.0, text="hi")
    )
    s2 = repository.add_segment(
        note.id, SegmentDraft(start_time=1.0, end_time=2.0, text="there")
    )
    listed = repository.list_segments(note.id)
    assert [s.id for s in listed] == [s1.id, s2.id]
    assert [s.text for s in listed] == ["hi", "there"]


def test_segments_for_unknown_note_is_empty(repository):
    assert repository.list_segments(12345) == []


def test_segments_ordered_by_start_time(repository):
    note = repository.create_note(NoteDraft(title="x"))
    repository.replace_segments(
        note.id,
        [
            SegmentDraft(start_time=2.0, end_time=3.0, text="late"),
            SegmentDraft(start_time=0.0, end_time=1.0, text="early"),
            SegmentDraft(start_time=1.0, end_time=2.0, text="mid"),
        ],
    )
    listed = repository.list_segments(note.id)
    assert [s.text for s in listed] == ["early", "mid", "late"]


# ---------- processing runs -------------------------------------------------


def test_add_processing_run_round_trip(repository):
    note = repository.create_note(NoteDraft(title="x"))
    run = repository.add_processing_run(
        note.id,
        ProcessingRunDraft(
            provider="lmstudio",
            model="local-7b",
            task="summarize",
            prompt="SYSTEM:\n…\n\nUSER:\n…",
            output="# Summary\n\nbullet",
        ),
    )
    assert run.id > 0
    runs = repository.list_processing_runs(note.id)
    assert [r.id for r in runs] == [run.id]
    assert runs[0].provider == "lmstudio"
    assert runs[0].task == "summarize"
    assert runs[0].output.startswith("# Summary")


def test_processing_runs_ordered_by_creation(repository):
    note = repository.create_note(NoteDraft(title="x"))
    a = repository.add_processing_run(
        note.id,
        ProcessingRunDraft(
            provider="lmstudio", model="m", task="clean", prompt="p", output="o"
        ),
    )
    b = repository.add_processing_run(
        note.id,
        ProcessingRunDraft(
            provider="lmstudio", model="m", task="summarize", prompt="p", output="o"
        ),
    )
    runs = repository.list_processing_runs(note.id)
    # Same created_at granularity → ordering falls through to id.
    assert [r.id for r in runs] == [a.id, b.id]
    assert [r.task for r in runs] == ["clean", "summarize"]
