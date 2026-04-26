from app.notes.models import NoteDraft


def test_notebook_round_trip(repository):
    nb = repository.create_notebook("Daily")
    assert nb.id > 0
    assert repository.get_or_create_notebook("Daily").id == nb.id
    assert [n.name for n in repository.list_notebooks()] == ["Daily"]


def test_note_round_trip(repository):
    nb = repository.get_or_create_notebook("Inbox")
    draft = NoteDraft(
        notebook_id=nb.id,
        title="Hello",
        raw_transcript="raw text",
        processed_text="cleaned text",
    )
    note = repository.create_note(draft)
    assert note.id > 0

    fetched = repository.get_note(note.id)
    assert fetched.title == "Hello"
    assert fetched.raw_transcript == "raw text"

    updated = repository.update_note(note.id, processed_text="even cleaner")
    assert updated.processed_text == "even cleaner"

    notes = repository.list_notes(notebook_id=nb.id)
    assert len(notes) == 1
    assert notes[0].id == note.id

    repository.delete_note(note.id)
    assert repository.list_notes(notebook_id=nb.id) == []
