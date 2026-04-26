"""SessionManager: save/load, autosave, LLM-run persistence side effects."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from app.config.settings import load_settings
from app.llm.base import LLMProvider, LLMResponse
from app.llm.prompt_templates import META_DELIMITER
from app.notes.database import Database
from app.notes.repository import NoteRepository
from app.services.session_manager import SessionManager
from app.transcription.base import TranscriptSegment


class _FakeLLM(LLMProvider):
    provider_key = "fake"

    def __init__(self, body: str = "# Cleaned\n\nbody text", title="Cleaned",
                 tags=("alpha",)):
        tag_array = "[" + ", ".join(f'"{t}"' for t in tags) + "]"
        self.body = (
            f"{body}\n{META_DELIMITER}\n"
            f'{{"title": "{title}", "tags": {tag_array}}}'
        )
        # Mimic an LM-Studio-shaped settings object so SessionManager can
        # read ``settings.model`` when persisting the run.
        self.settings = type("S", (), {"model": "fake-model"})()

    async def complete(self, *, system, user) -> LLMResponse:
        return LLMResponse(text=self.body, model="fake-model")

    async def stream_complete(self, *, system, user) -> AsyncIterator[str]:
        # Stream in small chunks so the streamer goes through its real path.
        for i in range(0, len(self.body), 5):
            yield self.body[i: i + 5]


@pytest.fixture
def session(monkeypatch):
    """Build a real SessionManager pointed at an in-memory SQLite repo."""
    settings = load_settings(bootstrap=False)

    db = Database(":memory:")
    repo = NoteRepository(db)

    # Pre-build the session against a fake LLM so we don't try to open httpx.
    sm = SessionManager(settings, repo)
    sm.llm_provider = _FakeLLM()
    sm.note_processor.provider = sm.llm_provider

    yield sm

    db.close()


def _push_segments(sm: SessionManager, segments: list[TranscriptSegment]) -> None:
    """Pretend the pipeline emitted these segments."""
    for s in segments:
        sm._on_segment(s)


def _seg(text: str, *, start: float = 0.0, end: float = 1.0,
         language: str = "en", confidence: float = 0.91) -> TranscriptSegment:
    return TranscriptSegment(
        text=text, is_final=True, start_s=start, end_s=end,
        metadata={"language": language, "language_probability": confidence},
    )


# ---------- manual save / load ---------------------------------------------


def test_save_note_persists_segments_and_language(session):
    _push_segments(session, [
        _seg("hello world", start=0.0, end=1.0),
        _seg("more text", start=1.0, end=2.0),
    ])

    note = session.save_note(title="My note", processed_text="(empty)",
                             tags=["meeting"])

    assert note.id > 0
    assert note.title == "My note"
    assert note.raw_transcript == "hello world more text"
    assert note.language == "en"
    assert note.tags == ["meeting"]

    persisted_segments = session.repository.list_segments(note.id)
    assert [s.text for s in persisted_segments] == ["hello world", "more text"]
    assert persisted_segments[0].confidence == pytest.approx(0.91)


def test_save_again_updates_in_place(session):
    _push_segments(session, [_seg("first chunk")])
    first = session.save_note(title="Draft")

    # Push only the new segment — the session keeps a running list internally.
    _push_segments(session, [_seg("second chunk")])
    second = session.save_note(title="Draft")

    assert second.id == first.id
    assert second.raw_transcript == "first chunk second chunk"
    # Replace-all semantics on segments: there should be exactly two now.
    assert len(session.repository.list_segments(second.id)) == 2


def test_load_note_then_save_does_not_wipe_raw_transcript(session):
    """Saving after loading (no live segments) must preserve the loaded text."""
    _push_segments(session, [_seg("original transcript")])
    saved = session.save_note(title="A")
    note_id = saved.id
    saved_raw = saved.raw_transcript

    # Simulate a fresh session opening the same note.
    session.clear_transcript()
    loaded = session.load_note(note_id)
    assert loaded.raw_transcript == saved_raw

    # Manual save with no live segments — only metadata fields move.
    re_saved = session.save_note(
        title="A renamed", processed_text="cleaned body", tags=["x"]
    )
    assert re_saved.id == note_id
    assert re_saved.raw_transcript == saved_raw  # untouched
    assert re_saved.title == "A renamed"
    assert re_saved.processed_text == "cleaned body"
    assert re_saved.tags == ["x"]


# ---------- autosave --------------------------------------------------------


def test_autosave_skips_when_nothing_new(session):
    _push_segments(session, [_seg("hi")])
    first = session.autosave()
    assert first is not None
    second = session.autosave()
    assert second is None  # no new segments → no work


def test_autosave_persists_after_new_segments(session):
    _push_segments(session, [_seg("first")])
    first = session.autosave()
    assert first is not None and first.raw_transcript == "first"

    # Only push the new chunk — the session keeps the running list internally.
    _push_segments(session, [_seg("second")])
    second = session.autosave()
    assert second is not None
    assert second.id == first.id  # same note
    assert second.raw_transcript == "first second"


def test_autosave_no_op_when_no_segments(session):
    assert session.autosave() is None


# ---------- LLM run persistence --------------------------------------------


def _drain_stream(session, action):
    async def _run():
        finals = []
        async for ev in session.stream_action(action):
            from app.services.note_processor import StreamFinal

            if isinstance(ev, StreamFinal):
                finals.append(ev)
        return finals

    return asyncio.run(_run())


def test_stream_action_records_processing_run_and_updates_note(session):
    _push_segments(session, [_seg("hello there", start=0.0, end=1.0)])
    saved = session.save_note(title="Original")
    note_id = saved.id

    finals = _drain_stream(session, "clean")
    assert len(finals) == 1

    runs = session.repository.list_processing_runs(note_id)
    assert len(runs) == 1
    assert runs[0].provider == "fake"
    assert runs[0].model == "fake-model"
    assert runs[0].task == "clean"
    assert "hello there" in runs[0].prompt
    # The output column carries the full raw model response (incl. the meta tail).
    assert META_DELIMITER in runs[0].output

    # Note row picked up the parsed structured output.
    refreshed = session.repository.get_note(note_id)
    assert refreshed.title == "Cleaned"
    assert refreshed.tags == ["alpha"]
    assert refreshed.processed_text.strip().startswith("# Cleaned")


def test_stream_action_auto_creates_note_when_unsaved(session):
    """Running an LLM task before saving auto-creates the note."""
    _push_segments(session, [_seg("just a transcript")])
    assert session.current_note is None

    _drain_stream(session, "summarize")

    assert session.current_note is not None
    runs = session.repository.list_processing_runs(session.current_note.id)
    assert len(runs) == 1
    assert runs[0].task == "summarize"


def test_stream_action_with_text_uses_override_not_transcript(session):
    """The command bar passes ``text=...`` to run an action on a selection
    instead of the full live transcript."""
    _push_segments(session, [_seg("FULL TRANSCRIPT TEXT")])
    saved = session.save_note(title="Note")

    captured_prompts: list[str] = []
    # ``stream_action`` uses ``stream_complete``; capture from that path.
    original_stream = session.llm_provider.stream_complete

    async def _spy(*, system, user):
        captured_prompts.append(user)
        async for delta in original_stream(system=system, user=user):
            yield delta

    session.llm_provider.stream_complete = _spy
    session.note_processor.provider = session.llm_provider

    finals = _drain_stream_with_text(session, "clean", text="just this slice")
    assert finals
    # The model saw the override, not the full transcript.
    assert any("just this slice" in p for p in captured_prompts)
    assert not any("FULL TRANSCRIPT TEXT" in p for p in captured_prompts)

    # Processing-run row records the override text as the prompt source.
    runs = session.repository.list_processing_runs(saved.id)
    assert any("just this slice" in r.prompt for r in runs)


def _drain_stream_with_text(session, action, *, text):
    async def _run():
        finals = []
        async for ev in session.stream_action(action, text=text):
            from app.services.note_processor import StreamFinal

            if isinstance(ev, StreamFinal):
                finals.append(ev)
        return finals

    return asyncio.run(_run())


def test_processing_runs_returns_newest_first(session):
    _push_segments(session, [_seg("body")])
    session.save_note(title="N")
    _drain_stream(session, "clean")
    _drain_stream(session, "summarize")

    runs = session.processing_runs()
    assert [r.task for r in runs] == ["summarize", "clean"]


def test_processing_runs_empty_without_current_note(session):
    assert session.processing_runs() == []


def test_stream_action_preserves_title_when_model_returns_empty_title(session):
    """A task that doesn't generate a title (e.g. ``clean`` with empty meta)
    must not blow away the existing title."""
    _push_segments(session, [_seg("body")])
    saved = session.save_note(title="Keep Me")

    # Override fake LLM to emit a structured response with empty title/tags.
    session.llm_provider = _FakeLLM(body="cleaned body", title="", tags=())
    session.note_processor.provider = session.llm_provider

    _drain_stream(session, "clean")

    refreshed = session.repository.get_note(saved.id)
    assert refreshed.title == "Keep Me"
