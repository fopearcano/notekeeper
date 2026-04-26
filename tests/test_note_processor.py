"""NoteProcessor: streaming flow + structured-output assembly."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from app.llm.base import LLMProvider, LLMResponse
from app.llm.prompt_templates import META_DELIMITER
from app.services.note_processor import (
    NoteProcessor,
    StreamDelta,
    StreamFinal,
)


class _FakeProvider(LLMProvider):
    provider_key = "fake"

    def __init__(self, deltas: list[str], *, models: list[str] | None = None):
        self.deltas = deltas
        self.models = models or []
        self.complete_calls = 0
        self.stream_calls = 0
        self.last_system: str | None = None
        self.last_user: str | None = None

    async def complete(self, *, system: str, user: str) -> LLMResponse:
        self.complete_calls += 1
        self.last_system, self.last_user = system, user
        return LLMResponse(text="".join(self.deltas), model="fake")

    async def stream_complete(
        self, *, system: str, user: str
    ) -> AsyncIterator[str]:
        self.stream_calls += 1
        self.last_system, self.last_user = system, user
        for d in self.deltas:
            yield d

    async def list_models(self) -> list[str]:
        return list(self.models)


def _model_response(body: str, *, title: str = "T", tags=("a",)) -> str:
    tag_array = "[" + ", ".join(f'"{t}"' for t in tags) + "]"
    return f"{body}\n{META_DELIMITER}\n{{\"title\": \"{title}\", \"tags\": {tag_array}}}\n"


# ---------- buffered ``process`` ---------------------------------------------


def test_process_returns_structured_output():
    raw = _model_response("# Hello\n\nbody", title="Hello", tags=("greeting",))
    processor = NoteProcessor(_FakeProvider([raw]))

    result = asyncio.run(processor.process(action="clean", transcript="hi"))

    assert "Hello" in result.markdown
    assert META_DELIMITER not in result.markdown
    assert result.title == "Hello"
    assert result.tags == ("greeting",)


def test_process_rejects_empty_transcript():
    processor = NoteProcessor(_FakeProvider(["x"]))
    with pytest.raises(ValueError):
        asyncio.run(processor.process(action="clean", transcript="   "))


# ---------- streaming -------------------------------------------------------


def _drain_stream(processor: NoteProcessor, *, action="clean", transcript="hi"):
    async def _run():
        deltas: list[str] = []
        finals: list[StreamFinal] = []
        async for ev in processor.process_streaming(action=action, transcript=transcript):
            if isinstance(ev, StreamDelta):
                deltas.append(ev.text)
            elif isinstance(ev, StreamFinal):
                finals.append(ev)
        return deltas, finals

    return asyncio.run(_run())


def test_streaming_yields_deltas_then_one_final():
    # Body ends without a trailing newline so the streamed visible output
    # is exactly the body (no `\n` between body and delimiter to leak).
    raw = "Hello world" + f"\n{META_DELIMITER}\n" + '{"title": "Hello", "tags": ["x"]}'
    # Split into many small deltas so the streamer has to handle the meta
    # delimiter arriving across token boundaries.
    chunked = [raw[i: i + 3] for i in range(0, len(raw), 3)]
    processor = NoteProcessor(_FakeProvider(chunked))

    deltas, finals = _drain_stream(processor)

    visible = "".join(deltas)
    # The meta delimiter and JSON tail must NEVER reach the visible stream.
    assert META_DELIMITER not in visible
    assert "tags" not in visible.lower()
    # Body content is fully streamed.
    assert visible.rstrip() == "Hello world"

    assert len(finals) == 1
    output = finals[0].output
    assert output.title == "Hello"
    assert output.tags == ("x",)


def test_streaming_handles_response_without_meta_block():
    """If the model omits the footer, the visible stream is the full text."""
    chunked = ["Hel", "lo ", "world"]
    processor = NoteProcessor(_FakeProvider(chunked))

    deltas, finals = _drain_stream(processor)

    assert "".join(deltas) == "Hello world"
    assert len(finals) == 1
    assert finals[0].output.title == "Hello world"


def test_streaming_passes_title_and_instruction_into_prompt():
    raw = _model_response("body")
    fake = _FakeProvider([raw])
    processor = NoteProcessor(fake)

    async def _run():
        async for _ in processor.process_streaming(
            action="clean",
            transcript="raw",
            title="My note",
            instruction="be terse",
        ):
            pass

    asyncio.run(_run())
    assert "My note" in fake.last_user
    assert "be terse" in fake.last_user


def test_default_stream_complete_buffers_complete_for_non_streaming_provider():
    """A provider that only implements ``complete`` still streams via the default fallback."""
    raw = (
        "buffered body"
        + f"\n{META_DELIMITER}\n"
        + '{"title": "T", "tags": []}'
    )

    class _BufferedOnly(LLMProvider):
        provider_key = "buffered"

        async def complete(self, *, system, user):  # type: ignore[override]
            return LLMResponse(text=raw, model="buffered")

    processor = NoteProcessor(_BufferedOnly())
    deltas, finals = _drain_stream(processor)

    assert "".join(deltas).rstrip() == "buffered body"
    assert finals[0].output.markdown == "buffered body"
    assert finals[0].output.title == "T"


def test_list_models_proxies_provider():
    fake = _FakeProvider(["x"], models=["a", "b"])
    processor = NoteProcessor(fake)
    assert asyncio.run(processor.list_models()) == ["a", "b"]
