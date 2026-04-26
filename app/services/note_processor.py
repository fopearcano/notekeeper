"""LLM-backed transformations on transcripts.

Pairs prompt templates with an :class:`LLMProvider` and exposes both
buffered and streaming entry points. Streaming yields a sequence of
:class:`StreamEvent` items so the UI can render markdown deltas live and
still receive the parsed structured output (title, tags) at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Optional, Union

from app.llm.base import LLMProvider, LLMResponse
from app.llm.prompt_templates import (
    StructuredOutput,
    parse_structured_output,
    render_template,
    split_streaming_chunk,
)
from app.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class StreamDelta:
    """A user-visible markdown chunk arriving from the model."""

    text: str


@dataclass(frozen=True)
class StreamFinal:
    """Terminal event for a streaming run; contains the parsed result."""

    output: StructuredOutput


StreamEvent = Union[StreamDelta, StreamFinal]


class NoteProcessor:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def process(
        self,
        *,
        action: str,
        transcript: str,
        title: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> StructuredOutput:
        """Run ``action`` against ``transcript`` and return the parsed result."""
        if not transcript.strip():
            raise ValueError("Cannot process an empty transcript")

        system, user = render_template(
            action, transcript, title=title, instruction=instruction
        )
        log.info(
            "NoteProcessor running %s on %d chars (title=%s, instruction=%s)",
            action,
            len(transcript),
            bool(title),
            bool(instruction),
        )
        response: LLMResponse = await self.provider.complete(system=system, user=user)
        return parse_structured_output(response.text)

    async def process_streaming(
        self,
        *,
        action: str,
        transcript: str,
        title: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield :class:`StreamDelta` events, then a single :class:`StreamFinal`.

        Listeners that don't care about structure can ignore everything
        except :class:`StreamDelta`. Listeners that need the title/tags read
        them from the trailing :class:`StreamFinal`.
        """
        if not transcript.strip():
            raise ValueError("Cannot process an empty transcript")

        system, user = render_template(
            action, transcript, title=title, instruction=instruction
        )
        log.info(
            "NoteProcessor streaming %s on %d chars",
            action,
            len(transcript),
        )

        from app.llm.prompt_templates import META_DELIMITER

        accumulated = ""
        held_back = ""
        past_delimiter = False
        async for delta in self.provider.stream_complete(system=system, user=user):
            if not delta:
                continue
            accumulated += delta
            if past_delimiter:
                # Everything after the delimiter is metadata; never display it
                # but keep it in ``accumulated`` so the parser can read it.
                continue
            held_back += delta
            visible, held_back = split_streaming_chunk(held_back)
            if visible:
                yield StreamDelta(text=visible)
            if META_DELIMITER in accumulated:
                past_delimiter = True
                held_back = ""

        # Below the holdback threshold and never crossed the delimiter →
        # emit whatever's still buffered.
        if held_back and not past_delimiter:
            yield StreamDelta(text=held_back)

        yield StreamFinal(output=parse_structured_output(accumulated))

    async def list_models(self) -> list[str]:
        return await self.provider.list_models()

    async def aclose(self) -> None:
        await self.provider.aclose()
