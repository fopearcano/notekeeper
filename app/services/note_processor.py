"""Runs LLM-backed transformations on transcripts."""

from __future__ import annotations

from app.llm.base import LLMProvider, LLMResponse
from app.llm.prompt_templates import render_template
from app.utils.logging import get_logger

log = get_logger(__name__)


class NoteProcessor:
    """Thin wrapper that pairs prompt templates with an LLM provider."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def process(self, *, action: str, transcript: str) -> LLMResponse:
        if not transcript.strip():
            raise ValueError("Cannot process an empty transcript")
        system, user = render_template(action, transcript)
        log.info("NoteProcessor running %s on %d chars", action, len(transcript))
        return await self.provider.complete(system=system, user=user)

    async def summarize(self, transcript: str) -> LLMResponse:
        return await self.process(action="summarize", transcript=transcript)

    async def organize(self, transcript: str) -> LLMResponse:
        return await self.process(action="organize", transcript=transcript)

    async def format(self, transcript: str) -> LLMResponse:
        return await self.process(action="format", transcript=transcript)

    async def cleanup(self, transcript: str) -> LLMResponse:
        return await self.process(action="cleanup", transcript=transcript)

    async def understand(self, transcript: str) -> LLMResponse:
        return await self.process(action="understand", transcript=transcript)

    async def aclose(self) -> None:
        await self.provider.aclose()
