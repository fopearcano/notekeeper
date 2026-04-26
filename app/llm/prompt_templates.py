"""Prompt templates used by the note processor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user_template: str  # uses ``str.format`` with a ``{transcript}`` placeholder


SUMMARIZE = PromptTemplate(
    name="summarize",
    system=(
        "You are a meticulous note-taking assistant. Summarize the user's "
        "transcript into a concise overview that preserves every concrete "
        "fact, decision, action item, and open question."
    ),
    user_template=(
        "Summarize the following transcript. Use short bullet points; group "
        "related items under headings if helpful.\n\nTranscript:\n{transcript}"
    ),
)

ORGANIZE = PromptTemplate(
    name="organize",
    system=(
        "You are an editor that turns rough spoken notes into a tidy "
        "outline. Preserve the speaker's intent and wording where it matters."
    ),
    user_template=(
        "Reorganize the following transcript into a structured outline with "
        "clear headings, sub-points, and any obvious action items separated "
        "into their own section.\n\nTranscript:\n{transcript}"
    ),
)

FORMAT = PromptTemplate(
    name="format",
    system=(
        "You are a copy editor. Produce a clean, readable version of the "
        "user's transcript without changing meaning."
    ),
    user_template=(
        "Reformat the following transcript into well-punctuated paragraphs. "
        "Fix obvious speech-to-text artifacts (filler words, false starts, "
        "stutters) but keep the speaker's voice and content intact.\n\n"
        "Transcript:\n{transcript}"
    ),
)

CLEANUP = PromptTemplate(
    name="cleanup",
    system=(
        "You clean up raw speech-to-text output. Remove filler words, fix "
        "punctuation, and merge fragmented sentences without altering meaning."
    ),
    user_template=(
        "Clean up the following transcript. Return only the cleaned text.\n\n"
        "Transcript:\n{transcript}"
    ),
)

UNDERSTAND = PromptTemplate(
    name="understand",
    system=(
        "You are a thoughtful study partner. Explain the user's notes back "
        "to them in plain language, surface implicit assumptions, and call "
        "out anything that seems contradictory or under-specified."
    ),
    user_template=(
        "Help me understand the following transcript. Provide: (1) a plain-"
        "language explanation, (2) any implicit assumptions, (3) open "
        "questions worth answering.\n\nTranscript:\n{transcript}"
    ),
)

TEMPLATES: dict[str, PromptTemplate] = {
    t.name: t for t in (SUMMARIZE, ORGANIZE, FORMAT, CLEANUP, UNDERSTAND)
}
# Aliases — ``llm.default_task = "clean"`` from default_config.toml maps here.
TEMPLATES["clean"] = CLEANUP


def render_template(name: str, transcript: str) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for the named template."""
    try:
        template = TEMPLATES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown prompt template {name!r}. Known: {sorted(TEMPLATES)}"
        ) from exc
    return template.system, template.user_template.format(transcript=transcript)
