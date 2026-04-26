"""Prompt templates for the note-processing tasks.

Every task takes a transcript plus optional context (current note title, ad-hoc
user instruction) and asks the model to return:

1. A markdown body — what the user sees streamed into the processed-note
   panel.
2. A trailing metadata footer with a suggested title and tags. The footer is
   delimited by :data:`META_DELIMITER` so a streaming consumer can show the
   markdown live and post-process the JSON tail when the stream completes.

The footer format is deliberately tiny so weak local models hit it reliably:

    ---NOTEKEEPER-META---
    {"title": "...", "tags": ["..."]}

If the model omits or mangles the footer the parser falls back to using the
markdown's first heading (or first line) as the title and an empty tag list,
so callers always get a usable :class:`StructuredOutput`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

META_DELIMITER = "---NOTEKEEPER-META---"

#: Common postscript appended to every system prompt so the model knows the
#: contract regardless of which task it's running.
_FOOTER_INSTRUCTION = (
    "Output the markdown body of your response first. After the markdown, on "
    f"its own line, output exactly:\n\n{META_DELIMITER}\n\nFollowed by a "
    "single-line JSON object with two keys: \"title\" (a short string, "
    "no markdown) and \"tags\" (an array of short lower-case strings, no "
    "leading '#'). Do not output anything after the JSON line."
)


@dataclass(frozen=True)
class PromptTemplate:
    """A single task's prompts.

    ``user_template`` is rendered with :meth:`render` rather than ``.format``
    so transcript text containing ``{`` / ``}`` characters doesn't crash the
    formatter.
    """

    name: str
    system: str
    user_template: str

    def render(
        self,
        *,
        transcript: str,
        title: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> tuple[str, str]:
        title_block = (
            f"\n\nCurrent note title: {title.strip()}"
            if title and title.strip()
            else ""
        )
        instruction_block = (
            f"\n\nAdditional user instruction: {instruction.strip()}"
            if instruction and instruction.strip()
            else ""
        )
        # Use re.sub instead of .format to avoid touching ``{}`` in the transcript.
        body = self.user_template
        body = body.replace("{title_block}", title_block)
        body = body.replace("{instruction_block}", instruction_block)
        body = body.replace("{transcript}", transcript)
        system_prompt = f"{self.system}\n\n{_FOOTER_INSTRUCTION}"
        return system_prompt, body


# --------------------------------------------------------------------------- #
# Tasks                                                                       #
# --------------------------------------------------------------------------- #


CLEAN = PromptTemplate(
    name="clean",
    system=(
        "You clean up raw speech-to-text output. Remove filler words, fix "
        "punctuation, and merge fragmented sentences without altering meaning. "
        "Preserve the speaker's voice and intent."
    ),
    user_template=(
        "Clean up the following transcript. Return the cleaned text as "
        "well-punctuated paragraphs.{title_block}{instruction_block}\n\n"
        "Transcript:\n{transcript}"
    ),
)


SUMMARIZE = PromptTemplate(
    name="summarize",
    system=(
        "You are a meticulous note-taking assistant. Summarize the user's "
        "transcript into a concise overview that preserves every concrete "
        "fact, decision, action item, and open question."
    ),
    user_template=(
        "Summarize the following transcript. Use short bullet points; group "
        "related items under markdown headings when helpful.{title_block}"
        "{instruction_block}\n\nTranscript:\n{transcript}"
    ),
)


ORGANIZE = PromptTemplate(
    name="organize",
    system=(
        "You turn rough spoken notes into a structured outline. Keep the "
        "speaker's intent and wording where it matters."
    ),
    user_template=(
        "Reorganize the following transcript into a markdown outline with "
        "clear headings, sub-points, and a separate section for any obvious "
        "action items.{title_block}{instruction_block}\n\nTranscript:\n"
        "{transcript}"
    ),
)


FORMAT_MARKDOWN = PromptTemplate(
    name="format_markdown",
    system=(
        "You are a markdown formatter. Convert the user's transcript into "
        "well-structured markdown without changing meaning. Use headings, "
        "bullet lists, numbered lists, code fences, and bold/italic emphasis "
        "where they help readability."
    ),
    user_template=(
        "Format the following transcript as readable markdown. Preserve every "
        "fact and the speaker's voice.{title_block}{instruction_block}\n\n"
        "Transcript:\n{transcript}"
    ),
)


EXTRACT_TASKS = PromptTemplate(
    name="extract_tasks",
    system=(
        "You are an action-item extractor. Read the user's transcript and "
        "produce a markdown checklist of every concrete task, decision, or "
        "follow-up. Preserve any owners, due dates, and dependencies that "
        "are stated. Do not invent tasks."
    ),
    user_template=(
        "Extract the action items from the following transcript as a markdown "
        "checklist (one ``- [ ]`` line per task). Group related items under "
        "headings when there are several distinct workstreams."
        "{title_block}{instruction_block}\n\nTranscript:\n{transcript}"
    ),
)


GENERATE_TITLE = PromptTemplate(
    name="generate_title",
    system=(
        "You generate a short, descriptive title for the user's note. The "
        "title should be 3–8 words, in title case, with no trailing "
        "punctuation."
    ),
    user_template=(
        "Suggest a title for the following transcript. The markdown body of "
        "your response should contain the title as a single ``#`` heading and "
        "nothing else; the JSON metadata should put the same title in the "
        "\"title\" field.{title_block}{instruction_block}\n\n"
        "Transcript:\n{transcript}"
    ),
)


STRUCTURED_ENTRY = PromptTemplate(
    name="structured_entry",
    system=(
        "You assemble a polished notebook entry from a raw transcript. The "
        "entry should be self-contained: open with a one-paragraph summary, "
        "then organized markdown sections, then a checklist of action items "
        "(if any), then a 'References' section if specific names, links, or "
        "documents were mentioned."
    ),
    user_template=(
        "Produce a structured notebook entry from the following transcript."
        "{title_block}{instruction_block}\n\nTranscript:\n{transcript}"
    ),
)


TAGS = PromptTemplate(
    name="tags",
    system=(
        "You extract a small set of relevant lower-case tags from the user's "
        "notes. Avoid generic tags like 'note', 'audio', or 'transcript'. "
        "Prefer specific topical tags over broad categories."
    ),
    user_template=(
        "Suggest 3 to 7 tags for the following transcript. The markdown body "
        "of your response should list them as a single comma-separated line; "
        "the JSON metadata must put the same tags in the \"tags\" array."
        "{title_block}{instruction_block}\n\nTranscript:\n{transcript}"
    ),
)


OUTLINE = PromptTemplate(
    name="outline",
    system=(
        "You produce a hierarchical outline of the user's content. Use "
        "markdown nested bullet lists; mirror the source's structure "
        "faithfully without inventing sections."
    ),
    user_template=(
        "Produce a markdown outline of the following transcript. Use nested "
        "bullets; preserve the source's order and intent."
        "{title_block}{instruction_block}\n\nTranscript:\n{transcript}"
    ),
)


KEYPOINTS = PromptTemplate(
    name="keypoints",
    system=(
        "You extract the most important key points from the user's content. "
        "Be concise — five to ten bullets. Each bullet stands alone."
    ),
    user_template=(
        "Extract the key points from the following transcript as a markdown "
        "bullet list. Each point should be one sentence and survive on its "
        "own without surrounding context."
        "{title_block}{instruction_block}\n\nTranscript:\n{transcript}"
    ),
)


REWRITE = PromptTemplate(
    name="rewrite",
    system=(
        "You rewrite the user's text in the requested style without changing "
        "its meaning. Preserve every concrete fact; only adjust voice, tone, "
        "and rhythm. Pass numbers, names, and quotations through unchanged."
    ),
    user_template=(
        "Rewrite the following transcript according to the user's "
        "instruction. Preserve all facts; only change the style."
        "{title_block}{instruction_block}\n\nTranscript:\n{transcript}"
    ),
)


TEMPLATES: dict[str, PromptTemplate] = {
    t.name: t for t in (
        CLEAN,
        SUMMARIZE,
        ORGANIZE,
        FORMAT_MARKDOWN,
        EXTRACT_TASKS,
        GENERATE_TITLE,
        STRUCTURED_ENTRY,
        TAGS,
        OUTLINE,
        KEYPOINTS,
        REWRITE,
    )
}

#: Stable list of task names exposed in the UI toolbar (in display order).
UI_TASKS: tuple[str, ...] = (
    "clean",
    "summarize",
    "organize",
    "format_markdown",
    "extract_tasks",
    "generate_title",
)


def render_template(
    name: str,
    transcript: str,
    *,
    title: Optional[str] = None,
    instruction: Optional[str] = None,
) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for the named task."""
    try:
        template = TEMPLATES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown prompt template {name!r}. Known: {sorted(TEMPLATES)}"
        ) from exc
    return template.render(transcript=transcript, title=title, instruction=instruction)


# --------------------------------------------------------------------------- #
# Output parsing                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StructuredOutput:
    """Parsed task result.

    ``markdown`` is what the UI renders; ``title`` and ``tags`` come from the
    metadata footer (with safe fallbacks if the model omitted it).

    ``title_explicit`` is ``True`` when the model produced a non-empty
    ``title`` field in the JSON footer, and ``False`` when ``title`` came
    from the fallback (first heading / first line). Persistence layers that
    don't want to overwrite a saved note's title should consult this flag.
    """

    markdown: str
    title: str = ""
    tags: tuple[str, ...] = ()
    raw: str = ""
    title_explicit: bool = False


_FIRST_HEADING = re.compile(r"^\s*#+\s*(.+?)\s*$", re.MULTILINE)


def _fallback_title(markdown: str) -> str:
    match = _FIRST_HEADING.search(markdown)
    if match:
        return match.group(1).strip()
    for line in markdown.splitlines():
        stripped = line.strip().lstrip("-* >").strip()
        if stripped:
            return stripped[:80]
    return ""


def _normalise_tags(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            tag = item.strip().lstrip("#").strip().lower()
            if tag and tag not in out:
                out.append(tag)
    return tuple(out)


def parse_structured_output(text: str) -> StructuredOutput:
    """Split a model response into ``(markdown, title, tags)``.

    The response should look like::

        ... markdown ...
        ---NOTEKEEPER-META---
        {"title": "...", "tags": ["..."]}

    Anything after :data:`META_DELIMITER` that parses as JSON populates the
    title/tags fields. If no delimiter is present (or the JSON is malformed)
    we fall back to using the entire text as markdown and deriving a title
    from its first heading or first non-empty line.
    """
    raw = text or ""
    body = raw
    title = ""
    tags: tuple[str, ...] = ()
    title_explicit = False

    if META_DELIMITER in raw:
        body, _, tail = raw.partition(META_DELIMITER)
        body = body.rstrip()
        # The model may have wrapped the JSON in fences or added stray text;
        # take the first {...} balanced object on the tail.
        match = re.search(r"\{.*\}", tail, flags=re.DOTALL)
        if match is not None:
            try:
                meta = json.loads(match.group(0))
            except json.JSONDecodeError:
                meta = {}
            if isinstance(meta, dict):
                meta_title = meta.get("title")
                if isinstance(meta_title, str) and meta_title.strip():
                    title = meta_title.strip()
                    title_explicit = True
                tags = _normalise_tags(meta.get("tags"))

    if not title:
        title = _fallback_title(body)

    return StructuredOutput(
        markdown=body.strip(),
        title=title,
        tags=tags,
        raw=raw,
        title_explicit=title_explicit,
    )


def split_streaming_chunk(buffer: str) -> tuple[str, str]:
    """Return ``(visible, retained)`` so streamers don't show the meta tail.

    The streamer maintains a rolling buffer; on each chunk it calls this
    with the buffer's current contents.

    * If the full delimiter is in the buffer, everything before it is visible
      and the rest is dropped.
    * Otherwise, hold back the longest suffix that is itself a prefix of
      :data:`META_DELIMITER` (so a delimiter arriving across token
      boundaries is still detected). Anything else streams through.
    """
    if META_DELIMITER in buffer:
        head, _, _ = buffer.partition(META_DELIMITER)
        return head, ""

    # Walk back from the longest possible partial-delimiter match.
    max_k = min(len(buffer), len(META_DELIMITER) - 1)
    for k in range(max_k, 0, -1):
        if buffer.endswith(META_DELIMITER[:k]):
            return buffer[:-k], buffer[-k:]
    return buffer, ""
