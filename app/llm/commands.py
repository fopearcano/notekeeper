"""Slash-command system for note processing.

Maps user-typed strings like ``/summarize`` or ``/rewrite clear`` to a
prompt template name plus an optional pre-baked instruction. The UI
combines that resolved instruction with whatever extra text the user typed
into the "Custom instruction" field, then hands the pair to
:meth:`SessionManager.stream_action`.

Parsing rules:
* Commands begin with ``/``.
* The verb is the first whitespace-delimited token after the slash.
* Everything after the first whitespace is the argument (preserved
  verbatim, not split further). Arguments are optional unless a command
  declares ``arg_required``.

The registry is intentionally small and explicit so the help text stays
discoverable. Adding a new command means: pick a prompt template, decide
whether the verb takes an argument, and append a :class:`Command` row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


__all__ = [
    "Command",
    "ResolvedCommand",
    "CommandError",
    "COMMANDS",
    "command_help",
    "parse_command",
    "resolve_command",
    "is_command",
]


class CommandError(ValueError):
    """Raised when a command can't be parsed or resolved."""


@dataclass(frozen=True)
class Command:
    verb: str                       # ``"/<verb>"`` (without the slash)
    template: str                   # name of a registered PromptTemplate
    description: str
    arg_required: bool = False
    arg_choices: tuple[str, ...] = ()  # optional whitelist (e.g. rewrite styles)
    #: Optional transform from ``arg`` to the auto-instruction the template
    #: gets. The user's ad-hoc instruction is appended after this.
    arg_to_instruction: Optional[Callable[[str], str]] = field(default=None)


@dataclass(frozen=True)
class ResolvedCommand:
    """The output of :func:`resolve_command`.

    ``instruction`` is the prompt-side instruction the registry built from
    the verb's argument; the UI may append the user's ad-hoc instruction
    on top of this with :func:`merge_instructions`.
    """

    verb: str
    template: str
    arg: str = ""
    instruction: str = ""


def _rewrite_instruction(style: str) -> str:
    return f"Rewrite the text in a {style.strip()} style."


# Order is the order shown in :func:`command_help`.
COMMANDS: dict[str, Command] = {
    cmd.verb: cmd
    for cmd in (
        Command(
            verb="clean",
            template="clean",
            description="Clean up filler words and punctuation.",
        ),
        Command(
            verb="summarize",
            template="summarize",
            description="Summarize into bullet points grouped by topic.",
        ),
        Command(
            verb="organize",
            template="organize",
            description="Rewrite as a structured outline with headings.",
        ),
        Command(
            verb="markdown",
            template="format_markdown",
            description="Format as readable markdown.",
        ),
        Command(
            verb="tasks",
            template="extract_tasks",
            description="Extract action items as a checklist.",
        ),
        Command(
            verb="title",
            template="generate_title",
            description="Generate a short descriptive title.",
        ),
        Command(
            verb="tags",
            template="tags",
            description="Suggest 3–7 relevant tags.",
        ),
        Command(
            verb="outline",
            template="outline",
            description="Produce a hierarchical markdown outline.",
        ),
        Command(
            verb="keypoints",
            template="keypoints",
            description="Extract 5–10 stand-alone key points.",
        ),
        Command(
            verb="rewrite",
            template="rewrite",
            description="Rewrite in a given style: clear / elegant / technical / cinematic.",
            arg_required=True,
            arg_choices=("clear", "elegant", "technical", "cinematic"),
            arg_to_instruction=_rewrite_instruction,
        ),
    )
}


_VERB_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def is_command(text: str) -> bool:
    """Return ``True`` when ``text`` looks like a slash command."""
    return bool(text) and text.lstrip().startswith("/")


def parse_command(text: str) -> tuple[str, str]:
    """Split ``"/verb arg…"`` into ``(verb, arg)``.

    Raises :class:`CommandError` for empty or malformed input.
    """
    raw = (text or "").strip()
    if not raw.startswith("/"):
        raise CommandError(f"Commands must start with '/': {raw!r}")
    body = raw[1:].strip()
    if not body:
        raise CommandError("Empty command")
    parts = body.split(maxsplit=1)
    verb = parts[0].lower()
    if not _VERB_RE.match(verb):
        raise CommandError(f"Invalid command verb: {verb!r}")
    arg = parts[1].strip() if len(parts) > 1 else ""
    return verb, arg


def resolve_command(text: str) -> ResolvedCommand:
    """Parse + look up a command. Raises :class:`CommandError` on failure."""
    verb, arg = parse_command(text)
    cmd = COMMANDS.get(verb)
    if cmd is None:
        known = ", ".join(f"/{name}" for name in COMMANDS)
        raise CommandError(f"Unknown command /{verb}. Known: {known}")

    if cmd.arg_required and not arg:
        choices = ""
        if cmd.arg_choices:
            choices = " (" + " / ".join(cmd.arg_choices) + ")"
        raise CommandError(f"/{verb} requires an argument{choices}")

    if cmd.arg_choices and arg and arg not in cmd.arg_choices:
        choices = " / ".join(cmd.arg_choices)
        raise CommandError(
            f"/{verb} argument must be one of: {choices} (got {arg!r})"
        )

    instruction = ""
    if arg and cmd.arg_to_instruction is not None:
        instruction = cmd.arg_to_instruction(arg)

    return ResolvedCommand(
        verb=verb, template=cmd.template, arg=arg, instruction=instruction
    )


def merge_instructions(*parts: Optional[str]) -> str:
    """Join non-empty instruction fragments with a blank line between them."""
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(cleaned)


def command_help() -> str:
    """Return a short multi-line help block listing every command."""
    lines = []
    for verb, cmd in COMMANDS.items():
        prefix = f"/{verb}"
        if cmd.arg_required and cmd.arg_choices:
            prefix += " <" + "|".join(cmd.arg_choices) + ">"
        elif cmd.arg_required:
            prefix += " <arg>"
        lines.append(f"{prefix:<32} {cmd.description}")
    return "\n".join(lines)
