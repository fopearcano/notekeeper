"""Slash-command parser and registry."""

from __future__ import annotations

import pytest

from app.llm import TEMPLATES
from app.llm.commands import (
    COMMANDS,
    CommandError,
    command_help,
    is_command,
    merge_instructions,
    parse_command,
    resolve_command,
)


# ---------- predicate ------------------------------------------------------


@pytest.mark.parametrize("text", ["/clean", "  /summarize", "/rewrite clear"])
def test_is_command_recognises_slash_prefix(text):
    assert is_command(text) is True


@pytest.mark.parametrize("text", ["", "clean", "  ", "?summarize"])
def test_is_command_rejects_non_slash(text):
    assert is_command(text) is False


# ---------- parser ---------------------------------------------------------


def test_parse_command_basic():
    assert parse_command("/clean") == ("clean", "")


def test_parse_command_with_argument():
    assert parse_command("/rewrite clear") == ("rewrite", "clear")


def test_parse_command_argument_preserved_verbatim():
    """Multi-word arguments are kept as one string, not re-split."""
    assert parse_command("/rewrite somewhat clear and elegant") == (
        "rewrite",
        "somewhat clear and elegant",
    )


def test_parse_command_lowercases_verb():
    assert parse_command("/Summarize")[0] == "summarize"


def test_parse_command_strips_whitespace():
    assert parse_command("   /summarize   ") == ("summarize", "")


@pytest.mark.parametrize("bad", ["", "   ", "summarize", "/", "/   "])
def test_parse_command_rejects_garbage(bad):
    with pytest.raises(CommandError):
        parse_command(bad)


def test_parse_command_rejects_invalid_verb():
    with pytest.raises(CommandError):
        parse_command("/123start")


# ---------- registry coverage ----------------------------------------------


REQUIRED_VERBS = {
    "clean",
    "summarize",
    "organize",
    "markdown",
    "tasks",
    "title",
    "tags",
    "outline",
    "keypoints",
    "rewrite",
}


def test_all_required_commands_registered():
    assert REQUIRED_VERBS.issubset(COMMANDS.keys())


def test_every_command_template_exists():
    for verb, cmd in COMMANDS.items():
        assert cmd.template in TEMPLATES, f"/{verb} → unknown template {cmd.template!r}"


def test_command_help_lists_every_verb():
    text = command_help()
    for verb in COMMANDS:
        assert f"/{verb}" in text


# ---------- resolution -----------------------------------------------------


def test_resolve_command_basic():
    resolved = resolve_command("/summarize")
    assert resolved.verb == "summarize"
    assert resolved.template == "summarize"
    assert resolved.arg == ""
    assert resolved.instruction == ""


def test_resolve_command_unknown_verb_lists_known():
    with pytest.raises(CommandError) as exc_info:
        resolve_command("/nope")
    msg = str(exc_info.value)
    assert "/nope" in msg
    assert "/summarize" in msg  # known commands listed for discoverability


@pytest.mark.parametrize("style", ["clear", "elegant", "technical", "cinematic"])
def test_resolve_rewrite_with_style(style):
    resolved = resolve_command(f"/rewrite {style}")
    assert resolved.template == "rewrite"
    assert resolved.arg == style
    assert style in resolved.instruction.lower()


def test_resolve_rewrite_requires_style():
    with pytest.raises(CommandError) as exc_info:
        resolve_command("/rewrite")
    msg = str(exc_info.value)
    assert "rewrite" in msg.lower()
    assert "clear" in msg  # whitelisted choices surfaced in the error


def test_resolve_rewrite_rejects_unknown_style():
    with pytest.raises(CommandError) as exc_info:
        resolve_command("/rewrite robotic")
    msg = str(exc_info.value)
    assert "robotic" in msg
    assert "clear" in msg  # whitelisted choices surfaced


def test_resolve_command_maps_aliases_to_correct_template():
    """Verbs and templates aren't always 1:1 — sanity-check the mapping."""
    pairs = {
        "clean": "clean",
        "markdown": "format_markdown",
        "tasks": "extract_tasks",
        "title": "generate_title",
        "keypoints": "keypoints",
        "outline": "outline",
        "tags": "tags",
    }
    for verb, expected in pairs.items():
        assert resolve_command(f"/{verb}").template == expected


# ---------- instruction merging --------------------------------------------


def test_merge_instructions_filters_empties():
    assert merge_instructions("a", "", None, "  ", "b") == "a\n\nb"


def test_merge_instructions_returns_empty_when_all_blank():
    assert merge_instructions("", None, "  ") == ""


def test_merge_instructions_strips_each_part():
    assert merge_instructions("  hello  ", " world ") == "hello\n\nworld"
