"""Prompt template registry, rendering, and structured-output parsing tests."""

from __future__ import annotations

import pytest

from app.llm.prompt_templates import (
    META_DELIMITER,
    TEMPLATES,
    UI_TASKS,
    parse_structured_output,
    render_template,
    split_streaming_chunk,
)


# ---------- registry --------------------------------------------------------


def test_required_templates_present():
    expected = {
        "clean",
        "summarize",
        "organize",
        "format_markdown",
        "extract_tasks",
        "generate_title",
        "structured_entry",
    }
    assert expected.issubset(TEMPLATES.keys())


def test_ui_tasks_are_all_registered():
    for task in UI_TASKS:
        assert task in TEMPLATES


# ---------- render ----------------------------------------------------------


def test_render_template_substitutes_transcript_and_meta_instruction():
    system, user = render_template("summarize", "Hello world")
    assert META_DELIMITER in system
    assert "title" in system.lower() and "tags" in system.lower()
    assert "Hello world" in user


def test_render_template_passes_through_optional_context():
    system, user = render_template(
        "clean",
        "raw transcript",
        title="Meeting notes",
        instruction="Keep it terse",
    )
    assert "Meeting notes" in user
    assert "Keep it terse" in user


def test_render_template_omits_blank_optional_context():
    _, user = render_template("clean", "raw", title="", instruction=None)
    assert "Current note title" not in user
    assert "Additional user instruction" not in user


def test_render_template_handles_braces_in_transcript():
    """Transcript text containing ``{`` / ``}`` must not crash a formatter."""
    transcript = "config = {key: value, list: [1, 2]}"
    _, user = render_template("clean", transcript)
    assert transcript in user


def test_render_unknown_template_raises():
    with pytest.raises(KeyError):
        render_template("nope", "x")


# ---------- parse_structured_output -----------------------------------------


def test_parse_structured_output_with_meta_block():
    raw = (
        "# My note\n\nbody text here\n\n"
        f"{META_DELIMITER}\n"
        '{"title": "My Title", "tags": ["meeting", "Q4"]}\n'
    )
    result = parse_structured_output(raw)
    assert result.markdown == "# My note\n\nbody text here"
    assert result.title == "My Title"
    assert result.tags == ("meeting", "q4")  # normalised lower-case
    assert META_DELIMITER not in result.markdown


def test_parse_structured_output_without_meta_falls_back_to_first_heading():
    raw = "# A Heading\n\nsome body"
    result = parse_structured_output(raw)
    assert result.title == "A Heading"
    assert result.tags == ()
    assert result.markdown == raw.strip()


def test_parse_structured_output_without_heading_uses_first_line():
    raw = "Just a sentence about something\n\nmore text"
    result = parse_structured_output(raw)
    assert result.title == "Just a sentence about something"


def test_parse_structured_output_strips_hash_prefix_from_tags():
    raw = (
        "body\n\n"
        f"{META_DELIMITER}\n"
        '{"title": "X", "tags": ["#one", "TWO", "  three  ", "one"]}'
    )
    result = parse_structured_output(raw)
    # ``#one`` → ``one``, dedup keeps insertion order, lower-cased.
    assert result.tags == ("one", "two", "three")


def test_parse_structured_output_handles_malformed_json():
    raw = (
        "body\n\n"
        f"{META_DELIMITER}\n"
        '{title: not json'
    )
    result = parse_structured_output(raw)
    assert result.title == "body"  # fallback to first line
    assert result.tags == ()


# ---------- split_streaming_chunk ------------------------------------------


def test_split_streaming_chunk_holds_back_potential_delimiter_prefix():
    """The streamer must not display partial delimiters."""
    visible, retained = split_streaming_chunk("hello ---NOTEKEEPER-MET")
    assert visible == "hello "
    assert retained == "---NOTEKEEPER-MET"


def test_split_streaming_chunk_drops_tail_once_delimiter_arrives():
    visible, retained = split_streaming_chunk(
        f"hello world{META_DELIMITER}\n{{\"title\": \"x\"}}"
    )
    assert visible == "hello world"
    assert retained == ""


def test_split_streaming_chunk_returns_full_buffer_when_no_match():
    """Plain text that can't be a delimiter prefix streams immediately."""
    visible, retained = split_streaming_chunk("hello world")
    assert visible == "hello world"
    assert retained == ""


def test_split_streaming_chunk_holds_only_minimal_dash_suffix():
    """A trailing ``-`` could be the start of ``---NOTEKEEPER-META---``."""
    visible, retained = split_streaming_chunk("body text-")
    assert visible == "body text"
    assert retained == "-"
