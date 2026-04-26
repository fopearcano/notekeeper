"""Note exporters + ``safe_filename``."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.notes.models import Note, ProcessingRun, Segment
from app.services.exporters import (
    ExportPayload,
    safe_filename,
    to_json,
    to_markdown,
    to_text,
    write_export,
)


# --------------------------------------------------------------------------- #
# safe_filename                                                                #
# --------------------------------------------------------------------------- #


def test_safe_filename_basic_passthrough():
    assert safe_filename("Notes for the team") == "Notes for the team"


def test_safe_filename_replaces_path_separators_and_specials():
    out = safe_filename('a/b\\c<d>e:f"g|h?i*j')
    assert "/" not in out and "\\" not in out
    for bad in '<>:"|?*':
        assert bad not in out
    assert out == "a-b-c-d-e-f-g-h-i-j"


def test_safe_filename_strips_control_chars():
    out = safe_filename("hello\x00world\x07")
    # Control bytes are dropped outright (not replaced with dashes) so the
    # readable text stays joined together.
    assert "\x00" not in out and "\x07" not in out
    assert out == "helloworld"


def test_safe_filename_collapses_whitespace_and_trims():
    assert safe_filename("  many   spaces \t here  ") == "many spaces here"


def test_safe_filename_strips_leading_and_trailing_dots():
    assert safe_filename(".hidden.") == "hidden"
    assert safe_filename("...") == "note"  # falls back


def test_safe_filename_truncates_long_titles():
    long_title = "x" * 500
    out = safe_filename(long_title, max_length=64)
    assert len(out) <= 64


def test_safe_filename_reserves_room_for_suffix():
    out = safe_filename("x" * 500, suffix=".md", max_length=20)
    assert out.endswith(".md")
    assert len(out) <= 20


def test_safe_filename_empty_uses_fallback():
    assert safe_filename("") == "note"
    assert safe_filename("???") == "note"  # all chars rewritten then stripped
    assert safe_filename("", fallback="my-export") == "my-export"


def test_safe_filename_handles_unicode_titles():
    """Unicode letters survive — they're allowed in modern filesystems."""
    out = safe_filename("会議メモ — 第3回")
    assert "会議メモ" in out
    assert "第3回" in out


def test_safe_filename_escapes_windows_reserved_names():
    for name in ("CON", "prn", "aux", "NUL", "COM1", "LPT9"):
        out = safe_filename(name)
        # Underscored prefix preserves the user's text but dodges Windows.
        assert out.startswith("_")


def test_safe_filename_reserved_with_extension_still_escaped():
    """``CON.md`` is still reserved on Windows."""
    assert safe_filename("CON.md") == "_CON.md"


def test_safe_filename_rejects_bad_args():
    with pytest.raises(TypeError):
        safe_filename(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        safe_filename("ok", max_length=0)


# --------------------------------------------------------------------------- #
# Test payload                                                                 #
# --------------------------------------------------------------------------- #


def _payload(*, with_runs: bool = True, with_segments: bool = True) -> ExportPayload:
    note = Note(
        id=1,
        title="Standup notes — 2026-04-26",
        raw_transcript="hi team\nstatus update",
        processed_text="# Standup\n\n- Updates\n- Decisions",
        created_at=datetime(2026, 4, 26, 9, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 26, 9, 5, 0, tzinfo=timezone.utc),
        source="recording",
        language="en",
        tags=["standup", "Q4"],
    )
    segments = (
        [
            Segment(
                id=10,
                note_id=1,
                start_time=0.0,
                end_time=2.0,
                text="hi team",
                confidence=0.91,
                created_at=datetime(2026, 4, 26, 9, 0, 1, tzinfo=timezone.utc),
            ),
            Segment(
                id=11,
                note_id=1,
                start_time=2.0,
                end_time=4.0,
                text="status update",
                confidence=None,
                created_at=datetime(2026, 4, 26, 9, 0, 3, tzinfo=timezone.utc),
            ),
        ]
        if with_segments
        else []
    )
    runs = (
        [
            ProcessingRun(
                id=100,
                note_id=1,
                provider="lmstudio",
                model="local-model",
                task="clean",
                prompt="SYSTEM:\n…\n\nUSER:\n…",
                output="cleaned body",
                created_at=datetime(2026, 4, 26, 9, 1, 0, tzinfo=timezone.utc),
            ),
            ProcessingRun(
                id=101,
                note_id=1,
                provider="lmstudio",
                model="local-model",
                task="summarize",
                prompt="SYSTEM:\n…\n\nUSER:\n…",
                output="# Summary\n\n- One",
                created_at=datetime(2026, 4, 26, 9, 2, 0, tzinfo=timezone.utc),
            ),
        ]
        if with_runs
        else []
    )
    return ExportPayload(note=note, segments=segments, processing_runs=runs)


# --------------------------------------------------------------------------- #
# to_markdown                                                                  #
# --------------------------------------------------------------------------- #


def test_markdown_has_yaml_frontmatter_and_sections():
    text = to_markdown(_payload())
    # Frontmatter delimited by '---' on its own line, twice.
    assert text.startswith("---\n")
    assert text.count("\n---\n") == 1
    # Required sections.
    assert "## Processed Note" in text
    assert "## Raw Transcript" in text
    assert "## Processing History" in text
    # Tag list is human-readable.
    assert "tags: [standup, Q4]" in text


def test_markdown_quotes_title_with_special_chars():
    payload = _payload()
    payload = ExportPayload(
        note=payload.note.model_copy(update={"title": 'with "quotes" and: colon'}),
        segments=payload.segments,
        processing_runs=payload.processing_runs,
    )
    text = to_markdown(payload)
    # The frontmatter scalar is YAML-quoted.
    assert 'title: "with \\"quotes\\" and: colon"' in text


def test_markdown_omits_history_when_disabled():
    text = to_markdown(_payload(), include_history=False)
    assert "## Processing History" not in text
    assert "summarize" not in text  # task name shouldn't leak


def test_markdown_no_history_when_no_runs():
    text = to_markdown(_payload(with_runs=False))
    assert "## Processing History" not in text


def test_markdown_handles_empty_processed_text():
    payload = _payload()
    blank = ExportPayload(
        note=payload.note.model_copy(update={"processed_text": ""}),
        segments=payload.segments,
        processing_runs=[],
    )
    text = to_markdown(blank, include_history=False)
    assert "_(empty)_" in text


# --------------------------------------------------------------------------- #
# to_text                                                                      #
# --------------------------------------------------------------------------- #


def test_text_export_is_plain_keys_then_sections():
    text = to_text(_payload())
    assert "Title: Standup notes — 2026-04-26" in text
    assert "=== Processed Note ===" in text
    assert "=== Raw Transcript ===" in text
    assert "=== Processing History ===" in text


def test_text_export_omits_optional_metadata_when_blank():
    payload = _payload()
    blank = ExportPayload(
        note=payload.note.model_copy(update={"language": None, "tags": []}),
        segments=payload.segments,
        processing_runs=[],
    )
    text = to_text(blank, include_history=False)
    assert "Language" not in text
    assert "Tags" not in text


# --------------------------------------------------------------------------- #
# to_json                                                                      #
# --------------------------------------------------------------------------- #


def test_json_export_is_valid_json_and_carries_every_field():
    text = to_json(_payload())
    parsed = json.loads(text)
    assert parsed["title"].startswith("Standup")
    assert parsed["tags"] == ["standup", "Q4"]
    assert parsed["language"] == "en"
    assert parsed["source"] == "recording"
    assert parsed["raw_transcript"].startswith("hi team")
    assert parsed["processed_text"].startswith("# Standup")
    assert len(parsed["segments"]) == 2
    assert parsed["segments"][0]["confidence"] == pytest.approx(0.91)
    assert parsed["segments"][1]["confidence"] is None
    assert len(parsed["processing_runs"]) == 2
    assert parsed["processing_runs"][0]["task"] == "clean"


def test_json_export_omits_history_when_disabled():
    parsed = json.loads(to_json(_payload(), include_history=False))
    assert "processing_runs" not in parsed


def test_json_export_preserves_unicode():
    payload = _payload()
    payload = ExportPayload(
        note=payload.note.model_copy(update={"title": "会議メモ"}),
        segments=payload.segments,
        processing_runs=payload.processing_runs,
    )
    text = to_json(payload)
    # Round-trip.
    assert json.loads(text)["title"] == "会議メモ"
    # And the raw bytes don't escape to \uXXXX.
    assert "会議メモ" in text


# --------------------------------------------------------------------------- #
# write_export                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt,suffix", [("md", ".md"), ("txt", ".txt"), ("json", ".json")])
def test_write_export_writes_to_disk(tmp_path: Path, fmt: str, suffix: str):
    target = tmp_path / f"out{suffix}"
    write_export(target, _payload(), fmt)
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert text.strip()  # non-empty
    if fmt == "json":
        assert json.loads(text)["title"].startswith("Standup")


def test_write_export_rejects_unknown_format(tmp_path: Path):
    with pytest.raises(ValueError):
        write_export(tmp_path / "x", _payload(), "html")


# --------------------------------------------------------------------------- #
# PDF                                                                          #
# --------------------------------------------------------------------------- #

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication(sys.argv)


def test_pdf_export_produces_pdf_signature(qapp, tmp_path: Path):
    """The output starts with ``%PDF-`` and contains rendered title text."""
    from app.services.exporters import to_pdf_bytes

    blob = to_pdf_bytes(_payload())
    assert blob[:5] == b"%PDF-"
    # Sanity: PDFs aren't tiny.
    assert len(blob) > 1024

    # round-trip via write_export
    target = tmp_path / "out.pdf"
    write_export(target, _payload(), "pdf")
    assert target.read_bytes()[:5] == b"%PDF-"


def test_pdf_export_without_qapplication_raises(monkeypatch):
    """Missing QApplication is a clear runtime error, not a segfault."""
    from app.services import exporters as ex

    # Pretend no QApplication exists for the duration of this test.
    real_instance = QApplication.instance
    monkeypatch.setattr(QApplication, "instance", classmethod(lambda cls: None))
    try:
        with pytest.raises(RuntimeError, match="QApplication"):
            ex.to_pdf_bytes(_payload())
    finally:
        monkeypatch.setattr(QApplication, "instance", real_instance)
