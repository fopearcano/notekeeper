import pytest

from app.llm.prompt_templates import TEMPLATES, render_template


def test_known_templates_present():
    expected = {"summarize", "organize", "format", "cleanup", "understand"}
    assert expected.issubset(TEMPLATES.keys())


def test_render_template_substitutes_transcript():
    system, user = render_template("summarize", "Hello world")
    assert "summary" in system.lower() or "summarize" in system.lower()
    assert "Hello world" in user


def test_render_unknown_template_raises():
    with pytest.raises(KeyError):
        render_template("nope", "x")
