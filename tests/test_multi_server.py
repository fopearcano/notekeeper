"""``[[llm_servers]]`` config selection + TOML round-trip."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from app.config.io import dumps_toml, save_settings
from app.config.settings import LLMServer, load_settings
from app.llm.factory import create_llm_provider


def _two_servers_overrides():
    return {
        "llm_servers": [
            {
                "name": "Local 3090",
                "base_url": "http://192.168.1.100:1234/v1",
                "provider": "lmstudio",
            },
            {
                "name": "Backup",
                "base_url": "http://192.168.1.101:1234/v1",
                "provider": "lmstudio",
                "api_key": "backup-token",
                "model": "qwen-2.5-coder",
                "timeout_seconds": 30,
            },
        ]
    }


def test_load_with_llm_servers_picks_first_by_default():
    settings = load_settings(bootstrap=False, overrides=_two_servers_overrides())
    assert settings.llm.active_server == 0
    active = settings.active_lmstudio_settings()
    assert active.base_url == "http://192.168.1.100:1234/v1"
    # Empty overrides fall through to the [lmstudio] base.
    assert active.api_key == settings.lmstudio.api_key
    assert active.model == settings.lmstudio.model


def test_active_server_index_picks_other_entry():
    overrides = _two_servers_overrides()
    overrides["llm"] = {"active_server": 1}
    settings = load_settings(bootstrap=False, overrides=overrides)

    active = settings.active_lmstudio_settings()
    assert active.base_url == "http://192.168.1.101:1234/v1"
    # Per-server overrides apply.
    assert active.api_key == "backup-token"
    assert active.model == "qwen-2.5-coder"
    assert active.timeout_seconds == 30


def test_active_server_index_out_of_range_falls_back_to_lmstudio_block():
    overrides = _two_servers_overrides()
    overrides["llm"] = {"active_server": 99}
    settings = load_settings(bootstrap=False, overrides=overrides)

    active = settings.active_lmstudio_settings()
    assert active.base_url == settings.lmstudio.base_url


def test_no_servers_uses_legacy_lmstudio_block():
    settings = load_settings(bootstrap=False)
    active = settings.active_lmstudio_settings()
    assert active.base_url == settings.lmstudio.base_url
    assert active.model == settings.lmstudio.model


def test_active_server_label_returns_friendly_name():
    settings = load_settings(bootstrap=False, overrides=_two_servers_overrides())
    assert settings.active_server_label() == "Local 3090"

    settings = load_settings(
        bootstrap=False, overrides={"llm": {"active_server": 1}, **_two_servers_overrides()}
    )
    assert settings.active_server_label() == "Backup"


def test_factory_uses_active_server():
    settings = load_settings(
        bootstrap=False,
        overrides={
            "llm": {"provider": "lmstudio", "active_server": 1},
            **_two_servers_overrides(),
        },
    )
    provider = create_llm_provider(settings)
    # The LMStudioProvider should hold the *active* server's settings.
    assert provider.settings.base_url == "http://192.168.1.101:1234/v1"
    assert provider.settings.api_key == "backup-token"


def test_dumps_toml_emits_array_of_tables():
    data = {
        "llm": {"provider": "lmstudio", "active_server": 0},
        "llm_servers": [
            {"name": "A", "base_url": "http://10.0.0.1/v1", "provider": "lmstudio"},
            {"name": "B", "base_url": "http://10.0.0.2/v1", "provider": "lmstudio"},
        ],
    }
    text = dumps_toml(data)
    assert text.count("[[llm_servers]]") == 2
    parsed = tomllib.loads(text)
    assert parsed["llm_servers"][0]["name"] == "A"
    assert parsed["llm_servers"][1]["base_url"] == "http://10.0.0.2/v1"
    assert parsed["llm"]["active_server"] == 0


def test_save_and_reload_round_trips_servers(tmp_path: Path):
    settings = load_settings(bootstrap=False, overrides=_two_servers_overrides())
    target = tmp_path / "config.toml"
    save_settings(settings, path=target)

    reloaded = load_settings(user_config=target, bootstrap=False)
    assert len(reloaded.llm_servers) == 2
    assert reloaded.llm_servers[0].name == "Local 3090"
    assert reloaded.llm_servers[1].api_key == "backup-token"


def test_llm_server_provider_constrained_to_lmstudio():
    """We only support LM Studio in [[llm_servers]] right now."""
    LLMServer(name="x", base_url="http://x/v1", provider="lmstudio")
    with pytest.raises(Exception):
        LLMServer(name="x", base_url="http://x/v1", provider="openai")  # type: ignore[arg-type]


def test_save_settings_does_not_emit_array_when_empty(tmp_path: Path):
    """Default config (no [[llm_servers]]) should round-trip without inventing one."""
    settings = load_settings(bootstrap=False)
    target = tmp_path / "config.toml"
    save_settings(settings, path=target)

    text = target.read_text(encoding="utf-8")
    assert "[[llm_servers]]" not in text
    reloaded = load_settings(user_config=target, bootstrap=False)
    assert reloaded.llm_servers == []
