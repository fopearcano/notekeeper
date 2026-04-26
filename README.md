# Notekeeper

A desktop notebook app that records microphone audio, transcribes speech in
near real time, and lets you push the transcript through an LLM for cleanup,
summarization, organization, and formatting.

This repository currently contains the **initial scaffold**: the GUI shell,
clean interfaces for audio / transcription / LLM providers, and placeholder
services. No real audio is captured yet — hitting *Start Recording* drives a
fake transcript through the same pipeline that the live providers will use.

## Stack

- Python 3.11+
- [PySide6](https://doc.qt.io/qtforpython-6/) for the GUI
- [sounddevice](https://python-sounddevice.readthedocs.io/) for microphone
  capture (wired up later)
- `asyncio` + `QThread` workers so the UI never blocks
- SQLite (via `sqlite3` stdlib) for local note storage
- [pydantic](https://docs.pydantic.dev/) for config models
- [httpx](https://www.python-httpx.org/) for API calls

## Layout

```
app/
  main.py                       # Entry point
  ui/                           # PySide6 widgets and main window
  audio/                        # Mic capture, VAD, ring buffer (placeholders)
  transcription/                # Provider interface + faster-whisper / OpenAI / LM Studio
  llm/                          # Provider interface + LM Studio / OpenAI / Anthropic
  notes/                        # SQLite storage and pydantic models
  config/                       # Settings + bundled default_config.toml
  services/                     # Session manager, transcript pipeline, note processor
  utils/                        # Logging helpers
tests/                          # pytest smoke tests
```

## Running

```bash
pip install -e .[dev]
python -m app.main
```

The UI opens with:

- a left **notebook list** placeholder,
- a central **live transcript** panel,
- a right **processed note** panel,
- a bottom **status bar**,
- buttons for *Start Recording*, *Stop Recording*, *Save Note*, *Summarize*,
  *Organize*, *Format*.

## Configuration

`app/config/default_config.toml` ships sensible defaults. A user-level config
is loaded from `~/.config/notekeeper/config.toml` when present and merged on
top of the defaults. Environment variables (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`) override the values from disk.

## Tests

```bash
pytest
```

The smoke tests verify that all packages import, the config loads, the SQLite
repository round-trips a note, and the transcript pipeline forwards segments
from a stub provider.
