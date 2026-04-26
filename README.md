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

`app/config/default_config.toml` ships sensible defaults. On first launch,
they are copied to `~/.notekeeper/config.toml` — edit that file to customise
providers and models. The bundled defaults are merged in as a base layer so
older user files keep validating after upgrades that add new keys.

API keys live in environment variables. Each provider section names the
variable in `api_key_env`:

- `[openai_audio]` and `[openai]` → `OPENAI_API_KEY`
- `[anthropic]` → `ANTHROPIC_API_KEY`
- `[lmstudio]` → uses an inline `api_key` (LM Studio accepts a placeholder string)

Active transcription and LLM providers are shown in the right side of the
status bar (`ASR: faster_whisper | LLM: lmstudio`).

### Provider factories

```python
from app.config.settings import load_settings
from app.transcription.factory import create_transcription_provider
from app.llm.factory import create_llm_provider

settings = load_settings()
asr = create_transcription_provider(settings)   # → FasterWhisperProvider, etc.
llm = create_llm_provider(settings)             # → LMStudioProvider, etc.
```

> **Note:** `lmstudio_audio` is a deliberate stub. LM Studio does not expose a
> real-time audio transcription endpoint today; the stub stays in place until
> a compatible endpoint is explicitly available. Use `faster_whisper` or
> `openai_audio` for transcription. LM Studio is fully wired up for **LLM**
> processing via its OpenAI-compatible `/chat/completions` endpoint.

## Tests

```bash
pytest
```

The smoke tests verify that all packages import, the config loads, the SQLite
repository round-trips a note, and the transcript pipeline forwards segments
from a stub provider.
