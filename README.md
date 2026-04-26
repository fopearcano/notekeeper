# Notekeeper

Voice-driven notebook with live transcription and LLM-assisted note
processing. Records the microphone, transcribes speech in near real time,
and lets you run slash-commands (`/clean`, `/summarize`, `/rewrite clear`,
…) over the transcript through a local LM Studio server (or OpenAI /
Anthropic) — all from a single Qt desktop app.

![Notekeeper layout](app/assets/notekeeper.svg)

- **Local-first.** Default stack is faster-whisper + LM Studio over the LAN.
  No data leaves your machine unless you switch to a cloud provider.
- **Streaming.** Tokens stream into the processed-note panel as the model
  produces them.
- **Resilient.** LM Studio reachability shows live in the status bar; LAN
  hiccups retry with exponential backoff; pause/resume preserves the
  audio clock.
- **Exportable.** Notes round-trip to Markdown / Plain text / JSON / PDF.

---

## Quick start

```bash
git clone https://github.com/fopearcano/notekeeper
cd notekeeper

python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

pip install -e .                       # core deps (no Whisper)
pip install -e .[full]                 # add faster-whisper for local STT
pip install -e .[dev]                  # add the test toolchain

notekeeper                             # console-script entry point
# or, equivalently:
python -m app.main
```

On first launch Notekeeper:

1. Creates `~/.notekeeper/config.toml` from the bundled defaults.
2. Creates `~/.notekeeper/data/notekeeper.db` (SQLite).
3. Opens the main window with three panels (notes list / live transcript /
   processed note) and a bottom status bar showing the active providers
   and LM Studio health.

---

## Stack

- Python 3.11+
- [PySide6](https://doc.qt.io/qtforpython-6/) for the GUI
- [sounddevice](https://python-sounddevice.readthedocs.io/) +
  [PortAudio](http://www.portaudio.com/) for microphone capture
- `asyncio` + `QThread` workers so the UI never blocks
- SQLite (via `sqlite3` stdlib) for local note storage
- [pydantic](https://docs.pydantic.dev/) for config models
- [httpx](https://www.python-httpx.org/) for API calls
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (optional
  extra) for local speech-to-text

---

## Layout

```
app/
  main.py                       # Entry point (also exposed as ``notekeeper``)
  ui/                           # PySide6 widgets, main window, dialogs
  audio/                        # Mic capture, VAD, ring buffer
  transcription/                # faster-whisper / OpenAI / LM Studio (stub)
  llm/                          # LM Studio / OpenAI / Anthropic + commands
  notes/                        # SQLite storage and pydantic models
  config/                       # Settings, TOML I/O, bundled default_config.toml
  services/                     # Session manager, pipeline, exporters, health
  assets/                       # App icon (SVG)
  utils/                        # Logging helpers
tests/                          # pytest suite
notekeeper.spec                 # PyInstaller spec for desktop builds
config.example.toml             # Annotated example user config
.env.example                    # Provider API-key environment variables
```

---

## Configuration

The bundled `app/config/default_config.toml` is copied to
`~/.notekeeper/config.toml` on first launch. Edit that copy — the bundled
file is overwritten by package upgrades. See `config.example.toml` at the
repo root for a fully annotated reference (multi-server LM Studio,
provider switching, VAD thresholds, etc.).

API keys live in environment variables; each provider names the variable
in `api_key_env` (see `.env.example`):

- `[openai_audio]` and `[openai]` → `OPENAI_API_KEY`
- `[anthropic]` → `ANTHROPIC_API_KEY`
- `[lmstudio]` → uses an inline `api_key` (LM Studio accepts any
  non-empty bearer string by default)

Notekeeper **never** reads or stores the API key value itself — it stores
only the variable name and resolves the value at request time.

### Recommended provider setup

| Stage                 | Recommended         | Why                                                       |
| --------------------- | ------------------- | --------------------------------------------------------- |
| Speech-to-text        | **faster-whisper**  | Fully local, GPU-accelerated, no roundtrip latency.       |
| LLM cleanup / summary | **LM Studio**       | Local OpenAI-compatible chat/completions, no API costs.   |

The `openai_audio` provider is available for users who want cloud
transcription or are pointing at a self-hosted server that mirrors OpenAI's
`/audio/transcriptions` shape. The `lmstudio_audio` provider is a disabled
experimental stub — LM Studio doesn't currently expose a real-time audio
transcription endpoint.

### Multiple LM Studio servers (LAN)

Add an array of tables to your config to switch between LM Studio servers
from the toolbar:

```toml
[llm]
provider = "lmstudio"
active_server = 0

[[llm_servers]]
name = "Local 3090 Server"
base_url = "http://192.168.1.100:1234/v1"
provider = "lmstudio"

[[llm_servers]]
name = "Backup Server"
base_url = "http://192.168.1.101:1234/v1"
provider = "lmstudio"
```

The status bar shows live health (online / model unavailable / generating
/ offline / error) of the active server. **Tools → Diagnostics** runs
parallel `GET /v1/models` probes against every configured server.

---

## Slash commands

Type a slash command in the bar below the processed-note panel and press
Enter or `Ctrl+Enter`:

```
/clean                  Clean filler words, fix punctuation
/summarize              Bullet-point summary
/organize               Structured outline
/markdown               Reformat as readable markdown
/tasks                  Extract a checklist of action items
/title                  Generate a short title
/tags                   Suggest 3–7 lower-case tags
/outline                Hierarchical outline
/keypoints              5–10 stand-alone key points
/rewrite clear          Rewrite in clear style
/rewrite elegant        ditto in elegant style
/rewrite technical      ditto in technical style
/rewrite cinematic      ditto in cinematic style
```

The command runs against the user's text selection if there is one,
otherwise the full transcript. Results stream into the processed-note
panel as they arrive; the suggested title and tags appear in the small
footer beneath it.

### Keyboard shortcuts

| Action                    | Shortcut         |
| ------------------------- | ---------------- |
| Save note                 | `Ctrl+S`         |
| Toggle recording          | `Ctrl+R`         |
| Clean transcript          | `Ctrl+Shift+C`   |
| Run command bar           | `Ctrl+Enter`     |
| Settings                  | `Ctrl+,`         |
| Export Markdown           | `Ctrl+Shift+M`   |
| Export Text               | `Ctrl+Shift+T`   |
| Export JSON               | `Ctrl+Shift+J`   |
| Export PDF                | `Ctrl+Shift+P`   |
| Copy processed note       | `Ctrl+Shift+N`   |
| Copy raw transcript       | `Ctrl+Shift+R`   |
| Copy selected text        | `Ctrl+Shift+S`   |

---

## Installation by platform

### Linux

```bash
sudo apt install libportaudio2 libegl1 libxkbcommon0 libdbus-1-3 libfontconfig1
pip install -e .[full]
```

`libportaudio2` is the runtime PortAudio used by `sounddevice`; the rest
are Qt's display dependencies.

### macOS

```bash
brew install portaudio
pip install -e .[full]
```

The first time you hit Start Recording, macOS prompts for microphone
access. Grant it under **System Settings → Privacy & Security →
Microphone**.

### Windows

`pip install -e .[full]` ships PortAudio as part of the `sounddevice`
wheel — no extra setup needed for audio. Some antivirus tools flag
PyInstaller-built executables; sign your build or whitelist the
`notekeeper.exe` path.

---

## faster-whisper / CUDA

`faster-whisper` runs on top of [CTranslate2](https://github.com/OpenNMT/CTranslate2),
which has separate wheels for CPU-only and CUDA. Default `pip install
-e .[full]` picks the wheel matching your platform; verify with:

```bash
python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
```

If CUDA isn't available the call raises. Either:

- Switch `[faster_whisper].device = "cpu"` and `compute_type = "int8"`
  in your config, **or**
- Leave `[faster_whisper].allow_cpu_fallback = true` (default) — the
  provider will retry on CPU automatically when CUDA load fails, with
  `compute_type=float16/int8_float16` coerced to `int8`.

Compute-type guide:

| Hardware                        | `device`  | `compute_type`     |
| ------------------------------- | --------- | ------------------ |
| NVIDIA GPU, ≥ 8 GB VRAM         | `cuda`    | `float16`          |
| NVIDIA GPU, ≤ 6 GB VRAM         | `cuda`    | `int8_float16`     |
| Apple Silicon                   | `cpu`     | `int8`             |
| Linux/Windows CPU only          | `cpu`     | `int8`             |

Models (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`)
download automatically into the Hugging Face cache on first use.
`small` is a good default — usable English transcription, ~250 MB.

---

## Tests

```bash
pip install -e .[dev]
pytest
```

The suite covers ~330 cases: imports, config + TOML round-trip, audio
chunk assembly, VAD, recorder pause/resume, transcript pipeline (VAD
filter, chunk events, status forwarding), faster-whisper / OpenAI-audio
providers (lazy load, CPU fallback, error paths via `httpx.MockTransport`),
LM Studio (streaming SSE + `/v1/models`), prompt templates and slash
commands, repository CRUD + cascade, session lifecycle (start / pause /
resume / stop / clear / save / autosave / load), exporters (md / txt /
json / PDF + safe filenames), and Settings / Diagnostics dialogs under
`QT_QPA_PLATFORM=offscreen`.

---

## Packaging

A PyInstaller spec is included at `notekeeper.spec`:

```bash
pip install -e .[packaging]
pyinstaller notekeeper.spec --noconfirm --clean
```

Build a single-file executable instead of a folder bundle:

```bash
ONEFILE=1 pyinstaller notekeeper.spec --noconfirm --clean
```

The spec collects PortAudio dynamic libraries, faster-whisper /
ctranslate2 data files, the bundled config TOML, and the SVG icon. On
macOS it produces a `.app` with the microphone-usage Info.plist key set.

---

## Troubleshooting

**"Audio backend unavailable: PortAudio library not found"**
Install PortAudio for your OS (see the [platform notes](#installation-by-platform)).

**"Could not open microphone"**
Another app is holding the mic exclusively, or the configured device
name doesn't exist. Check the toolbar's *Mic* combobox or run
`python -c "import sounddevice as sd; print(sd.query_devices())"` to see
what's available.

**Status bar shows "🔴 offline" for LM Studio**
Verify the server is running and reachable: open
`http://192.168.1.100:1234/v1/models` in a browser, or click
**Tools → Test LM Studio Connection…**. If the server is on a different
machine, make sure your firewall allows incoming connections on its
port.

**Status bar shows "🟡 model unavailable"**
The base URL is reachable, but the model id named in `[lmstudio].model`
isn't loaded. Open LM Studio's UI and load the model, or change the id
in **Tools → Settings…**.

**"Whisper model could not be loaded: CUDA out of memory"**
Lower the model size (`small` → `base`), switch `compute_type` to
`int8_float16`, or set `device = "cpu"`.

**"This endpoint does not support audio transcription"**
You pointed `[transcription].provider = "openai_audio"` at a server
that doesn't expose `/audio/transcriptions`. Either switch to
`faster_whisper` for local STT, or update `base_url`.

**"401 authentication failed"**
The `api_key_env` variable is unset (or wrong). Run
`echo $OPENAI_API_KEY` / `printenv ANTHROPIC_API_KEY`. Notekeeper
re-reads the env on the next request, so you don't need to restart.

**Transcript shows "(silence — skipped)" for everything**
The energy-based VAD is dropping your audio. Lower
`[audio].vad_threshold` (default `0.01` is "quiet room speech") in
**Tools → Settings… → Audio**, or untick **VAD** in the toolbar.

**"Stop the recording before applying new settings"**
Apply happens on the asyncio worker; live audio settings (sample rate,
chunk seconds, transcription provider) can't change mid-stream. Hit
Stop, then re-open Settings.

**`pip install -e .[full]` fails to build ctranslate2 wheel**
Pre-built wheels exist for Linux/macOS/Windows on x86-64 and Apple
Silicon. ARM Linux usually falls back to a source build that needs
`cmake` + a C++ compiler. Either install those, or stay on `pip
install -e .` (no Whisper) and use the cloud `openai_audio` provider.

**`notekeeper` command not found**
Either you skipped the editable install, or your venv isn't on `PATH`.
Use `python -m app.main` as a fallback — it always works from a checkout.

---

## License

MIT — see [pyproject.toml](pyproject.toml).
