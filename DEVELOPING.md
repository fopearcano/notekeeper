# Developing Notekeeper

Conventions and how-tos for working in the codebase. Read alongside
`README.md` (which has the user-facing install / configuration story
and the high-level architecture diagram).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,packaging]   # dev tooling + PyInstaller
```

Run the suite headless:

```bash
QT_QPA_PLATFORM=offscreen pytest
```

The Qt-heavy tests (`test_main_window`, `test_settings_dialog`,
`test_diagnostics_dialog`, `test_graceful_shutdown::test_main_window_*`)
need an offscreen platform plugin and a single `QApplication` per
process. The shared `qapp` module-scoped fixture in each file handles
that — don't construct your own.

## Threading model

Three threads ever touch shared state:

| Thread                | Owns                                | Must never                          |
| --------------------- | ----------------------------------- | ----------------------------------- |
| **GUI** (Qt)          | All `QWidget`s, all `QTimer`s       | Block on I/O or sleep               |
| **AsyncWorker**       | The `asyncio` loop, `SessionManager` | Touch widgets directly             |
| **PortAudio worker**  | The audio callback                  | Make HTTP calls or grab Qt locks    |

Cross-thread events flow through one of two mechanisms:

* **Qt signals with queued connections** for events the GUI must
  handle. Slots run on the GUI thread no matter which thread emitted.
* **`app.utils.listeners.ListenerList`** for in-process pub/sub inside
  the worker. Use it for any new "channel" the SessionManager (or
  similar service) wants to broadcast. Exceptions in listeners are
  logged, never re-raised — your producer keeps running.

When the GUI needs to call into the worker, go through
`MainWindow._submit(coro, error_label)` (returns a future or `None` if
the worker isn't ready) or the higher-level
`MainWindow._run_on_worker(coro, on_success=…, on_finally=…,
error_label=…)` which wires a uniform done-callback.

When the worker needs to update the GUI, **emit a Qt signal** —
don't reach into widgets directly. Each widget update has a dedicated
queued signal on `MainWindow` (see the `_segment_received`,
`_stream_delta`, `_health_received`, etc. fields).

## Adding a new transcription provider

1. Add a `Literal` value to `TranscriptionProviderName` in
   `app/config/settings.py` and (optionally) a new pydantic settings
   block.
2. Implement a class in `app/transcription/<name>_provider.py` that
   subclasses `TranscriptionProvider`:
   - set `provider_key` and `is_stub` (True if the backend isn't
     fully wired yet — the pipeline shows a clear UI warning).
   - implement `start`, `stop`, and `stream(chunks)` (async generator
     yielding `TranscriptSegment`).
   - report transient issues via `self._emit_warning(...)` and
     short-lived status via `self._emit_status(...)` — both come from
     the base class.
3. Wire the verb into `app/transcription/factory.py` so
   `create_transcription_provider(settings)` can build it.
4. Add a tab option in `_TRANSCRIPTION_PROVIDERS` in
   `app/ui/dialogs/settings_dialog.py` if it should be user-pickable.
5. Tests: mirror `tests/test_openai_audio_provider.py` — use
   `httpx.MockTransport` for HTTP-shaped providers and a fake model
   class injected via the existing `_import_*` indirection for ML
   backends.

## Adding a new LLM provider

Same shape, in `app/llm/`:

1. Add the verb to `LLMProviderName` and a settings block.
2. Implement `LLMProvider` (`complete` / `stream_complete` /
   `list_models`).
3. Update `app/llm/factory.py`.
4. Surface the verb in `_LLM_PROVIDERS` in the settings dialog and
   add fields under a `QGroupBox` per the existing OpenAI / Anthropic
   examples.
5. If your provider supports a ``GET /models``-style health probe, add
   a `probe_<name>` to `app/services/connection_tester.py` and an
   entry to the connections-tab `PROBES` tuple.

## Adding a new slash command

1. If a fresh prompt template is needed, add a `PromptTemplate` to
   `app/llm/prompt_templates.py` and append it to `TEMPLATES`.
2. Add a `Command(...)` row to `COMMANDS` in
   `app/llm/commands.py`. Use `arg_required=True` + `arg_choices` for
   verbs like `/rewrite <style>`; supply `arg_to_instruction` to fold
   the argument into the model's system instruction.
3. Add a test case to `tests/test_commands.py` and to
   `tests/test_prompt_templates.py` for the new template.
4. The toolbar exposes only the `UI_TASKS` shortlist; everything else
   is reachable via the command bar. Add to `UI_TASKS` only for verbs
   you expect users to run dozens of times a day.

## Adding a UI dialog

Put it under `app/ui/dialogs/<name>_dialog.py` and re-export from
`app/ui/dialogs/__init__.py`. Pattern in `SettingsDialog` /
`DiagnosticsDialog`:

* Take the parent's `run_async(coro, on_done)` callable for any
  network work — the dialog must not own a worker loop.
* Hold non-modal dialogs in `MainWindow._open_dialogs` so they
  survive past the spawning slot returning.
* Emit a single `*_saved` / `*_done` signal for state the parent must
  pick up — never reach back into the parent's private API.

## Logging

Module-level `log = get_logger(__name__)` (from `app.utils.logging`).
Levels in use:

| Level    | When                                                          |
| -------- | ------------------------------------------------------------- |
| DEBUG    | Per-chunk / per-line plumbing; useful when debugging audio.   |
| INFO     | Lifecycle: starts, stops, settings reloads, saves.            |
| WARNING  | Transient non-fatal issues (network retry, VAD silence).      |
| ERROR    | Fatal-for-the-current-operation; usually paired with a UI msg. |

`log.exception(...)` for handled exceptions — keeps the traceback in
the file log without surfacing it to the user.

## Async patterns

* All long-running work runs on the AsyncWorker loop. The
  `SessionManager` enforces that any task it kicks off (audio
  pipeline, LLM stream, health probe) lives on the same loop —
  cross-loop tasks raise `ValueError: future belongs to a different
  loop`. Tests that exercise multiple lifecycle calls must wrap the
  whole sequence in a single `asyncio.run`.
* `with_retry(fn, ...)` (`app/services/retry.py`) handles transient
  `httpx` errors with exponential backoff + jitter. Use it for any
  new network call that's safe to replay; never wrap a streaming
  iteration that has already yielded data (replays would duplicate
  output).
* The pipeline catches `NotImplementedError` from
  `provider.stream(...)` and turns it into a warning instead of
  crashing the worker. Use this for stub/placeholder providers.

## Tests

* Configuration / pure logic → standard `pytest` test files.
* HTTP-shaped providers → `httpx.MockTransport` injected via the
  module-level `httpx.AsyncClient` patch (see
  `tests/test_connection_tester.py::_patched_async_client` for the
  pattern).
* Audio recorder → fake `sd.InputStream` (`tests/test_recorder*.py`)
  so PortAudio isn't required.
* Multi-loop async tests → wrap the full sequence in a single
  `asyncio.run`. Pipelines bind tasks to whichever loop ran `start()`.
* Qt widgets → set `QT_QPA_PLATFORM=offscreen` early; share the
  `QApplication` via a module-scoped `qapp` fixture; assert visibility
  with `widget.isHidden()` (offscreen widgets aren't *visible* by Qt's
  definition).
* SessionManager fixtures → stub `health_monitor.probe_once` to a
  no-op so the post-action probe doesn't burn ~3 s on a default httpx
  timeout against an unreachable LAN host.

## Release checklist

1. Bump `app/__init__.__version__` (the source of truth).
2. `pytest` — must be green.
3. Smoke-test both entry points:
   ```bash
   pip install -e .
   notekeeper                      # console-script
   python -m app.main              # module form
   ```
4. (Optional) Build a desktop bundle:
   ```bash
   pip install -e .[packaging]
   pyinstaller notekeeper.spec --noconfirm --clean
   ```
5. Tag the commit and push.

## Outstanding cleanups

These are noted but deliberately deferred so the stability pass
doesn't risk behavior changes:

* Migrate the ~12 hand-rolled async submission sites in `MainWindow`
  to `_run_on_worker(...)`. The helper exists; switch one site per
  PR with a focused test.
* Fold `TranscriptPipeline`'s warning + status channels onto
  `ListenerList` (the segment channel needs async dispatch and stays
  manual).
* The `DiagnosticsDialog` currently builds a fresh `HealthMonitor`
  per probe; sharing the live monitor's snapshot would avoid a small
  duplicate request burst when the dialog is opened mid-session.
