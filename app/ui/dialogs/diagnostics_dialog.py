"""LM Studio diagnostics dialog.

One row per ``[[llm_servers]]`` entry (plus the legacy ``[lmstudio]``
fallback when no servers are configured) showing:

* server name + base URL,
* current selected model,
* live status from the most recent health probe,
* latency,
* last error message,
* the model list returned by ``GET /v1/models``.

A **Refresh all** button issues a fresh probe per server through the
parent-supplied ``run_async`` runner so the dialog never blocks the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import AppSettings, LLMServer, LMStudioLLMSettings
from app.services.health_monitor import (
    HealthMonitor,
    HealthSnapshot,
    LLMServerStatus,
)
from app.utils.logging import get_logger


log = get_logger(__name__)


_STATUS_BADGES: dict[LLMServerStatus, str] = {
    LLMServerStatus.UNKNOWN: "⚪ unknown",
    LLMServerStatus.OFFLINE: "🔴 offline",
    LLMServerStatus.ONLINE: "🟢 online",
    LLMServerStatus.MODEL_UNAVAILABLE: "🟡 model unavailable",
    LLMServerStatus.GENERATING: "🔵 generating",
    LLMServerStatus.ERROR: "🔴 error",
}

#: Same shape the SettingsDialog uses for its connection tab.
RunAsync = Callable[[Awaitable[Any], Callable[[Any], None]], None]


@dataclass(frozen=True)
class _ServerEntry:
    label: str
    settings: LMStudioLLMSettings
    is_active: bool


def _entries_for(settings: AppSettings) -> list[_ServerEntry]:
    """Return one ``_ServerEntry`` per known LM Studio server."""
    if not settings.llm_servers:
        return [
            _ServerEntry(
                label=settings.lmstudio.base_url,
                settings=settings.lmstudio,
                is_active=True,
            )
        ]
    out: list[_ServerEntry] = []
    for idx, srv in enumerate(settings.llm_servers):
        effective = LMStudioLLMSettings(
            base_url=srv.base_url,
            api_key=srv.api_key or settings.lmstudio.api_key,
            model=srv.model or settings.lmstudio.model,
            timeout_seconds=srv.timeout_seconds or settings.lmstudio.timeout_seconds,
        )
        out.append(
            _ServerEntry(
                label=f"{srv.name}  ({srv.base_url})",
                settings=effective,
                is_active=(idx == settings.llm.active_server),
            )
        )
    return out


class _ServerCard(QGroupBox):
    """Single server's probe-result panel inside the dialog."""

    def __init__(self, entry: _ServerEntry, parent: QWidget | None = None):
        title = entry.label + ("  ← active" if entry.is_active else "")
        super().__init__(title, parent)

        self.entry = entry

        self._status = QLabel(_STATUS_BADGES[LLMServerStatus.UNKNOWN])
        self._status.setStyleSheet("font-weight: 600;")

        self._model = QLabel(f"<b>Selected model:</b> {entry.settings.model or '—'}")
        self._latency = QLabel("<b>Latency:</b> —")
        self._error = QLabel("<b>Last error:</b> —")
        self._error.setWordWrap(True)
        self._error.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._models = QPlainTextEdit()
        self._models.setReadOnly(True)
        self._models.setPlaceholderText("(refresh to populate)")
        self._models.setMaximumBlockCount(64)
        self._models.setFixedHeight(110)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addWidget(self._model)
        layout.addWidget(self._latency)
        layout.addWidget(self._error)
        layout.addWidget(QLabel("<b>GET /v1/models:</b>"))
        layout.addWidget(self._models)

    def apply_snapshot(self, snapshot: HealthSnapshot) -> None:
        self._status.setText(_STATUS_BADGES.get(snapshot.status, snapshot.status.label))
        self._model.setText(
            f"<b>Selected model:</b> {snapshot.model or '—'}"
        )
        if snapshot.latency_ms:
            self._latency.setText(f"<b>Latency:</b> {snapshot.latency_ms:.0f} ms")
        else:
            self._latency.setText("<b>Latency:</b> —")
        self._error.setText(f"<b>Last error:</b> {snapshot.error or '—'}")
        if snapshot.available_models:
            self._models.setPlainText("\n".join(snapshot.available_models))
        else:
            self._models.setPlainText("")
        self._models.setPlaceholderText(
            "(no models reported)" if snapshot.status != LLMServerStatus.UNKNOWN
            else "(refresh to populate)"
        )


class DiagnosticsDialog(QDialog):
    """Per-server health probe dashboard."""

    refreshed = Signal()

    def __init__(
        self,
        settings: AppSettings,
        *,
        run_async: RunAsync,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("LM Studio diagnostics")
        self.resize(640, 560)

        self._settings = settings
        self._run_async = run_async

        self._cards: list[_ServerCard] = []
        cards_box = QVBoxLayout()
        for entry in _entries_for(settings):
            card = _ServerCard(entry)
            self._cards.append(card)
            cards_box.addWidget(card)

        self.refresh_button = QPushButton("Refresh all")
        self.refresh_button.clicked.connect(self.refresh)

        button_row = QHBoxLayout()
        button_row.addWidget(self.refresh_button)
        button_row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addLayout(cards_box)
        layout.addLayout(button_row)
        layout.addWidget(buttons)

        # Kick off an initial refresh so the dialog isn't full of em-dashes.
        self.refresh()

    @Slot()
    def refresh(self) -> None:
        for card in self._cards:
            self._probe_card(card)

    def _probe_card(self, card: _ServerCard) -> None:
        # Build a fresh ephemeral monitor so the dialog reads exactly the
        # values from this card's entry — including overrides like a
        # custom timeout — without disturbing the live monitor.
        monitor = HealthMonitor(card.entry.settings, interval_s=60.0)

        def _on_done(result: Any) -> None:
            if isinstance(result, Exception):
                card.apply_snapshot(
                    HealthSnapshot(
                        status=LLMServerStatus.ERROR,
                        base_url=card.entry.settings.base_url,
                        model=card.entry.settings.model,
                        error=f"{type(result).__name__}: {result}",
                    )
                )
            else:
                assert isinstance(result, HealthSnapshot)
                card.apply_snapshot(result)
            self.refreshed.emit()

        try:
            self._run_async(monitor.probe_once(), _on_done)
        except Exception as exc:
            log.exception("Could not schedule diagnostic probe")
            _on_done(exc)
