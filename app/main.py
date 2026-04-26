"""Entry point.

Two ways to launch:

* ``python -m app.main`` — module form, works from a source checkout.
* ``notekeeper`` — console script registered by ``pyproject.toml``'s
  ``[project.scripts]`` once the package is installed.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app import __version__
from app.assets import APP_ICON
from app.config.settings import load_settings, user_data_dir
from app.notes.database import Database
from app.notes.repository import NoteRepository
from app.ui.main_window import MainWindow
from app.utils.logging import configure_logging, get_logger


def main(argv: list[str] | None = None) -> int:
    """Boot Notekeeper and run the Qt event loop. Returns the exit code."""
    settings = load_settings()
    configure_logging(level=getattr(logging, settings.app.log_level, logging.INFO))
    log = get_logger("app.main")
    log.info("Starting Notekeeper %s", __version__)

    data_dir = user_data_dir()
    db_path = settings.storage.resolved_path(data_dir)
    log.info("Using database at %s", db_path)

    db = Database(db_path)
    repository = NoteRepository(db)

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(settings.app.name)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("Notekeeper")
    if APP_ICON.exists():
        app.setWindowIcon(QIcon(str(APP_ICON)))

    window = MainWindow(settings, repository)
    window.show()

    try:
        return app.exec()
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
