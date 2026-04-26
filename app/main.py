"""Entry point: ``python -m app.main``."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from app.config.settings import load_settings, user_data_dir
from app.notes.database import Database
from app.notes.repository import NoteRepository
from app.ui.main_window import MainWindow
from app.utils.logging import configure_logging, get_logger


def main(argv: list[str] | None = None) -> int:
    settings = load_settings()
    configure_logging(level=getattr(logging, settings.app.log_level, logging.INFO))
    log = get_logger("app.main")

    data_dir = user_data_dir()
    db_path = settings.storage.resolved_path(data_dir)
    log.info("Using database at %s", db_path)

    db = Database(db_path)
    repository = NoteRepository(db)

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    window = MainWindow(settings, repository)
    window.show()

    try:
        return app.exec()
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
