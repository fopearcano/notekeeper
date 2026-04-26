"""Bundled icon + image assets.

Use :func:`asset_path` to resolve a file inside this package — works whether
Notekeeper is run from a source checkout, an editable install, or a wheel.
"""

from __future__ import annotations

from pathlib import Path

ASSETS_DIR: Path = Path(__file__).resolve().parent

#: Path to the application icon. The SVG is the canonical source; PyInstaller
#: builds a PNG/ICO from it for OS-level icon registration.
APP_ICON: Path = ASSETS_DIR / "notekeeper.svg"


def asset_path(filename: str) -> Path:
    """Return an absolute path to ``filename`` inside this asset package."""
    return ASSETS_DIR / filename


__all__ = ["ASSETS_DIR", "APP_ICON", "asset_path"]
