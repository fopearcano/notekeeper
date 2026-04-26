# PyInstaller spec for Notekeeper.
#
# Build a one-folder bundle:
#   pyinstaller notekeeper.spec --noconfirm --clean
#
# Build a single-file executable (slower startup, simpler to ship):
#   set ONEFILE=1 in your shell, then re-run the above.
#
# Notes:
#  * faster-whisper pulls in ctranslate2, which has CUDA / CPU variants
#    in separate wheels. PyInstaller picks up whichever is installed in
#    the current Python; rebuild on each target platform.
#  * sounddevice ships PortAudio binaries — they need to be collected
#    explicitly with collect_dynamic_libs.
#  * The bundled default_config.toml and the icon ride along as data
#    files so first launch can copy them into ``~/.notekeeper/``.

# -*- mode: python ; coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (  # type: ignore[import-not-found]
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


PROJECT_ROOT = Path(SPECPATH).resolve()
ENTRY_SCRIPT = str(PROJECT_ROOT / "app" / "main.py")
ONEFILE = bool(os.environ.get("ONEFILE"))

# ----- Data files ------------------------------------------------------------

# Bundle the default config + icon assets at the same package paths Notekeeper
# expects (so ``importlib.resources`` and ``Path(__file__).parent`` both work).
datas = [
    (str(PROJECT_ROOT / "app" / "config" / "default_config.toml"), "app/config"),
    (str(PROJECT_ROOT / "app" / "assets" / "notekeeper.svg"), "app/assets"),
]

# faster-whisper / ctranslate2 ship optional resources (model assets,
# tokenizers). ``collect_data_files`` is a no-op when the package isn't
# installed, so it's safe to leave on.
for pkg in ("faster_whisper", "ctranslate2", "sounddevice"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# ----- Binaries --------------------------------------------------------------

# PortAudio is shipped inside the sounddevice wheel; PyInstaller needs a hand
# to discover the .so/.dll/.dylib next to it.
binaries = []
try:
    binaries += collect_dynamic_libs("sounddevice")
except Exception:
    pass

# ----- Hidden imports --------------------------------------------------------

# Some sub-modules are looked up dynamically (factories, lazy provider
# imports). Pull the whole tree in so missing-module errors at runtime don't
# bite users on a stripped-down build.
hiddenimports: list[str] = []
hiddenimports += collect_submodules("PySide6")
hiddenimports += collect_submodules("app")
for opt in ("faster_whisper", "ctranslate2"):
    try:
        hiddenimports += collect_submodules(opt)
    except Exception:
        pass


block_cipher = None


a = Analysis(
    [ENTRY_SCRIPT],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)


_ICON = str(PROJECT_ROOT / "app" / "assets" / "notekeeper.svg")
_NAME = "notekeeper"


if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        name=_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        icon=_ICON,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        icon=_ICON,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name=_NAME,
    )

# Reduce some noise on macOS — bundle as a .app there.
if sys.platform == "darwin":
    app = BUNDLE(  # type: ignore[name-defined]
        exe if ONEFILE else coll,
        name=f"{_NAME}.app",
        icon=_ICON,
        bundle_identifier="com.notekeeper.app",
        info_plist={
            "NSMicrophoneUsageDescription":
                "Notekeeper records microphone audio for transcription.",
            "CFBundleShortVersionString": "0.1.0",
        },
    )
