# -*- mode: python ; coding: utf-8 -*-
"""Cross-platform PyInstaller spec for OcrPipeline (Windows, macOS, Linux)."""

from pathlib import Path
import sys

ROOT = Path(SPECPATH).resolve().parent
ENTRY = ROOT / "installer" / "app_entry.py"
STATIC_SRC = ROOT / "src" / "ocr_pipeline" / "web" / "static"

icon_path = None
if sys.platform == "win32":
    cand = ROOT / "installer" / "build_assets" / "app_icon.ico"
    if cand.exists():
        icon_path = str(cand)
elif sys.platform == "darwin":
    cand = ROOT / "installer" / "build_assets" / "app_icon.icns"
    if cand.exists():
        icon_path = str(cand)
else:
    cand = ROOT / "installer" / "build_assets" / "app_icon.png"
    if cand.exists():
        icon_path = str(cand)

hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "pystray",
    "webview",
    "notifypy",
    "pymupdf",
]
if sys.platform == "win32":
    hiddenimports.extend(["pystray._win32", "webview.platforms.winforms", "clr_loader", "pythonnet"])
elif sys.platform == "darwin":
    hiddenimports.extend(["pystray._darwin", "webview.platforms.cocoa"])
else:
    hiddenimports.extend(["pystray._appindicator", "pystray._xorg", "webview.platforms.gtk"])

a = Analysis(
    [str(ENTRY)],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[(str(STATIC_SRC), "ocr_pipeline/web/static")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OcrPipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="OcrPipeline",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="OcrPipeline.app",
        icon=icon_path,
        bundle_identifier="com.weienxu.ocrpipeline",
    )
