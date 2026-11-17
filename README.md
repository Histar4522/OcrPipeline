# OcrPipeline

A 100% free, offline, cross-platform AI OCR desktop app and CLI powered by local Vision-Language & OCR Models via **Ollama**. Built for archiving PDFs and image series—including handwritten notes, tables, and graphics—into structured **Markdown (`.md`)**.

## Features

- **Native Desktop Window (`pywebview`)**: Standalone cross-platform app window (Windows, macOS, Linux).
- **Background System Tray**: Closing the window hides it to the system tray while background OCR jobs keep running.
- **Desktop Toast Notifications**: Cross-platform OS notifications when jobs finish.
- **Setup Wizard & Model Manager**: Check Ollama status, pull/delete models (`glm-ocr`, `deepseek-ocr`, `qwen3-vl:4b`, `qwen3-vl:8b`), and set defaults in-app.
- **Artifact Manager**: Search, preview (rendered & raw Markdown), rename, delete, download, and open output folders directly.
- **Resume & Checkpointing**: Page-level `.ocr_cache/` checkpoints allow interrupted multi-page PDFs to resume cleanly.

## Setup

```powershell
cd D:\Works\Workspaces\OcrPipeline
uv sync
```

## Usage

### Desktop App (Recommended)
```powershell
uv run ocr-pipeline --app
# or:
uv run ocr-pipeline-app
```

### Browser Web UI Mode
```powershell
uv run ocr-pipeline --server
```

### CLI Batch Mode
```powershell
uv run ocr-pipeline "D:\path\to\docs" -o "D:\path\to\archive" --model glm-ocr
```

## Building Installers (Cross-Platform)

Run the unified Python build script on **Windows** (`.msi` via WiX v6), **macOS** (`.app` + `.dmg`), or **Linux** (`.tar.gz` with `.desktop` + `install.sh`):
```powershell
uv run --no-sync --with pyinstaller python installer/build_installer.py
```
Use `--bundle-only` to build only the PyInstaller bundle in `dist/`, or `--skip-pyinstaller` to re-package an existing `dist/` bundle.
