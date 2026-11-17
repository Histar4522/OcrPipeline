# OcrPipeline

A 100% free, offline, cross-platform AI OCR desktop application and CLI powered by local models via **Ollama**. Built for archiving PDFs and image series—including handwritten notes, LaTeX math, tables, and diagrams—into structured **Markdown (`.md`)**.

## Features

- **Native Desktop App (`pywebview`)**: Cross-platform window with **Auto / Light / Dark** appearance modes, background **system tray** retention (`pystray`), and **OS toast notifications** when jobs finish.
- **5-Step Post-Install Setup Wizard**: Checks OS/GPU/VRAM hardware, detects and launches local Ollama automatically, streams model download progress, and configures quality presets and archive preferences.
- **Dedicated Local OCR Models**: Defaults to **`glm-ocr`** (~0.9B, fast, fits comfortably in 8 GB VRAM without shared-memory spill), with support for `deepseek-ocr`, `qwen3-vl:4b`, `qwen3-vl:8b`, and any Ollama vision model.
- **Job & Resume Pipeline**: Upload files/folders or point to local paths; set output name, archive folder, and tags; track live progress & **ETA**; terminate jobs anytime; and resume interrupted multi-page PDFs via `.ocr_cache/`.
- **Indexed Artifact Manager**: Full-text search across names, tags, and content; filter by folder or tag; sort; preview rendered Markdown (**KaTeX** LaTeX equations + GFM tables) or raw source with selectable text; **Rename**, **Move**, edit **Tags**, export `.md`/`.zip`, or open in **Obsidian**.
- **Obsidian Vault Mode**: Optional setting that initializes `.obsidian/` in your archive directory and writes YAML frontmatter (`title`, `created`, `source`, `model`, `tags`) to archived notes.

## Installation

### Option 1: Prebuilt Releases (Windows & Linux)
Download from the [GitHub Releases](https://github.com/Histar4522/OcrPipeline/releases):
- **Windows (`OcrPipeline-<version>-x64.msi`)**: Run the installer wizard for a per-user install (`%LOCALAPPDATA%\Programs\OcrPipeline`) with Start Menu and Desktop shortcuts. Requires [Ollama](https://ollama.com/download).
- **Linux (`OcrPipeline-<version>-linux-x86_64.tar.gz`)**: Extract and run `./install.sh` (installs to `~/.local/share/OcrPipeline`, creates a desktop entry, and links `~/.local/bin/ocr-pipeline-app`; run `./uninstall.sh` to remove). Requires [Ollama](https://ollama.com/download) and system GTK/WebKit (`python3-gi`, `gir1.2-webkit2-4.1`) or Qt bindings for `pywebview`.

### Option 2: Run from Source (Windows, macOS, Linux)
Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):
```bash
git clone https://github.com/Histar4522/OcrPipeline.git
cd OcrPipeline
uv sync
```

## Using the App

### 1. Launch the Desktop App
- From a prebuilt install, launch **OcrPipeline** from your Start Menu / Desktop / Application launcher.
- From source:
  ```bash
  uv run ocr-pipeline --app
  # or:
  uv run ocr-pipeline-app
  ```

### 2. First-Run Setup Wizard
On first launch (or anytime via **Settings → Start Setup Wizard**), follow the 5 steps:
1. **System**: Inspect detected OS, RAM, and GPU/VRAM.
2. **Ollama**: Connect to or auto-launch your local Ollama service (`http://localhost:11434`).
3. **Model**: Choose a default model and download `glm-ocr` (or another recommended model) with live progress.
4. **Archive & App**: Set your archive directory, quality preset (*Fast Draft* `150 DPI`, *Balanced* `220 DPI`, *High-Detail* `300 DPI`), Obsidian Vault Mode, tray, and toast preferences.
5. **Finish**: Review your summary and open the workspace.

### 3. Run OCR Jobs (**Jobs** tab)
- Choose **Upload Files / Folder** (drag-and-drop or browse) or **Local Path** (process large folders directly on disk without copying).
- Optionally set **Output Name**, **Archive Folder**, **Tags** (comma-separated), model, DPI, and **Combine images into 1 doc** (for multi-image page series).
- Monitor document/page progress and **ETA**, inspect live logs, preview rendered Markdown, or click **Terminate** to stop safely while preserving cached pages.
- Closing the window hides it to the system tray (when enabled) so long jobs continue in the background; use the tray icon menu to reopen, open the archive folder, or **Quit**.

### 4. Browse & Organize (**Artifacts**, **Models**, **Settings** tabs)
- **Artifacts**: Search by filename, tag, or content; filter by folder/tag; **Create Folder**, **Rename**, **Move**, edit **Tags**, **Open in Obsidian**, or reveal in your OS file manager.
- **Models**: Inspect installed Ollama models, pull new ones, set defaults, or delete unused models.
- **Settings**: Customize Ollama URL, archive path, DPI, `num_ctx`, transcription prompt, **Auto / Light / Dark** theme, tray, notifications, and Obsidian Vault Mode.

### Browser & CLI Modes
- **Browser UI mode** (fixed port `8420` by default):
  ```bash
  uv run ocr-pipeline --server
  ```
- **CLI batch mode**:
  ```bash
  uv run ocr-pipeline "/path/to/docs" -o "/path/to/archive" --model glm-ocr
  ```
  Add `--series` to combine a folder of images into one `.md` file, `--dpi 300` for fine handwriting, or `--force` to ignore `.ocr_cache/` and reprocess existing outputs.

## Building Installers (Cross-Platform)

Run the unified Python build script on **Windows** (`.msi` via WiX Toolset v6), **macOS** (`.app` + `.dmg`, untested on macOS hardware), or **Linux** (`.tar.gz` with `.desktop` + `install.sh`):
```bash
uv sync
uv run --no-sync --with pyinstaller python installer/build_installer.py
```
Use `--bundle-only` to build only the PyInstaller bundle in `dist/`, or `--skip-pyinstaller` to re-package an existing `dist/` bundle.

## License

Dual-licensed under the **MIT License** and the **Apache License, Version 2.0** (`MIT OR Apache-2.0`). See [LICENSE](LICENSE).

Copyright © 2026 Weien Xu
