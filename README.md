# OcrPipeline

A cross-platform AI OCR desktop application and CLI for archiving PDFs and image series—including handwritten notes, LaTeX math, tables, and diagrams—into structured **Markdown (`.md`)**. Runs 100% locally via **Ollama** or **llama.cpp**, or connects to cloud Vision/LLM providers (**OpenAI**, **Anthropic**, **Google Gemini**, **DeepSeek**, **Grok**, **Qwen**, or **Custom OpenAI-compatible** servers).

## Features

- **Multi-Provider Vision OCR (Mandatory) + Optional Assistant LLM**:
  - **Vision / OCR Role (Required)**: Use local engines (**Ollama**, **llama.cpp**) or cloud providers (**OpenAI**, **Anthropic**, **Google Gemini**, **Grok**, **Qwen**, **Custom**) for accurate page-to-Markdown OCR.
  - **Assistant LLM Role (Optional)**: Pair a small local LLM (e.g. `qwen3:1.7b`, `llama3.2:3b`) or cloud model to unlock **Auto-Tagging**, **Auto-Renaming**, **Auto-Summarization**, **AI Folder Content Summary**, and **OCR Proofread/Cleanup** (with sidecar `.md.meta` JSON metadata). When left unconfigured, core OCR runs normally without AI post-processing.
- **Native Desktop App (`pywebview`)**: Cross-platform window with **Auto / Light / Dark** appearance modes, background **system tray** retention (`pystray`), desktop toast notifications (*Always / Not in focus / Minimized to tray / Never*), and in-app GitHub update checking.
- **5-Step Post-Install Setup Wizard**: Checks OS/GPU/VRAM hardware, configures your Vision provider (or auto-launches local Ollama), sets optional Assistant LLM, and configures archive preferences.
- **High-Throughput Pipeline**: Parallel 2-page CPU prefetching/rendering, FIFO job queue with `waiting` classification, page-level `.ocr_cache/` resume, live ETA, and interactive file selection.
- **Indexed Artifact Explorer**: Breadcrumb folder navigation, full-text search across filenames/tags/summaries/content, tag-chip editing, drag-and-drop move, **KaTeX** math + GFM table rendering, native Save As export, and **Obsidian Vault Mode**.

## Installation

### Option 1: Prebuilt Releases (Windows, macOS, Linux)
Download from [GitHub Releases](https://github.com/Histar4522/OcrPipeline/releases):
- **Windows (`OcrPipeline-<version>-x64.msi`)**: Run the MSI installer wizard for a per-user install (`%LOCALAPPDATA%\Programs\OcrPipeline`).
- **macOS (`OcrPipeline-<version>-macOS-arm64.dmg`)** *(untested on macOS hardware)*: Open the `.dmg` and drag `OcrPipeline.app` to `Applications`.
- **Linux (`OcrPipeline-<version>-linux-x86_64.tar.gz`)** *(untested on desktop Linux hardware)*: Extract and run `./install.sh` (installs to `~/.local/share/OcrPipeline` and links `~/.local/bin/ocr-pipeline-app`; run `./uninstall.sh` to remove).

For 100% local offline OCR, install [Ollama](https://ollama.com/download) or run `llama.cpp` `llama-server`; cloud providers only need an API key configured in **Providers & Models** (or standard env vars like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `DASHSCOPE_API_KEY`).

### Option 2: Run from Source (Windows, macOS, Linux)
Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):
```bash
git clone https://github.com/Histar4522/OcrPipeline.git
cd OcrPipeline
uv sync
```

## Using the App

### 1. Launch the Desktop App
- From a prebuilt install, launch **OcrPipeline** from your application menu.
- From source:
  ```bash
  uv run ocr-pipeline --app
  ```

### 2. Configure Providers & Models
In **Providers & Models** (or the Setup Wizard):
- **Vision / OCR (Required)**: Choose your provider and model (e.g. `ollama` + `glm-ocr`, or `openai` + `gpt-4.1-mini`).
- **Assistant LLM (Optional)**: Toggle on and select a lightweight local LLM (e.g. `qwen3:1.7b`, `llama3.2:3b`) or cloud LLM (`deepseek-chat`, `gemini-2.5-flash-lite`, etc.), plus post-OCR automation defaults.

### 3. Run OCR Jobs & Organize Artifacts
- **Jobs**: Drop or browse PDFs/images (click filenames to open or reveal in folder), choose archive folder/tags, toggle per-job AI post-processing (`Auto-tag`, `Auto-summarize`, `Auto-rename`), and monitor progress/ETA.
- **Artifacts**: Browse subfolders, filter/search, edit tag chips, or use **AI → Auto-Generate Tags / Suggest Rename / Generate Summary / Proofread OCR** and **AI Summarize Folder**.
- **Settings**: Customize archive paths, OCR quality, notifications, theme, and check GitHub releases under **Settings → Update**.

### Browser & CLI Modes
- **Browser UI mode**: `uv run ocr-pipeline --server`
- **CLI batch mode**:
  ```bash
  uv run ocr-pipeline "/path/to/docs" -o "/path/to/archive" --provider ollama --model glm-ocr
  ```
  Supports `--provider`, `--model`, `--base-url`, `--api-key`, `--series`, `--dpi`, `--force`.

## Building Installers (Cross-Platform)

```bash
uv sync
uv run --no-sync --with pyinstaller python installer/build_installer.py --release
```
Omit `--release` to produce a local beta build stamped `x.x.x-dev-<commit>-<timestamp>`.

## License

Dual-licensed under the **MIT License** and the **Apache License, Version 2.0** (`MIT OR Apache-2.0`). See [LICENSE](LICENSE).

Copyright © 2026 Weien Xu
