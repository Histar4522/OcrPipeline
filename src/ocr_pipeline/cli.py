"""Command-line interface for the local AI OCR pipeline."""

import argparse
from pathlib import Path
import sys
import threading
import webbrowser

from rich.console import Console
import uvicorn

from .config import OcrConfig
from .discovery import discover_documents
from .ollama_client import OllamaError, check_ollama_ready
from .runner import run_pipeline
from .web.server import create_app


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ocr-pipeline",
        description="Local Ollama-powered AI OCR pipeline for PDFs and image series -> Markdown.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Path to a PDF, image, image-series folder, or batch archive folder (omit when using --server).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory for archived .md files, cache, and ocr_pipeline.log.",
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Launch the native pywebview Desktop App with system tray and OS notifications.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Launch the local Web UI server instead of running a one-off CLI job.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for --server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8420,
        help="Port for --server (default: 8420).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the Web UI in the default browser.",
    )
    parser.add_argument(
        "--model",
        default="glm-ocr",
        help="Ollama OCR/vision model name (default: glm-ocr; alternatives: deepseek-ocr, qwen3-vl:4b, qwen3-vl:8b).",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama server base URL (default: from settings or http://localhost:11434).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PDF render resolution in DPI (default: 220).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1536,
        help="Max image dimension in pixels before downscaling (default: 1536).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-page Ollama request timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--series",
        action="store_true",
        help="Treat input_path directory as a single combined multi-page image series.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess documents and cached pages even if outputs already exist.",
    )
    parser.add_argument(
        "--no-page-headers",
        action="store_true",
        help="Omit <!-- Page N --> markers in multi-page outputs.",
    )
    return parser


def _launch_server(args: argparse.Namespace, console: Console) -> int:
    url = f"http://localhost:{args.port}"
    console.print(f"[bold cyan]Starting OcrPipeline Web UI[/bold cyan] at [underline]{url}[/underline]")
    app = create_app(ollama_url=args.ollama_url)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.app:
        from .desktop import launch_desktop_app

        return launch_desktop_app(ollama_url=args.ollama_url)

    if args.server:
        return _launch_server(args, console)

    if args.input_path is None or args.output is None:
        parser.error("input_path and -o/--output are required unless --app or --server is specified.")

    config = OcrConfig(
        model=args.model,
        ollama_url=args.ollama_url or "http://localhost:11434",
        dpi=args.dpi,
        max_image_dim=args.max_dim,
        timeout_seconds=args.timeout,
        force=args.force,
        include_page_headers=not args.no_page_headers,
    )

    try:
        tasks = discover_documents(args.input_path, args.output, series_mode=args.series)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Input error:[/red] {exc}")
        return 2

    if not tasks:
        console.print("[yellow]No PDF or image documents found to process.[/yellow]")
        return 0

    try:
        check_ollama_ready(config.ollama_url, config.model)
    except OllamaError as exc:
        console.print(f"[red]Ollama check failed:[/red] {exc}")
        return 1

    summary = run_pipeline(tasks, args.output, config, console=console)
    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"Processed: {summary.processed_docs}, "
        f"Skipped: {summary.skipped_docs}, "
        f"Failed pages: {summary.failed_pages}. "
        f"Log: {(args.output.resolve() / 'ocr_pipeline.log')}"
    )
    return 1 if summary.failed_pages > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
