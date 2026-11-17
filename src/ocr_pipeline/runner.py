"""Batch OCR runner with page-level checkpointing, resume, cancellation, and logging."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
import hashlib
import inspect
import logging
from pathlib import Path
import threading
import time

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .artifacts import format_frontmatter, normalize_tags
from .config import OcrConfig
from .discovery import DocumentTask
from .ollama_client import JobCancelledError, ocr_image_b64
from .preprocess import count_pdf_pages, iter_pdf_pages_png, prepare_image_b64

OcrFn = Callable[..., str]


@dataclass(slots=True)
class ProgressEvent:
    """Snapshot of pipeline progress emitted after each document/page step."""

    total_docs: int
    completed_docs: int
    skipped_docs: int
    failed_pages: int
    current_doc: str = ""
    current_page: int = 0
    total_pages: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    status: str = "running"


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass(slots=True)
class RunSummary:
    """Summary counts from a pipeline run."""

    total_docs: int = 0
    processed_docs: int = 0
    skipped_docs: int = 0
    failed_pages: int = 0
    cancelled: bool = False


def _setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "ocr_pipeline.log"
    logger = logging.getLogger(f"ocr_pipeline.{id(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def _doc_cache_dir(output_dir: Path, task: DocumentTask, config: OcrConfig) -> Path:
    key = f"{task.output_path.resolve()}|{config.model}|{config.dpi}|{config.max_image_dim}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return output_dir / ".ocr_cache" / digest


def _iter_task_pages(task: DocumentTask, config: OcrConfig) -> Iterator[tuple[int, int, str]]:
    """Yield (1-based page_num, total_pages, base64_png) for a document task."""
    if task.kind == "pdf":
        for page_num, total, png_bytes in iter_pdf_pages_png(task.source_paths[0], dpi=config.dpi):
            yield page_num, total, prepare_image_b64(png_bytes, max_dim=config.max_image_dim)
        return

    total = len(task.source_paths)
    for idx, img_path in enumerate(task.source_paths, start=1):
        yield idx, total, prepare_image_b64(img_path, max_dim=config.max_image_dim)


def _total_pages_for_task(task: DocumentTask) -> int:
    if task.kind == "pdf":
        return count_pdf_pages(task.source_paths[0])
    return len(task.source_paths)


def _invoke_ocr(
    ocr_fn: OcrFn,
    image_b64: str,
    config: OcrConfig,
    client: httpx.Client,
    cancel_event: threading.Event | None,
) -> str:
    params = inspect.signature(ocr_fn).parameters
    if "cancel_event" in params:
        return ocr_fn(image_b64, config, client, cancel_event=cancel_event)
    return ocr_fn(image_b64, config, client)


def run_pipeline(
    tasks: list[DocumentTask],
    output_dir: Path,
    config: OcrConfig,
    ocr_fn: OcrFn = ocr_image_b64,
    console: Console | None = None,
    on_progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    on_client_ready: Callable[[httpx.Client], None] | None = None,
    work_dir: Path | None = None,
) -> RunSummary:
    """Execute OCR over discovered documents with resume, checkpointing, cancellation, and logging."""
    resolved_work = (work_dir or output_dir).resolve()
    logger = _setup_logger(resolved_work)
    ui = console or Console()
    summary = RunSummary(total_docs=len(tasks))
    run_start = time.monotonic()
    ocr_durations: list[float] = []
    pending_pages_total = sum(
        _total_pages_for_task(t) for t in tasks if config.force or not t.output_path.exists()
    )
    pages_done_overall = 0

    def is_cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def compute_eta() -> float | None:
        remaining = max(0, pending_pages_total - pages_done_overall)
        if remaining == 0:
            return 0.0
        if not ocr_durations:
            return None
        avg = sum(ocr_durations) / len(ocr_durations)
        return round(avg * remaining, 1)

    def emit(doc: str = "", page: int = 0, pages: int = 0, status: str = "running") -> None:
        if on_progress is not None:
            on_progress(
                ProgressEvent(
                    total_docs=summary.total_docs,
                    completed_docs=summary.processed_docs + summary.skipped_docs,
                    skipped_docs=summary.skipped_docs,
                    failed_pages=summary.failed_pages,
                    current_doc=doc,
                    current_page=page,
                    total_pages=pages,
                    elapsed_seconds=round(time.monotonic() - run_start, 1),
                    eta_seconds=compute_eta(),
                    status=status,
                )
            )

    logger.info(
        "Starting OCR run: docs=%d model=%s dpi=%d max_dim=%d force=%s",
        len(tasks),
        config.model,
        config.dpi,
        config.max_image_dim,
        config.force,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=ui,
    )

    client = httpx.Client(timeout=config.timeout_seconds)
    if on_client_ready is not None:
        on_client_ready(client)

    try:
        with client, progress:
            doc_bar = progress.add_task("[cyan]Documents", total=len(tasks))

            for task in tasks:
                if is_cancelled():
                    summary.cancelled = True
                    break

                if task.output_path.exists() and not config.force:
                    summary.skipped_docs += 1
                    logger.info("SKIP (already exists): %s", task.display_name)
                    progress.advance(doc_bar)
                    emit(doc=task.display_name)
                    continue

                total_pages = _total_pages_for_task(task)
                page_bar = progress.add_task(f"  [green]{task.display_name}", total=total_pages)
                cache_dir = _doc_cache_dir(resolved_work, task, config)
                cache_dir.mkdir(parents=True, exist_ok=True)
                emit(doc=task.display_name, page=0, pages=total_pages)

                page_outputs: list[str] = []
                doc_failed_pages = 0

                for page_num, _total, image_b64 in _iter_task_pages(task, config):
                    if is_cancelled():
                        summary.cancelled = True
                        break

                    cache_file = cache_dir / f"page_{page_num:05d}.md"
                    if cache_file.exists() and not config.force:
                        text = cache_file.read_text(encoding="utf-8")
                        logger.info("CACHE HIT: %s page %d/%d", task.display_name, page_num, total_pages)
                    else:
                        t0 = time.perf_counter()
                        try:
                            text = _invoke_ocr(ocr_fn, image_b64, config, client, cancel_event)
                            if is_cancelled():
                                summary.cancelled = True
                                break
                            cache_file.write_text(text, encoding="utf-8")
                            elapsed = time.perf_counter() - t0
                            ocr_durations.append(elapsed)
                            logger.info(
                                "OK: %s page %d/%d (%.2fs)",
                                task.display_name,
                                page_num,
                                total_pages,
                                elapsed,
                            )
                        except JobCancelledError:
                            summary.cancelled = True
                            break
                        except Exception as exc:
                            if is_cancelled():
                                summary.cancelled = True
                                break
                            doc_failed_pages += 1
                            summary.failed_pages += 1
                            logger.error(
                                "FAIL: %s page %d/%d: %s",
                                task.display_name,
                                page_num,
                                total_pages,
                                exc,
                            )
                            text = f"<!-- OCR ERROR on Page {page_num}: {exc} -->"

                    if config.include_page_headers and total_pages > 1:
                        page_outputs.append(f"<!-- Page {page_num} -->\n\n{text}")
                    else:
                        page_outputs.append(text)
                    pages_done_overall += 1
                    progress.advance(page_bar)
                    emit(doc=task.display_name, page=page_num, pages=total_pages)

                progress.remove_task(page_bar)
                if summary.cancelled:
                    logger.warning("CANCELLED during: %s (cached pages preserved)", task.display_name)
                    break

                task.output_path.parent.mkdir(parents=True, exist_ok=True)
                separator = "\n\n---\n\n"
                body = separator.join(page_outputs) + "\n"
                cleaned_tags = normalize_tags(list(config.tags))
                if config.obsidian_mode or cleaned_tags:
                    meta: dict[str, object] = {
                        "title": task.output_path.stem,
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "source": task.display_name,
                        "model": config.model,
                    }
                    if cleaned_tags:
                        meta["tags"] = cleaned_tags
                    if config.obsidian_mode:
                        meta["ocr_pipeline"] = True
                    final_content = format_frontmatter(meta, body)
                else:
                    final_content = body
                task.output_path.write_text(final_content, encoding="utf-8")
                summary.processed_docs += 1
                logger.info(
                    "SAVED: %s -> %s (failed_pages=%d)",
                    task.display_name,
                    task.output_path,
                    doc_failed_pages,
                )
                progress.advance(doc_bar)
                emit(doc=task.display_name, page=total_pages, pages=total_pages)
    finally:
        emit(status="cancelled" if summary.cancelled else "completed")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    return summary
