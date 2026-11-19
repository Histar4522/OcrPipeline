"""Batch OCR runner with page-level checkpointing, resume, cancellation, and logging."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
import hashlib
import inspect
import logging
from pathlib import Path
import queue
import threading
import time

import httpx
import pymupdf
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .artifacts import format_frontmatter, normalize_tags, update_artifact_tags
from .config import OcrConfig
from .discovery import DocumentTask
from .ollama_client import JobCancelledError
from .preprocess import PreparedImage, count_pdf_pages, prepare_image, render_pdf_page
from .providers import vision_ocr_with_config

OcrFn = Callable[..., str]
# ponytail: per-document prefetch (2 pages); add cross-document prefetch only if multi-1-page-PDF batches show stalls
_PREFETCH_PAGES = 2
_DONE = object()


@dataclass(slots=True)
class _PrefetchedPage:
    page_num: int
    total_pages: int
    image: PreparedImage | None = None
    cached: bool = False
    error: Exception | None = None
    fatal: bool = False


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


def _format_ollama_metrics(metrics: dict[str, object]) -> str:
    def num(key: str) -> float | None:
        val = metrics.get(key)
        return float(val) if isinstance(val, (int, float)) else None

    parts: list[str] = []
    total_ns = num("total_duration")
    load_ns = num("load_duration")
    prefill_ns = num("prompt_eval_duration")
    prefill_tok = num("prompt_eval_count")
    eval_ns = num("eval_duration")
    eval_tok = num("eval_count")

    if total_ns is not None:
        parts.append(f"total={total_ns / 1e9:.2f}s")
    if load_ns is not None:
        parts.append(f"load={load_ns / 1e9:.2f}s")
    if prefill_ns is not None:
        tok_str = f"({int(prefill_tok)}tok)" if prefill_tok is not None else ""
        parts.append(f"prefill={prefill_ns / 1e9:.2f}s{tok_str}")
    if eval_ns is not None:
        eval_s = eval_ns / 1e9
        if eval_tok is not None and eval_s > 0:
            parts.append(f"gen={eval_s:.2f}s({int(eval_tok)}tok, {eval_tok / eval_s:.1f}tok/s)")
        else:
            parts.append(f"gen={eval_s:.2f}s")
    if not parts:
        return ""
    done_reason = metrics.get("done_reason")
    if isinstance(done_reason, str) and done_reason and done_reason != "stop":
        parts.append(f"stop={done_reason}")
    return "ollama: " + " ".join(parts)


def _iter_prefetched_pages(
    task: DocumentTask,
    config: OcrConfig,
    cache_dir: Path,
    total_pages: int,
    cancel_event: threading.Event | None,
    logger: logging.Logger,
) -> Iterator[tuple[_PrefetchedPage, float]]:
    """Render/preprocess upcoming pages on a background thread while GPU runs OCR."""
    out_q: queue.Queue[object] = queue.Queue(maxsize=_PREFETCH_PAGES)
    stop_event = threading.Event()

    def should_stop() -> bool:
        return stop_event.is_set() or (cancel_event is not None and cancel_event.is_set())

    def put_item(item: object) -> bool:
        while not should_stop():
            try:
                out_q.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def handle_page(page_num: int, render_fn: Callable[[], PreparedImage]) -> bool:
        if should_stop():
            return False
        cache_file = cache_dir / f"page_{page_num:05d}.md"
        if cache_file.exists() and not config.force:
            return put_item(_PrefetchedPage(page_num=page_num, total_pages=total_pages, cached=True))
        try:
            img = render_fn()
            logger.info(
                "PREP: %s page %d/%d (%dx%d, %.0f KB, prep=%.2fs)",
                task.display_name,
                page_num,
                total_pages,
                img.width,
                img.height,
                img.size_bytes / 1024.0,
                img.prep_seconds,
            )
            item = _PrefetchedPage(page_num=page_num, total_pages=total_pages, image=img)
        except Exception as exc:
            item = _PrefetchedPage(page_num=page_num, total_pages=total_pages, error=exc)
        return put_item(item)

    def producer() -> None:
        try:
            if task.kind == "pdf":
                try:
                    doc_ctx = pymupdf.open(task.source_paths[0])
                except Exception as exc:
                    put_item(_PrefetchedPage(page_num=0, total_pages=total_pages, error=exc, fatal=True))
                    return
                with doc_ctx as doc:
                    for idx in range(doc.page_count):
                        ok = handle_page(
                            idx + 1,
                            lambda i=idx: render_pdf_page(
                                doc, i, dpi=config.dpi, max_dim=config.max_image_dim
                            ),
                        )
                        if not ok:
                            break
            else:
                for idx, img_path in enumerate(task.source_paths):
                    ok = handle_page(
                        idx + 1,
                        lambda p=img_path: prepare_image(p, max_dim=config.max_image_dim),
                    )
                    if not ok:
                        break
        finally:
            put_item(_DONE)

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()
    try:
        while True:
            t_wait = time.perf_counter()
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return
                try:
                    item = out_q.get(timeout=0.1)
                    break
                except queue.Empty:
                    continue
            wait_s = time.perf_counter() - t_wait
            if item is _DONE:
                return
            assert isinstance(item, _PrefetchedPage)
            yield item, wait_s
    finally:
        stop_event.set()
        while True:
            try:
                out_q.get_nowait()
            except queue.Empty:
                break
        worker.join(timeout=5.0)


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
    metrics_out: dict[str, object] | None = None,
) -> str:
    params = inspect.signature(ocr_fn).parameters
    kwargs: dict[str, object] = {}
    if "cancel_event" in params:
        kwargs["cancel_event"] = cancel_event
    if "metrics_out" in params:
        kwargs["metrics_out"] = metrics_out
    return ocr_fn(image_b64, config, client, **kwargs)


def run_pipeline(
    tasks: list[DocumentTask],
    output_dir: Path,
    config: OcrConfig,
    ocr_fn: OcrFn = vision_ocr_with_config,
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
        "Starting OCR run: docs=%d provider=%s model=%s dpi=%d max_dim=%d num_ctx=%d force=%s prefetch=%d",
        len(tasks),
        config.provider,
        config.model,
        config.dpi,
        config.max_image_dim,
        config.num_ctx,
        config.force,
        _PREFETCH_PAGES,
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
                logger.info("DOC START: %s (kind=%s, pages=%d)", task.display_name, task.kind, total_pages)
                emit(doc=task.display_name, page=0, pages=total_pages)

                doc_start = time.perf_counter()
                doc_wait_total = 0.0
                doc_ocr_durations: list[float] = []
                page_outputs: list[str] = []
                doc_failed_pages = 0

                for prefetched, wait_s in _iter_prefetched_pages(
                    task, config, cache_dir, total_pages, cancel_event, logger
                ):
                    if is_cancelled():
                        summary.cancelled = True
                        break
                    if prefetched.fatal and prefetched.error is not None:
                        raise prefetched.error

                    page_num = prefetched.page_num
                    cache_file = cache_dir / f"page_{page_num:05d}.md"
                    if prefetched.cached:
                        text = cache_file.read_text(encoding="utf-8")
                        logger.info("CACHE HIT: %s page %d/%d", task.display_name, page_num, total_pages)
                    elif prefetched.error is not None or prefetched.image is None:
                        exc = prefetched.error or RuntimeError("Failed to prepare page image.")
                        doc_failed_pages += 1
                        summary.failed_pages += 1
                        logger.error(
                            "FAIL: %s page %d/%d (prep): %s",
                            task.display_name,
                            page_num,
                            total_pages,
                            exc,
                        )
                        text = f"<!-- OCR ERROR on Page {page_num}: {exc} -->"
                    else:
                        image = prefetched.image
                        doc_wait_total += wait_s
                        logger.info(
                            "OCR START: %s page %d/%d (wait=%.2fs, %dx%d, %.0f KB)",
                            task.display_name,
                            page_num,
                            total_pages,
                            wait_s,
                            image.width,
                            image.height,
                            image.size_bytes / 1024.0,
                        )
                        t0 = time.perf_counter()
                        ollama_metrics: dict[str, object] = {}
                        try:
                            text = _invoke_ocr(
                                ocr_fn,
                                image.b64,
                                config,
                                client,
                                cancel_event,
                                metrics_out=ollama_metrics,
                            )
                            if is_cancelled():
                                summary.cancelled = True
                                break
                            cache_file.write_text(text, encoding="utf-8")
                            elapsed = time.perf_counter() - t0
                            ocr_durations.append(elapsed)
                            doc_ocr_durations.append(elapsed)
                            metrics_str = _format_ollama_metrics(ollama_metrics)
                            suffix = f" | {metrics_str}" if metrics_str else ""
                            logger.info(
                                "OK: %s page %d/%d (ocr=%.2fs, chars=%d%s)",
                                task.display_name,
                                page_num,
                                total_pages,
                                elapsed,
                                len(text),
                                suffix,
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
                                "FAIL: %s page %d/%d (%.2fs): %s",
                                task.display_name,
                                page_num,
                                total_pages,
                                time.perf_counter() - t0,
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

                if is_cancelled():
                    summary.cancelled = True
                progress.remove_task(page_bar)
                if summary.cancelled:
                    logger.warning("CANCELLED during: %s (cached pages preserved)", task.display_name)
                    break

                task.output_path.parent.mkdir(parents=True, exist_ok=True)
                separator = "\n\n---\n\n"
                body = separator.join(page_outputs) + "\n"
                cleaned_tags = normalize_tags(list(config.tags))
                if config.obsidian_mode:
                    meta: dict[str, object] = {
                        "title": task.output_path.stem,
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "source": task.display_name,
                        "model": config.model,
                        "ocr_pipeline": True,
                    }
                    final_content = format_frontmatter(meta, body)
                else:
                    final_content = body
                task.output_path.write_text(final_content, encoding="utf-8")
                if cleaned_tags:
                    update_artifact_tags(task.output_path, cleaned_tags)
                summary.processed_docs += 1
                doc_elapsed = time.perf_counter() - doc_start
                avg_ocr = sum(doc_ocr_durations) / len(doc_ocr_durations) if doc_ocr_durations else 0.0
                logger.info(
                    "SAVED: %s -> %s (pages=%d, failed_pages=%d, total=%.2fs, avg_ocr=%.2fs, total_wait=%.2fs)",
                    task.display_name,
                    task.output_path,
                    total_pages,
                    doc_failed_pages,
                    doc_elapsed,
                    avg_ocr,
                    doc_wait_total,
                )
                progress.advance(doc_bar)
                emit(doc=task.display_name, page=total_pages, pages=total_pages)
    finally:
        logger.info(
            "Finished OCR run: processed=%d skipped=%d failed_pages=%d cancelled=%s elapsed=%.2fs",
            summary.processed_docs,
            summary.skipped_docs,
            summary.failed_pages,
            summary.cancelled,
            time.monotonic() - run_start,
        )
        emit(status="cancelled" if summary.cancelled else "completed")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    return summary
