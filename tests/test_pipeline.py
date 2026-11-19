"""Unit tests for the local AI OCR pipeline."""

from pathlib import Path

import pymupdf
from PIL import Image

from ocr_pipeline.config import OcrConfig
from ocr_pipeline.discovery import discover_documents, natural_sort_key
from ocr_pipeline.runner import run_pipeline


def _make_png(path: Path, size: tuple[int, int] = (200, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path, format="PNG")


def _make_pdf(path: Path, num_pages: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for idx in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Sample page {idx + 1}")
    doc.save(path)
    doc.close()


def test_natural_sort_orders_numeric_filenames() -> None:
    paths = [Path("page_10.png"), Path("page_2.png"), Path("page_1.png")]
    ordered = [p.name for p in sorted(paths, key=natural_sort_key)]
    assert ordered == ["page_1.png", "page_2.png", "page_10.png"]


def test_discover_series_and_batch_modes(tmp_path: Path) -> None:
    series_dir = tmp_path / "notebook"
    _make_png(series_dir / "img_10.png")
    _make_png(series_dir / "img_2.png")
    _make_pdf(tmp_path / "sub" / "report.pdf", num_pages=2)
    out_dir = tmp_path / "out"

    series_tasks = discover_documents(series_dir, out_dir, series_mode=True)
    assert len(series_tasks) == 1
    assert [p.name for p in series_tasks[0].source_paths] == ["img_2.png", "img_10.png"]
    assert series_tasks[0].output_path == out_dir.resolve() / "notebook.md"

    batch_tasks = discover_documents(tmp_path, out_dir, series_mode=False)
    assert len(batch_tasks) == 2
    kinds = {t.kind for t in batch_tasks}
    assert kinds == {"images", "pdf"}


def test_runner_combines_pages_and_resumes(tmp_path: Path) -> None:
    pdf_path = tmp_path / "docs" / "notes.pdf"
    _make_pdf(pdf_path, num_pages=3)
    out_dir = tmp_path / "archive"
    tasks = discover_documents(pdf_path, out_dir)
    config = OcrConfig(dpi=72, max_image_dim=512)

    calls: list[int] = []

    def fake_ocr(image_b64: str, cfg: OcrConfig, client: object) -> str:
        assert len(image_b64) > 20
        calls.append(1)
        return f"Transcribed block #{len(calls)}"

    summary1 = run_pipeline(tasks, out_dir, config, ocr_fn=fake_ocr)
    assert summary1.processed_docs == 1
    assert summary1.skipped_docs == 0
    assert summary1.failed_pages == 0
    assert len(calls) == 3

    md_text = (out_dir / "notes.md").read_text(encoding="utf-8")
    assert "<!-- Page 1 -->" in md_text
    assert "Transcribed block #1" in md_text
    assert "Transcribed block #3" in md_text
    assert (out_dir / "ocr_pipeline.log").exists()

    # Second run without --force should skip the already-archived document
    summary2 = run_pipeline(tasks, out_dir, config, ocr_fn=fake_ocr)
    assert summary2.processed_docs == 0
    assert summary2.skipped_docs == 1
    assert len(calls) == 3


def test_render_pdf_page_caps_dimensions_and_prefetch_overlaps_and_logs(tmp_path: Path) -> None:
    import time

    from ocr_pipeline.preprocess import render_pdf_page

    pdf_path = tmp_path / "prefetch.pdf"
    _make_pdf(pdf_path, num_pages=3)
    with pymupdf.open(pdf_path) as doc:
        capped = render_pdf_page(doc, 0, dpi=300, max_dim=512)
        assert max(capped.width, capped.height) == 512
        assert capped.size_bytes > 0
        assert capped.prep_seconds >= 0.0

    out_dir = tmp_path / "archive"
    tasks = discover_documents(pdf_path, out_dir)
    config = OcrConfig(dpi=150, max_image_dim=1024)

    def fake_ocr(
        image_b64: str,
        cfg: OcrConfig,
        client: object,
        metrics_out: dict[str, object] | None = None,
    ) -> str:
        assert len(image_b64) > 20
        time.sleep(0.12)
        if metrics_out is not None:
            metrics_out.update(
                {
                    "total_duration": 120_000_000,
                    "load_duration": 10_000_000,
                    "prompt_eval_count": 512,
                    "prompt_eval_duration": 40_000_000,
                    "eval_count": 64,
                    "eval_duration": 50_000_000,
                    "done_reason": "stop",
                }
            )
        return "Transcribed page"

    summary = run_pipeline(tasks, out_dir, config, ocr_fn=fake_ocr)
    assert summary.processed_docs == 1
    assert summary.failed_pages == 0

    log_lines = (out_dir / "ocr_pipeline.log").read_text(encoding="utf-8").splitlines()
    log_text = "\n".join(log_lines)
    for marker in ("DOC START:", "PREP:", "OCR START:", "OK:", "ollama:", "tok/s", "SAVED:", "Finished OCR run:"):
        assert marker in log_text

    prep2_idx = next(i for i, line in enumerate(log_lines) if "PREP:" in line and "page 2/3" in line)
    ok1_idx = next(i for i, line in enumerate(log_lines) if "OK:" in line and "page 1/3" in line)
    assert prep2_idx < ok1_idx


def test_cache_hit_skips_pdf_rendering(tmp_path: Path, monkeypatch: object) -> None:
    import ocr_pipeline.runner as runner_mod

    pdf_path = tmp_path / "cached.pdf"
    _make_pdf(pdf_path, num_pages=2)
    out_dir = tmp_path / "archive"
    tasks = discover_documents(pdf_path, out_dir)
    config = OcrConfig(dpi=72, max_image_dim=512)

    cache_dir = runner_mod._doc_cache_dir(out_dir.resolve(), tasks[0], config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "page_00001.md").write_text("Cached page 1", encoding="utf-8")

    rendered_indices: list[int] = []
    real_render = runner_mod.render_pdf_page

    def tracked_render(doc: pymupdf.Document, page_index: int, dpi: int = 220, max_dim: int = 1536):
        rendered_indices.append(page_index)
        return real_render(doc, page_index, dpi=dpi, max_dim=max_dim)

    monkeypatch.setattr(runner_mod, "render_pdf_page", tracked_render)  # type: ignore[attr-defined]
    summary = run_pipeline(tasks, out_dir, config, ocr_fn=lambda b64, cfg, c: "Live page 2")
    assert summary.processed_docs == 1
    assert rendered_indices == [1]

    md_text = (out_dir / "cached.md").read_text(encoding="utf-8")
    assert "Cached page 1" in md_text
    assert "Live page 2" in md_text


def test_ocr_image_b64_populates_metrics_out() -> None:
    import httpx

    from ocr_pipeline.ollama_client import ocr_image_b64

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": "  Hello OCR  "},
                "total_duration": 2_000_000_000,
                "load_duration": 100_000_000,
                "prompt_eval_count": 300,
                "prompt_eval_duration": 400_000_000,
                "eval_count": 150,
                "eval_duration": 1_500_000_000,
                "done_reason": "stop",
            },
        )

    metrics: dict[str, object] = {}
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        text = ocr_image_b64("QUJD", OcrConfig(), client=client, metrics_out=metrics)

    assert text == "Hello OCR"
    assert metrics["total_duration"] == 2_000_000_000
    assert metrics["prompt_eval_count"] == 300
    assert metrics["eval_count"] == 150
    assert metrics["done_reason"] == "stop"


def test_prefetch_cancellation_stops_cleanly(tmp_path: Path) -> None:
    import threading

    pdf_path = tmp_path / "cancel.pdf"
    _make_pdf(pdf_path, num_pages=5)
    out_dir = tmp_path / "archive"
    tasks = discover_documents(pdf_path, out_dir)
    config = OcrConfig(dpi=72, max_image_dim=512)

    started = threading.Event()
    cancel_event = threading.Event()

    def slow_ocr(
        image_b64: str,
        cfg: OcrConfig,
        client: object,
        cancel_event: threading.Event | None = None,
    ) -> str:
        started.set()
        if cancel_event is not None:
            cancel_event.wait(timeout=3.0)
        return "partial"

    def trigger_cancel() -> None:
        if started.wait(timeout=2.0):
            cancel_event.set()

    canceller = threading.Thread(target=trigger_cancel, daemon=True)
    canceller.start()
    summary = run_pipeline(tasks, out_dir, config, ocr_fn=slow_ocr, cancel_event=cancel_event)
    canceller.join(timeout=2.0)

    assert summary.cancelled is True
    # Producer thread must have closed PyMuPDF handle cleanly so file unlink succeeds on Windows
    pdf_path.unlink()
    assert not pdf_path.exists()
