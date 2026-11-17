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
