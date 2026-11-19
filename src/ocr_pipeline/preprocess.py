"""PDF rendering and image preprocessing utilities."""

import base64
from collections.abc import Iterator
from dataclasses import dataclass
import io
from pathlib import Path
import time

import pymupdf
from PIL import Image, ImageOps


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Base64-encoded PNG plus dimensions, encoded byte size, and prep duration."""

    b64: str
    width: int
    height: int
    size_bytes: int
    prep_seconds: float


def prepare_image(source: Path | bytes, max_dim: int = 2048) -> PreparedImage:
    """Load an image, apply EXIF orientation, downscale if needed, and return PreparedImage."""
    t0 = time.perf_counter()
    if isinstance(source, Path):
        with Image.open(source) as img:
            b64, width, height, size_bytes = _encode_pil_image(img, max_dim)
    else:
        with Image.open(io.BytesIO(source)) as img:
            b64, width, height, size_bytes = _encode_pil_image(img, max_dim)
    return PreparedImage(
        b64=b64,
        width=width,
        height=height,
        size_bytes=size_bytes,
        prep_seconds=time.perf_counter() - t0,
    )


def prepare_image_b64(source: Path | bytes, max_dim: int = 2048) -> str:
    """Load an image, apply EXIF orientation, downscale if needed, and return base64 PNG."""
    return prepare_image(source, max_dim=max_dim).b64


def _encode_pil_image(img: Image.Image, max_dim: int) -> tuple[str, int, int, int]:
    oriented = ImageOps.exif_transpose(img)
    rgb = oriented.convert("RGB") if oriented.mode != "RGB" else oriented
    if max_dim > 0 and max(rgb.size) > max_dim:
        rgb = rgb.copy()
        rgb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="PNG", compress_level=1)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), rgb.width, rgb.height, len(raw)


def render_pdf_page(
    doc: pymupdf.Document,
    page_index: int,
    dpi: int = 220,
    max_dim: int = 1536,
) -> PreparedImage:
    """Render one PDF page directly at target DPI capped by max_dim without Pillow roundtrip."""
    t0 = time.perf_counter()
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    max_side = max(page.rect.width, page.rect.height)
    if max_dim > 0 and max_side > 0 and max_side * zoom > max_dim:
        zoom = max_dim / max_side
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    png_bytes = pix.tobytes("png")
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return PreparedImage(
        b64=b64,
        width=pix.width,
        height=pix.height,
        size_bytes=len(png_bytes),
        prep_seconds=time.perf_counter() - t0,
    )


def count_pdf_pages(pdf_path: Path) -> int:
    """Return the number of pages in a PDF."""
    with pymupdf.open(pdf_path) as doc:
        return doc.page_count


def iter_pdf_pages_png(pdf_path: Path, dpi: int = 220) -> Iterator[tuple[int, int, bytes]]:
    """Yield (1-based page_number, total_pages, png_bytes) for each page in a PDF."""
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    with pymupdf.open(pdf_path) as doc:
        total = doc.page_count
        for idx in range(total):
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            yield idx + 1, total, pix.tobytes("png")
