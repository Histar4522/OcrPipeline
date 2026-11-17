"""PDF rendering and image preprocessing utilities."""

import base64
from collections.abc import Iterator
import io
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps


def prepare_image_b64(source: Path | bytes, max_dim: int = 2048) -> str:
    """Load an image, apply EXIF orientation, downscale if needed, and return base64 PNG."""
    if isinstance(source, Path):
        with Image.open(source) as img:
            return _encode_pil_image(img, max_dim)
    with Image.open(io.BytesIO(source)) as img:
        return _encode_pil_image(img, max_dim)


def _encode_pil_image(img: Image.Image, max_dim: int) -> str:
    oriented = ImageOps.exif_transpose(img)
    rgb = oriented.convert("RGB") if oriented.mode != "RGB" else oriented
    if max_dim > 0 and max(rgb.size) > max_dim:
        rgb = rgb.copy()
        rgb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
