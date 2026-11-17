"""Document discovery and natural sorting for PDFs and image series."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"})
PDF_EXTENSION = ".pdf"
_NATURAL_SPLIT_RE = re.compile(r"(\d+)")


@dataclass(frozen=True, slots=True)
class DocumentTask:
    """Represents one logical document (a PDF or an ordered series of images)."""

    kind: Literal["pdf", "images"]
    source_paths: tuple[Path, ...]
    output_path: Path
    display_name: str


def natural_sort_key(path: Path) -> tuple[int | str, ...]:
    """Return a key for natural sorting (e.g., page2.png before page10.png)."""
    parts = _NATURAL_SPLIT_RE.split(path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _is_pdf(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == PDF_EXTENSION


def discover_documents(
    input_path: Path,
    output_dir: Path,
    series_mode: bool = False,
) -> list[DocumentTask]:
    """Discover PDFs and image-series documents from a file or directory."""
    resolved_input = input_path.resolve()
    resolved_output = output_dir.resolve()

    if not resolved_input.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved_input}")

    if resolved_input.is_file():
        if _is_pdf(resolved_input):
            return [
                DocumentTask(
                    kind="pdf",
                    source_paths=(resolved_input,),
                    output_path=resolved_output / f"{resolved_input.stem}.md",
                    display_name=resolved_input.name,
                )
            ]
        if _is_image(resolved_input):
            return [
                DocumentTask(
                    kind="images",
                    source_paths=(resolved_input,),
                    output_path=resolved_output / f"{resolved_input.stem}.md",
                    display_name=resolved_input.name,
                )
            ]
        raise ValueError(f"Unsupported file type: {resolved_input}")

    if series_mode:
        images = sorted((p for p in resolved_input.iterdir() if _is_image(p)), key=natural_sort_key)
        if not images:
            raise ValueError(f"No supported image files found in {resolved_input}")
        return [
            DocumentTask(
                kind="images",
                source_paths=tuple(images),
                output_path=resolved_output / f"{resolved_input.name}.md",
                display_name=f"{resolved_input.name} ({len(images)} images)",
            )
        ]

    tasks: list[DocumentTask] = []
    _walk_directory(resolved_input, resolved_input, resolved_output, tasks)
    return tasks


def _walk_directory(
    root_dir: Path,
    current_dir: Path,
    output_dir: Path,
    tasks: list[DocumentTask],
) -> None:
    entries = sorted(current_dir.iterdir(), key=natural_sort_key)
    images_here = [p for p in entries if _is_image(p)]
    pdfs_here = [p for p in entries if _is_pdf(p)]
    subdirs_here = [p for p in entries if p.is_dir() and not p.name.startswith(".")]

    rel_dir = current_dir.relative_to(root_dir)

    for pdf_path in pdfs_here:
        rel_pdf = pdf_path.relative_to(root_dir).with_suffix(".md")
        tasks.append(
            DocumentTask(
                kind="pdf",
                source_paths=(pdf_path,),
                output_path=output_dir / rel_pdf,
                display_name=str(pdf_path.relative_to(root_dir)),
            )
        )

    if images_here:
        if rel_dir == Path("."):
            out_file = output_dir / f"{root_dir.name}.md"
            label = f"{root_dir.name} ({len(images_here)} images)"
        else:
            out_file = (output_dir / rel_dir).with_suffix(".md")
            label = f"{rel_dir} ({len(images_here)} images)"
        tasks.append(
            DocumentTask(
                kind="images",
                source_paths=tuple(images_here),
                output_path=out_file,
                display_name=label,
            )
        )

    for subdir in subdirs_here:
        _walk_directory(root_dir, subdir, output_dir, tasks)
