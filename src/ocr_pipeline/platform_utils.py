"""Cross-platform OS integration helpers (file manager, notifications, app icon)."""

import os
from pathlib import Path
import platform
import subprocess

from notifypy import Notify
from PIL import Image, ImageDraw
from platformdirs import user_data_dir


def open_in_file_manager(path: Path) -> None:
    """Open a file or directory in the native OS file manager."""
    target = path.resolve()
    if not target.exists():
        target = target.parent
    system = platform.system()
    if system == "Windows":
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


def generate_app_icon_image(size: int = 128) -> Image.Image:
    """Generate a crisp, minimal app icon image using Pillow (no external asset files needed)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = size // 12
    draw.rounded_rectangle(
        (pad, pad, size - pad, size - pad),
        radius=size // 5,
        fill=(13, 110, 253, 255),
    )

    doc_left = int(size * 0.28)
    doc_top = int(size * 0.22)
    doc_right = int(size * 0.72)
    doc_bottom = int(size * 0.78)
    draw.rounded_rectangle(
        (doc_left, doc_top, doc_right, doc_bottom),
        radius=size // 16,
        fill=(255, 255, 255, 255),
    )

    line_x1 = int(size * 0.35)
    line_x2 = int(size * 0.65)
    stroke = max(2, size // 24)
    for frac in (0.38, 0.50, 0.62):
        y = int(size * frac)
        end_x = line_x2 if frac < 0.60 else int(size * 0.55)
        draw.line((line_x1, y, end_x, y), fill=(13, 110, 253, 255), width=stroke)

    return img


def ensure_app_icon_file(icon_dir: Path | None = None) -> Path:
    """Write the generated app icon PNG to user data dir and return its path."""
    target_dir = icon_dir or Path(user_data_dir("OcrPipeline", appauthor=False))
    target_dir.mkdir(parents=True, exist_ok=True)
    icon_path = target_dir / "app_icon.png"
    if not icon_path.exists():
        generate_app_icon_image(128).save(icon_path, format="PNG")
    return icon_path


def send_toast(title: str, message: str, icon_path: Path | None = None) -> bool:
    """Send a cross-platform desktop toast notification without raising on failure."""
    try:
        notification = Notify(default_notification_application_name="OcrPipeline")
        notification.title = title
        notification.message = message
        if icon_path and icon_path.exists():
            notification.icon = str(icon_path)
        notification.send(block=False)
        return True
    except Exception:
        return False
