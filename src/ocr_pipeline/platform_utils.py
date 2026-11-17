"""Cross-platform OS integration helpers (file manager, notifications, app icon)."""

import os
from pathlib import Path
import platform
import shutil
import subprocess

from notifypy import Notify
from PIL import Image, ImageDraw
from platformdirs import user_data_dir
import psutil


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


def ensure_window_icon_file(icon_dir: Path | None = None) -> Path:
    """Return a platform-appropriate window icon (.ico on Windows, .png elsewhere)."""
    if platform.system() != "Windows":
        return ensure_app_icon_file(icon_dir)
    target_dir = icon_dir or Path(user_data_dir("OcrPipeline", appauthor=False))
    target_dir.mkdir(parents=True, exist_ok=True)
    ico_path = target_dir / "app_icon.ico"
    if not ico_path.exists():
        generate_app_icon_image(128).save(
            ico_path,
            format="ICO",
            sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128)],
        )
    return ico_path


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


def get_system_info() -> dict:
    """Collect OS, CPU, RAM, and optional NVIDIA GPU/VRAM info for the Setup Wizard."""
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    gpu_name: str | None = None
    vram_gb: float | None = None

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": 2.0,
        }
        if platform.system() == "Windows":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                **kwargs,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                line = proc.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    gpu_name = parts[0]
                    vram_gb = round(float(parts[1]) / 1024.0, 1)
        except Exception:
            pass

    if vram_gb is not None and vram_gb <= 8.5:
        vram_note = f"{vram_gb} GB VRAM detected — glm-ocr (~2.2 GB) fits cleanly without shared-memory spill."
    elif vram_gb is not None:
        vram_note = f"{vram_gb} GB VRAM detected — supports both glm-ocr and larger VL models."
    else:
        vram_note = "No dedicated NVIDIA GPU detected — glm-ocr (~0.9B) is strongly recommended for CPU/unified memory."

    return {
        "os": f"{platform.system()} {platform.release()}",
        "system": platform.system(),
        "cpu_count": os.cpu_count() or 1,
        "ram_gb": ram_gb,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "recommended_model": "glm-ocr",
        "vram_note": vram_note,
    }
