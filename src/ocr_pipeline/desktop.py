"""Cross-platform desktop application shell (pywebview + system tray + OS notifications)."""

import argparse
import os
from pathlib import Path
import socket
import sys
import threading
import time

import httpx
import pystray
import uvicorn
import webview

from .platform_utils import (
    ensure_app_icon_file,
    ensure_window_icon_file,
    generate_app_icon_image,
    open_in_file_manager,
    send_toast,
)
from .settings import load_settings
from .web.server import JobState, create_app


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _wait_for_server(url: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{url}/api/health", timeout=0.5)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for local OcrPipeline server at {url}")


class DesktopApi:
    """JavaScript bridge exposed to the pywebview frontend for native OS dialogs."""

    def __init__(self, base_url: str, window_ref: dict[str, webview.Window | None]) -> None:
        self._base_url = base_url
        self._window_ref = window_ref

    def save_url(self, rel_url: str, default_filename: str = "output.md") -> dict:
        """Open a native Save As dialog and write the fetched endpoint bytes to the chosen file."""
        win = self._window_ref.get("win")
        if win is None:
            return {"saved": False, "error": "Window unavailable"}

        ext = Path(default_filename).suffix.lower()
        file_types = (
            ("ZIP Archive (*.zip)", "All files (*.*)")
            if ext == ".zip"
            else ("Markdown (*.md)", "All files (*.*)")
        )
        result = win.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename=default_filename,
            file_types=file_types,
        )
        if not result:
            return {"saved": False}

        dest = Path(result[0] if isinstance(result, (list, tuple)) else result)
        target_url = rel_url if rel_url.startswith("http") else f"{self._base_url}{rel_url}"
        resp = httpx.get(target_url, timeout=60.0)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return {"saved": True, "path": str(dest)}


def launch_desktop_app(port: int | None = None, ollama_url: str | None = None) -> int:
    """Launch OcrPipeline in a pywebview window with system tray retention and OS toasts."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    host = "127.0.0.1"
    chosen_port = port or _find_free_port(host)
    base_url = f"http://{host}:{chosen_port}"
    icon_path = ensure_app_icon_file()
    window_icon_path = ensure_window_icon_file()
    tray_ref: dict[str, pystray.Icon | None] = {"icon": None}
    quitting = threading.Event()
    window_ref: dict[str, webview.Window | None] = {"win": None}

    def on_job_finished(job: JobState) -> None:
        settings = load_settings()
        if not settings.desktop_notifications:
            return
        if job.status == "completed":
            title = "OCR Job Completed"
            msg = f"{job.label} — {job.completed_docs} doc(s) archived."
        elif job.status == "cancelled":
            title = "OCR Job Cancelled"
            msg = f"{job.label} was stopped."
        else:
            title = "OCR Job Failed"
            msg = f"{job.label}: {job.error or 'Check logs for details.'}"

        if not send_toast(title, msg, icon_path=icon_path):
            tray = tray_ref.get("icon")
            if tray is not None:
                try:
                    tray.notify(msg, title)
                except Exception:
                    pass

    app = create_app(ollama_url=ollama_url, on_job_finished=on_job_finished)
    uv_config = uvicorn.Config(app, host=host, port=chosen_port, log_level="warning", log_config=None)
    server = uvicorn.Server(uv_config)
    threading.Thread(target=server.run, daemon=True).start()
    _wait_for_server(base_url)

    def show_window(_icon: object = None, _item: object = None) -> None:
        win = window_ref.get("win")
        if win is not None:
            win.show()
            win.restore()

    def open_archive(_icon: object = None, _item: object = None) -> None:
        archive = Path(load_settings().archive_dir).expanduser()
        archive.mkdir(parents=True, exist_ok=True)
        open_in_file_manager(archive)

    def quit_app(_icon: object = None, _item: object = None) -> None:
        quitting.set()
        tray = tray_ref.get("icon")
        if tray is not None:
            tray.stop()
        win = window_ref.get("win")
        if win is not None:
            win.destroy()
        server.should_exit = True

    tray_menu = pystray.Menu(
        pystray.MenuItem("Open OcrPipeline", show_window, default=True),
        pystray.MenuItem("Open Archive Folder", open_archive),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )
    tray_icon = pystray.Icon("ocr_pipeline", generate_app_icon_image(64), "OcrPipeline", tray_menu)
    tray_ref["icon"] = tray_icon
    tray_icon.run_detached()

    api = DesktopApi(base_url, window_ref)
    window = webview.create_window(
        "OcrPipeline",
        base_url,
        js_api=api,
        width=1280,
        height=840,
        min_size=(960, 640),
        text_select=True,
    )
    window_ref["win"] = window

    def on_closing() -> bool:
        if quitting.is_set():
            return True
        if load_settings().minimize_to_tray:
            window.hide()
            return False
        quit_app()
        return True

    window.events.closing += on_closing

    try:
        webview.start(icon=str(window_icon_path))
    finally:
        if not quitting.is_set():
            quit_app()

    return 0


def main(argv: list[str] | None = None) -> int:
    """GUI script entry point for OcrPipeline."""
    parser = argparse.ArgumentParser(prog="ocr-pipeline-app", description="Launch OcrPipeline Desktop App.")
    parser.add_argument("--port", type=int, default=None, help="Local port (default: auto free port).")
    parser.add_argument("--ollama-url", default=None, help="Override Ollama URL.")
    args = parser.parse_args(argv)
    return launch_desktop_app(port=args.port, ollama_url=args.ollama_url)


if __name__ == "__main__":
    sys.exit(main())
