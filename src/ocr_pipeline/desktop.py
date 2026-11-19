"""Cross-platform desktop application shell (pywebview + system tray + OS notifications)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
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
from webview.dom import DOMEventHandler

from .discovery import IMAGE_EXTENSIONS, PDF_EXTENSION, natural_sort_key
from .platform_utils import (
    ensure_app_icon_file,
    ensure_window_icon_file,
    generate_app_icon_image,
    open_in_file_manager,
    open_with_default_app,
    send_toast,
)
from .settings import load_settings, normalize_notification_mode
from .web.server import JobState, create_app

_SUPPORTED_EXTS = IMAGE_EXTENSIONS | {PDF_EXTENSION}
# pywebview parse_file_type only permits [\w ]+ in the description before '('.
_PICK_FILE_TYPES = (
    "PDF and Images (*.pdf;*.png;*.jpg;*.jpeg;*.webp;*.tif;*.tiff;*.bmp)",
    "All files (*.*)",
)


def collect_supported_files(raw_paths: Sequence[str]) -> list[dict]:
    """Expand file/folder paths into deduplicated supported PDF/image metadata."""
    items: list[dict] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            if p.suffix.lower() in _SUPPORTED_EXTS and p not in seen:
                seen.add(p)
                items.append({"name": p.name, "rel": p.name, "path": str(p), "size": p.stat().st_size})
        elif p.is_dir():
            candidates = [
                f
                for f in p.rglob("*")
                if f.is_file()
                and f.suffix.lower() in _SUPPORTED_EXTS
                and not any(part.startswith(".") for part in f.relative_to(p).parts)
            ]
            candidates.sort(key=lambda f: (f.parent.relative_to(p).as_posix().lower(), natural_sort_key(f)))
            for f in candidates:
                resolved = f.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                rel = f"{p.name}/{f.relative_to(p).as_posix()}"
                items.append({"name": f.name, "rel": rel, "path": str(resolved), "size": f.stat().st_size})
    return items


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


def should_send_notification(
    mode: str | bool,
    *,
    focused: bool,
    in_tray: bool,
    minimized: bool,
) -> bool:
    """Return True if a desktop toast should fire for the notification setting and window state."""
    normalized = normalize_notification_mode(mode)
    if normalized == "never":
        return False
    if normalized == "minimized":
        return in_tray or minimized
    if normalized == "not_in_focus":
        return in_tray or minimized or not focused
    return True


class DesktopApi:
    """JavaScript bridge exposed to the pywebview frontend for native OS dialogs."""

    def __init__(
        self,
        base_url: str,
        window_ref: dict[str, webview.Window | None],
        window_state: dict[str, bool] | None = None,
        window_state_lock: threading.Lock | None = None,
    ) -> None:
        self._base_url = base_url
        self._window_ref = window_ref
        self._window_state = window_state
        self._allowed_paths: set[Path] = set()
        self._lock = threading.Lock()
        self._window_state_lock = window_state_lock or self._lock

    def set_focus(self, focused: bool) -> None:
        """Update tracked window focus state from frontend focus/blur events."""
        if self._window_state is not None:
            with self._window_state_lock:
                self._window_state["focused"] = bool(focused)

    def register_paths(self, raw_paths: Sequence[str]) -> list[dict]:
        """Collect supported files from raw paths and allowlist them for open/reveal calls."""
        items = collect_supported_files(raw_paths)
        with self._lock:
            for item in items:
                self._allowed_paths.add(Path(item["path"]))
        return items

    def _pick_paths(self, dialog_type: int, **kwargs: object) -> list[dict]:
        win = self._window_ref.get("win")
        if win is None:
            return []
        result = win.create_file_dialog(dialog_type, **kwargs)
        if not result:
            return []
        paths = list(result) if isinstance(result, (list, tuple)) else [str(result)]
        return self.register_paths(paths)

    def pick_files(self) -> list[dict]:
        """Open a native multi-file dialog for supported PDF and image inputs."""
        return self._pick_paths(webview.FileDialog.OPEN, allow_multiple=True, file_types=_PICK_FILE_TYPES)

    def pick_folder(self) -> list[dict]:
        """Open a native folder dialog and return supported PDF and image inputs inside it."""
        return self._pick_paths(webview.FileDialog.FOLDER)

    def _resolve_allowed(self, path_str: str) -> Path:
        target = Path(path_str).expanduser().resolve()
        with self._lock:
            if target not in self._allowed_paths:
                raise PermissionError("File not in current selection.")
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {target.name}")
        return target

    def open_file(self, path_str: str) -> dict:
        """Open a selected input file with the OS default application."""
        try:
            open_with_default_app(self._resolve_allowed(path_str))
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def reveal_file(self, path_str: str) -> dict:
        """Highlight a selected input file in the native OS file manager."""
        try:
            open_in_file_manager(self._resolve_allowed(path_str))
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

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
    window_state = {"focused": True, "in_tray": False, "minimized": False}
    win_state_lock = threading.Lock()

    def on_job_finished(job: JobState) -> None:
        settings = load_settings()
        with win_state_lock:
            focused = window_state["focused"]
            in_tray = window_state["in_tray"]
            minimized = window_state["minimized"]
        if not should_send_notification(
            settings.desktop_notifications,
            focused=focused,
            in_tray=in_tray,
            minimized=minimized,
        ):
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
        with win_state_lock:
            window_state["in_tray"] = False
            window_state["minimized"] = False
        win = window_ref.get("win")
        if win is not None:
            win.show()
            win.restore()

    def open_archive(_icon: object = None, _item: object = None) -> None:
        archive = Path(load_settings().archive_dir).expanduser()
        archive.mkdir(parents=True, exist_ok=True)
        open_in_file_manager(archive)

    def stop_services() -> None:
        quitting.set()
        tray = tray_ref.get("icon")
        if tray is not None:
            tray.stop()
        server.should_exit = True

    def quit_app(_icon: object = None, _item: object = None) -> None:
        stop_services()
        win = window_ref.get("win")
        if win is not None:
            win.destroy()

    tray_menu = pystray.Menu(
        pystray.MenuItem("Open OcrPipeline", show_window, default=True),
        pystray.MenuItem("Open Archive Folder", open_archive),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )
    tray_icon = pystray.Icon("ocr_pipeline", generate_app_icon_image(64), "OcrPipeline", tray_menu)
    tray_ref["icon"] = tray_icon
    tray_icon.run_detached()

    api = DesktopApi(base_url, window_ref, window_state=window_state, window_state_lock=win_state_lock)
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

    def on_drop(event: dict) -> None:
        raw_files = event.get("dataTransfer", {}).get("files", [])
        paths = [f["pywebviewFullPath"] for f in raw_files if isinstance(f, dict) and f.get("pywebviewFullPath")]
        if not paths:
            return
        items = api.register_paths(paths)
        if items:
            window.evaluate_js(
                f"window.__addNativeSelectedFiles && window.__addNativeSelectedFiles({json.dumps(items)})"
            )

    def on_loaded() -> None:
        drop_el = window.dom.get_element("#dropZone")
        if drop_el is not None:
            drop_el.events.drop += DOMEventHandler(on_drop, prevent_default=True)

    window.events.loaded += on_loaded

    def on_minimized() -> None:
        with win_state_lock:
            window_state["minimized"] = True
            window_state["focused"] = False

    def on_restored() -> None:
        with win_state_lock:
            window_state["minimized"] = False
            window_state["in_tray"] = False

    def on_closing() -> bool:
        if quitting.is_set():
            return True
        if getattr(webview.guilib, "_sigint_received", False) or not load_settings().minimize_to_tray:
            stop_services()
            return True
        with win_state_lock:
            window_state["in_tray"] = True
            window_state["focused"] = False
        window.hide()
        return False

    window.events.minimized += on_minimized
    window.events.restored += on_restored
    window.events.shown += on_restored
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
