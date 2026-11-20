"""Cross-platform desktop application shell (pywebview + system tray + OS notifications)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import secrets
import socket
import sys
import threading
import time

import httpx
from platformdirs import user_data_dir
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


def default_instance_lock_dir() -> Path:
    """Return the per-user directory used for single-instance lock files."""
    return Path(user_data_dir("OcrPipeline", appauthor=False))


class SingleInstanceGuard:
    """Cross-platform non-blocking OS file lock ensuring one desktop instance per user."""

    def __init__(self, lock_dir: Path | None = None) -> None:
        self.lock_dir = (lock_dir or default_instance_lock_dir()).expanduser().resolve()
        self.lock_file = self.lock_dir / "instance.lock"
        self.info_file = self.lock_dir / "instance.json"
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Attempt to acquire the exclusive instance lock without blocking."""
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        fd = os.open(self.lock_file, flags, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"1")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fd = fd
            return True
        except OSError:
            os.close(fd)
            return False

    def write_info(self, port: int, token: str) -> None:
        """Publish the active server port and activation secret for duplicate launches."""
        payload = {"pid": os.getpid(), "port": int(port), "token": token}
        tmp_path = self.info_file.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(self.info_file)

    def read_info(self) -> dict | None:
        """Read metadata for the currently locked desktop instance, if available."""
        try:
            data = json.loads(self.info_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        """Release the OS lock and clean up instance metadata on shutdown."""
        if self._fd is None:
            return
        try:
            self.info_file.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


def notify_existing_instance(
    guard: SingleInstanceGuard,
    *,
    title: str = "Already Running",
    message: str = "OcrPipeline is already running. Multiple instances are not allowed.",
    variant: str = "danger",
    wait_seconds: float = 2.5,
) -> bool:
    """Signal an already-running desktop instance to raise its window and show an in-app toast."""
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        info = guard.read_info()
        port = int(info.get("port") or 0) if info else 0
        token = str(info.get("token") or "") if info else ""
        if port > 0 and token:
            try:
                resp = httpx.post(
                    f"http://127.0.0.1:{port}/api/desktop/activate",
                    json={"token": token, "title": title, "message": message, "variant": variant},
                    timeout=2.0,
                )
                if resp.status_code == 200 and resp.json().get("ok"):
                    return True
            except (httpx.HTTPError, ValueError):
                pass
        time.sleep(0.1)
    return False


def _show_fallback_error_alert(title: str, message: str) -> None:
    """Show a native error dialog as fallback if the existing instance did not respond."""
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, title, 0x00000010)
        except Exception:
            pass


def launch_desktop_app(
    port: int | None = None,
    ollama_url: str | None = None,
    lock_dir: Path | None = None,
) -> int:
    """Launch OcrPipeline in a pywebview window with single-instance lock, tray, and in-app activation."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    guard = SingleInstanceGuard(lock_dir)
    if not guard.acquire():
        activated = notify_existing_instance(guard)
        sys.stderr.write("Error: OcrPipeline is already running.\n")
        sys.stderr.flush()
        if not activated:
            _show_fallback_error_alert(
                "OcrPipeline Already Running",
                "Another instance of OcrPipeline is already running.",
            )
        return 1

    host = "127.0.0.1"
    chosen_port = port or _find_free_port(host)
    base_url = f"http://{host}:{chosen_port}"
    activate_token = secrets.token_urlsafe(24)
    icon_path = ensure_app_icon_file()
    window_icon_path = ensure_window_icon_file()
    tray_ref: dict[str, pystray.Icon | None] = {"icon": None}
    quitting = threading.Event()
    page_loaded = threading.Event()
    window_ref: dict[str, webview.Window | None] = {"win": None}
    window_state = {"focused": True, "in_tray": False, "minimized": False}
    pending_toasts: list[tuple[str, str, str]] = []
    win_state_lock = threading.Lock()

    try:

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

        def show_window(_icon: object = None, _item: object = None) -> None:
            with win_state_lock:
                was_hidden_or_min = window_state["in_tray"] or window_state["minimized"]
                window_state["in_tray"] = False
                window_state["minimized"] = False
                window_state["focused"] = True
            win = window_ref.get("win")
            if win is not None:
                win.show()
                if was_hidden_or_min:
                    win.restore()
                try:
                    win.on_top = True
                    win.on_top = False
                except Exception:
                    pass

        def _emit_window_toast(win: webview.Window, title: str, message: str, variant: str) -> None:
            win.evaluate_js(
                f"window.__showAppToast && window.__showAppToast({json.dumps(title)}, {json.dumps(message)}, {json.dumps(variant)})"
            )

        def on_activate(title: str, message: str, variant: str) -> bool:
            show_window()
            win = window_ref.get("win")
            if win is not None and page_loaded.is_set():
                try:
                    _emit_window_toast(win, title, message, variant)
                    return True
                except Exception:
                    pass
            with win_state_lock:
                pending_toasts.append((title, message, variant))
            return True

        app = create_app(
            ollama_url=ollama_url,
            on_job_finished=on_job_finished,
            on_activate=on_activate,
            activate_token=activate_token,
        )
        uv_config = uvicorn.Config(app, host=host, port=chosen_port, log_level="warning", log_config=None)
        server = uvicorn.Server(uv_config)
        threading.Thread(target=server.run, daemon=True).start()
        _wait_for_server(base_url)
        guard.write_info(chosen_port, activate_token)

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
            page_loaded.set()
            drop_el = window.dom.get_element("#dropZone")
            if drop_el is not None:
                drop_el.events.drop += DOMEventHandler(on_drop, prevent_default=True)
            with win_state_lock:
                queued = list(pending_toasts)
                pending_toasts.clear()
            for title, msg, variant in queued:
                try:
                    _emit_window_toast(window, title, msg, variant)
                except Exception:
                    pass

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
    finally:
        guard.release()

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
