"""FastAPI backend server for OcrPipeline Desktop & Web App."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import io
import json
import mimetypes
from pathlib import Path
import platform
import queue
import shutil
import sys
import threading
import uuid
import zipfile

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
from rich.console import Console

from . import __name__ as _pkg  # noqa: F401
from .. import __version__, get_display_version
from ..artifacts import (
    delete_artifact_sidecar,
    delete_folder,
    ensure_obsidian_vault,
    move_artifact_sidecar,
    normalize_tags,
    rename_folder,
    sanitize_rel_folder,
    scan_artifacts,
    update_artifact_summary,
    update_artifact_tags,
)
from ..config import DEFAULT_SYSTEM_PROMPT, OcrConfig
from ..discovery import IMAGE_EXTENSIONS, PDF_EXTENSION, DocumentTask, discover_documents
from ..llm_context import calculate_vram_plan
from ..llm_features import (
    analyze_artifact_all,
    cleanup_ocr_markdown,
    suggest_filename,
    suggest_tags,
    summarize_document,
    summarize_folder_contents,
)
from ..ollama_client import (
    OllamaError,
    ProviderError,
    check_ollama_ready,
    delete_model,
    find_ollama_executable,
    list_models_detailed,
    pull_model,
    pull_model_stream,
    start_ollama_service,
)
from ..platform_utils import generate_app_icon_image, get_system_info, open_in_file_manager
from ..providers import (
    PROVIDER_REGISTRY,
    ProviderEndpoint,
    STATUS_LABELS,
    check_endpoint_ready,
    classify_exception_status,
    list_providers_metadata,
    probe_endpoint_status,
    test_provider_connection,
    vision_ocr_with_config,
)
from ..runner import OcrFn, ProgressEvent, run_pipeline
from ..settings import (
    AppSettings,
    DuplicateCredentialError,
    OLLAMA_LOCAL_CRED_ID,
    apply_provider_updates,
    default_jobs_dir,
    delete_credential,
    effective_api_key,
    effective_base_url,
    find_credential,
    load_settings,
    mask_settings_for_api,
    normalize_notification_mode,
    resolve_llm_endpoint,
    resolve_vision_endpoint,
    save_settings,
    upsert_credential,
)
from ..updater import UpdateError, download_and_install_update, fetch_github_releases

# Ensure .js/.css serve valid MIME types even on Windows where registry maps .js to text/plain
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

def _resolve_static_dir() -> Path:
    local = Path(__file__).parent / "static"
    if local.is_dir():
        return local
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "ocr_pipeline" / "web" / "static"
    return local


STATIC_DIR = _resolve_static_dir()

RECOMMENDED_MODELS = [
    {
        "name": "glm-ocr",
        "size": "~0.9B",
        "vram": "~2.2 GB",
        "license": "MIT",
        "description": "Fast dedicated OCR model. Best default for 8 GB VRAM with zero shared-memory spill.",
    },
    {
        "name": "deepseek-ocr",
        "size": "~3B",
        "vram": "~3.5 GB",
        "license": "MIT",
        "description": "Token-efficient low-hallucination OCR model for dense scans and notes.",
    },
    {
        "name": "qwen3-vl:4b",
        "size": "4B",
        "vram": "~4.0 GB",
        "license": "Apache-2.0",
        "description": "Balanced multimodal model for messy handwriting and diagram descriptions.",
    },
    {
        "name": "qwen3-vl:8b",
        "size": "8B",
        "vram": "~6.5 GB+",
        "license": "Apache-2.0",
        "description": "Heavy VLM for complex historical cursive & charts (may spill on 8 GB GPUs).",
    },
]


@dataclass
class JobState:
    """In-memory state for an OCR job."""

    job_id: str
    label: str
    input_dir: Path
    output_dir: Path
    work_dir: Path
    model: str
    dpi: int
    provider: str = "ollama"
    status: str = "waiting"
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    total_docs: int = 0
    completed_docs: int = 0
    skipped_docs: int = 0
    failed_pages: int = 0
    current_doc: str = ""
    current_page: int = 0
    total_pages: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    folder: str = ""
    tags: list[str] = field(default_factory=list)
    expected_files: list[Path] = field(default_factory=list, repr=False)
    archive_root: Path | None = field(default=None, repr=False)
    post_llm: dict[str, object] = field(default_factory=dict, repr=False)
    error: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    active_client: httpx.Client | None = field(default=None, repr=False)

    def _rel_path(self, p: Path) -> str:
        if p.is_relative_to(self.output_dir):
            return p.relative_to(self.output_dir).as_posix()
        if self.archive_root is not None and p.is_relative_to(self.archive_root):
            return p.relative_to(self.archive_root).as_posix()
        return p.name

    def resolve_output_file(self, file_path: str) -> Path | None:
        roots = [self.output_dir]
        if self.archive_root is not None:
            roots.append(self.archive_root)
        for base in roots:
            cand = (base / file_path).resolve()
            if cand.is_relative_to(base) and cand.is_file():
                return cand
        return None

    def to_dict(self) -> dict:
        files = sorted(self._rel_path(p) for p in self.expected_files if p.is_file())
        return {
            "job_id": self.job_id,
            "label": self.label,
            "output_dir": str(self.output_dir),
            "provider": self.provider,
            "model": self.model,
            "dpi": self.dpi,
            "status": self.status,
            "created_at": self.created_at,
            "total_docs": self.total_docs,
            "completed_docs": self.completed_docs,
            "skipped_docs": self.skipped_docs,
            "failed_pages": self.failed_pages,
            "current_doc": self.current_doc,
            "current_page": self.current_page,
            "total_pages": self.total_pages,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "folder": self.folder,
            "tags": self.tags,
            "error": self.error,
            "files": files,
        }


def _save_uploads(files: list[UploadFile], dest_dir: Path) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for upload in files:
        if not upload.filename:
            continue
        rel = Path(upload.filename.replace("\\", "/"))
        safe_parts = [p for p in rel.parts if p not in ("", ".", "..", "/", "\\")]
        if not safe_parts:
            continue
        target = _unique_output_path(dest_dir.joinpath(*safe_parts))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            while chunk := upload.file.read(1024 * 1024):
                out.write(chunk)
        saved.append(target.relative_to(dest_dir).as_posix())
    return saved


def _stage_local_files(raw_json: str, dest_dir: Path) -> list[str]:
    """Copy individually selected local PDF/image files into job input directory."""
    try:
        entries = json.loads(raw_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid selected_paths payload.") from exc
    if not isinstance(entries, list):
        raise HTTPException(status_code=400, detail="Invalid selected_paths payload.")

    supported = IMAGE_EXTENSIONS | {PDF_EXTENSION}
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            raw_path = str(entry.get("path", "")).strip()
            raw_rel = str(entry.get("rel", "")).strip()
        else:
            raw_path = str(entry).strip()
            raw_rel = ""
        if not raw_path:
            continue
        src = Path(raw_path).expanduser().resolve()
        if not src.is_file():
            raise HTTPException(status_code=400, detail=f"Selected file not found: {src.name}")
        if src.suffix.lower() not in supported:
            continue
        rel = Path((raw_rel or src.name).replace("\\", "/"))
        safe_parts = [p for p in rel.parts if p not in ("", ".", "..", "/", "\\")] or [src.name]
        target = _unique_output_path(dest_dir.joinpath(*safe_parts))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        staged.append(target.relative_to(dest_dir).as_posix())
    return staged


def _unique_output_path(path: Path) -> Path:
    """Avoid overwriting existing archived .md files on new uploads."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    idx = 2
    while True:
        cand = path.with_name(f"{stem} ({idx}){suffix}")
        if not cand.exists():
            return cand
        idx += 1


def _resolve_artifact_path(archive_dir: Path, rel_path: str) -> Path:
    root = archive_dir.expanduser().resolve()
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root) or ".ocr_cache" in target.parts:
        raise HTTPException(status_code=400, detail="Invalid artifact path.")
    return target


def create_app(
    ollama_url: str | None = None,
    jobs_root: Path | None = None,
    settings_path: Path | None = None,
    ocr_fn: OcrFn | None = None,
    on_job_finished: Callable[[JobState], None] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI Desktop & Web UI application."""
    app = FastAPI(title="OcrPipeline", version=get_display_version())
    root = (jobs_root or default_jobs_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, JobState] = {}
    job_queue: queue.Queue[tuple[JobState, list[DocumentTask], OcrConfig]] = queue.Queue()
    lock = threading.Lock()
    pull_state: dict = {
        "active": False,
        "model": "",
        "status": "idle",
        "percent": 0.0,
        "completed_mb": 0.0,
        "total_mb": 0.0,
        "done": False,
        "error": "",
    }
    update_state: dict = {
        "active": False,
        "tag": "",
        "status": "idle",
        "percent": 0.0,
        "completed_mb": 0.0,
        "total_mb": 0.0,
        "done": False,
        "error": "",
        "message": "",
        "path": "",
    }
    effective_ocr_fn: OcrFn = ocr_fn or vision_ocr_with_config
    runtime_role_status: dict[str, dict[str, str]] = {}

    def _record_role_error(role: str, credential_id: str, exc: Exception) -> None:
        code = classify_exception_status(exc)
        with lock:
            runtime_role_status[role] = {
                "credential_id": credential_id,
                "status": code,
                "status_text": STATUS_LABELS.get(code, "Error"),
                "message": str(exc),
            }

    def _clear_role_error(role: str | None = None) -> None:
        with lock:
            if role is None:
                runtime_role_status.clear()
            else:
                runtime_role_status.pop(role, None)

    def _display_label(model: str, cred_name: str) -> str:
        clean_model = model.strip() or "(unset)"
        return f"{clean_model} ({cred_name})" if cred_name else clean_model

    def _resolve_ollama_url_for_cred(cfg: AppSettings, credential_id: str | None = None) -> str:
        cid = (credential_id or "").strip() or OLLAMA_LOCAL_CRED_ID
        cred = find_credential(cfg, cid)
        if cred is None:
            if cid == OLLAMA_LOCAL_CRED_ID:
                return cfg.ollama_url
            raise HTTPException(status_code=404, detail="Ollama credential not found.")
        spec = PROVIDER_REGISTRY.get(str(cred.get("provider") or ""))
        if spec is None or spec.protocol != "ollama":
            raise HTTPException(status_code=400, detail="Selected credential is not an Ollama endpoint.")
        return str(cred.get("base_url") or cfg.ollama_url or spec.default_base_url).strip()

    def get_settings() -> AppSettings:
        cfg = load_settings(settings_path)
        if ollama_url:
            cfg.ollama_url = ollama_url
            cfg.providers["ollama"]["base_url"] = ollama_url
            cfg.providers["ollama_local"]["base_url"] = ollama_url
            local_cred = find_credential(cfg, OLLAMA_LOCAL_CRED_ID)
            if local_cred is not None:
                local_cred["base_url"] = ollama_url
        return cfg

    def _append_job_log(state: JobState, line: str) -> None:
        log_path = state.work_dir / "ocr_pipeline.log"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,000")
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} [INFO] {line}\n")

    def _apply_post_ocr_llm(state: JobState, cfg: AppSettings) -> None:
        llm_endpoint = resolve_llm_endpoint(cfg)
        opts = state.post_llm
        do_tag = bool(opts.get("auto_tag"))
        do_rename = bool(opts.get("auto_rename"))
        do_summarize = bool(opts.get("auto_summarize"))
        if llm_endpoint is None or not (do_tag or do_rename or do_summarize):
            return

        archive_base = (state.archive_root or state.output_dir).resolve()
        index = scan_artifacts(archive_base)
        known_tags = [str(t["name"]) for t in (index.get("tags") or [])]
        multi_doc = len(state.expected_files) != 1

        for idx, md_path in enumerate(list(state.expected_files)):
            if state.cancel_event.is_set() or not md_path.is_file():
                continue
            try:
                raw_text = md_path.read_text(encoding="utf-8", errors="replace")
                res = analyze_artifact_all(
                    raw_text,
                    current_name=md_path.name,
                    known_tags=known_tags,
                    endpoint=llm_endpoint,
                )
                if do_tag and res["tags"]:
                    merged_tags = normalize_tags(list(state.tags) + list(res["tags"]))
                    update_artifact_tags(md_path, merged_tags)
                    for t in merged_tags:
                        if t not in known_tags:
                            known_tags.append(t)
                if do_summarize and res["summary"]:
                    update_artifact_summary(md_path, res["summary"])

                dst_name = md_path.name
                if do_rename and (multi_doc or not opts.get("explicit_name")) and res["filename"]:
                    dst_name = res["filename"]

                final_path = md_path
                if dst_name != md_path.name:
                    final_path = _unique_output_path(md_path.parent / dst_name)
                    md_path.rename(final_path)
                    move_artifact_sidecar(md_path, final_path)
                    with lock:
                        state.expected_files[idx] = final_path.resolve()

                _append_job_log(
                    state,
                    f"LLM POST ({llm_endpoint.provider_id}/{llm_endpoint.model}): {md_path.name} -> {state._rel_path(final_path)} tags={res['tags']}",
                )
                _clear_role_error("llm")
            except Exception as exc:
                _record_role_error("llm", llm_endpoint.credential_id, exc)
                _append_job_log(state, f"LLM WARN ({md_path.name}): {exc}")

    def _execute_job(state: JobState, tasks: list[DocumentTask], config: OcrConfig) -> None:
        with lock:
            if state.cancel_event.is_set() or state.status == "cancelled":
                state.status = "cancelled"
                return
            state.status = "running"

        def on_progress(ev: ProgressEvent) -> None:
            with lock:
                state.total_docs = ev.total_docs
                state.completed_docs = ev.completed_docs
                state.skipped_docs = ev.skipped_docs
                state.failed_pages = ev.failed_pages
                state.current_doc = ev.current_doc
                state.current_page = ev.current_page
                state.total_pages = ev.total_pages
                state.elapsed_seconds = ev.elapsed_seconds
                state.eta_seconds = ev.eta_seconds
                state.status = ev.status

        def on_client_ready(client: httpx.Client) -> None:
            with lock:
                state.active_client = client

        try:
            quiet_console = Console(file=io.StringIO(), quiet=True)
            summary = run_pipeline(
                tasks,
                state.output_dir,
                config,
                ocr_fn=effective_ocr_fn,
                console=quiet_console,
                on_progress=on_progress,
                cancel_event=state.cancel_event,
                on_client_ready=on_client_ready,
                work_dir=state.work_dir,
            )
            if not summary.cancelled and summary.processed_docs > 0:
                _apply_post_ocr_llm(state, get_settings())
            with lock:
                state.active_client = None
                state.completed_docs = summary.processed_docs + summary.skipped_docs
                state.skipped_docs = summary.skipped_docs
                state.failed_pages = summary.failed_pages
                state.status = "cancelled" if summary.cancelled else "completed"
            if not summary.cancelled and summary.processed_docs > 0 and summary.failed_pages == 0:
                _clear_role_error("vision")
        except Exception as exc:
            cfg_now = get_settings()
            _record_role_error("vision", cfg_now.vision_credential_id, exc)
            with lock:
                state.active_client = None
                if state.cancel_event.is_set():
                    state.status = "cancelled"
                else:
                    state.status = "failed"
                    state.error = str(exc)
        if on_job_finished is not None:
            on_job_finished(state)

    def _queue_worker() -> None:
        while True:
            state, tasks, config = job_queue.get()
            try:
                _execute_job(state, tasks, config)
            finally:
                job_queue.task_done()

    # ponytail: single FIFO job queue prevents concurrent OCR runs from fighting for GPU VRAM
    threading.Thread(target=_queue_worker, daemon=True).start()

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/icon.png")
    def app_icon() -> Response:
        buf = io.BytesIO()
        generate_app_icon_image(64).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    @app.post("/api/open-url")
    def open_external_url(payload: dict = Body(...)) -> dict:
        from urllib.parse import urlparse
        import webbrowser

        raw_url = str(payload.get("url", "")).strip()
        parsed = urlparse(raw_url)
        if parsed.scheme.lower() not in ("http", "https", "mailto"):
            raise HTTPException(status_code=400, detail="Unsupported URL scheme.")
        webbrowser.open(raw_url)
        return {"ok": True}

    @app.get("/api/health")
    def health() -> dict:
        cfg = get_settings()
        return {
            "status": "ok",
            "version": get_display_version(),
            "base_version": __version__,
            "license": "MIT OR Apache-2.0",
            "copyright": "Copyright (c) 2026 Weien Xu",
            "platform": platform.system(),
            "setup_completed": cfg.setup_completed,
        }

    @app.get("/api/system-info")
    def system_info() -> dict:
        return get_system_info()

    @app.get("/api/settings")
    def read_settings() -> dict:
        cfg = get_settings()
        data = mask_settings_for_api(cfg)
        data["default_system_prompt"] = DEFAULT_SYSTEM_PROMPT
        return data

    @app.put("/api/settings")
    def update_settings(payload: dict = Body(...)) -> dict:
        current = get_settings()
        if payload.get("reset_prompt"):
            payload["system_prompt"] = DEFAULT_SYSTEM_PROMPT
        if "providers" in payload:
            apply_provider_updates(current, payload["providers"])
        allowed = current.to_dict()
        for key in allowed:
            if key in payload and key != "providers":
                setattr(current, key, payload[key])
        if "ollama_url" in payload and "providers" not in payload:
            new_url = str(payload["ollama_url"]).strip()
            current.providers["ollama"]["base_url"] = new_url
            current.providers["ollama_local"]["base_url"] = new_url
            local_cred = find_credential(current, OLLAMA_LOCAL_CRED_ID)
            if local_cred is not None and new_url:
                local_cred["base_url"] = new_url
        if "vision_credential_id" in payload:
            v_cred = find_credential(current, str(payload["vision_credential_id"]))
            if v_cred is not None:
                current.vision_credential_id = str(v_cred["id"])
                current.vision_provider = str(v_cred["provider"])
        elif "vision_provider" in payload:
            v_cred = find_credential(current, current.vision_credential_id)
            if v_cred is None or str(v_cred["provider"]) != str(current.vision_provider):
                from ..settings import find_credential_for_provider

                matched_v = find_credential_for_provider(current, current.vision_provider)
                if matched_v is not None:
                    current.vision_credential_id = str(matched_v["id"])
        if "llm_credential_id" in payload:
            l_cred = find_credential(current, str(payload["llm_credential_id"]))
            if l_cred is not None:
                current.llm_credential_id = str(l_cred["id"])
                current.llm_provider = str(l_cred["provider"])
        elif "llm_provider" in payload:
            l_cred = find_credential(current, current.llm_credential_id)
            if l_cred is None or str(l_cred["provider"]) != str(current.llm_provider):
                from ..settings import find_credential_for_provider

                matched_l = find_credential_for_provider(current, current.llm_provider)
                if matched_l is not None:
                    current.llm_credential_id = str(matched_l["id"])
        if "default_model" in payload and "vision_model" not in payload:
            current.vision_model = str(payload["default_model"]).strip() or current.vision_model
        current.dpi = max(72, min(int(current.dpi), 600))
        current.max_image_dim = max(512, min(int(current.max_image_dim), 4096))
        current.num_ctx = max(1024, min(int(current.num_ctx), 131072))
        if current.theme not in ("auto", "light", "dark"):
            current.theme = "auto"
        current.desktop_notifications = normalize_notification_mode(current.desktop_notifications)
        current.auto_check_updates = bool(current.auto_check_updates)
        current.llm_enabled = bool(current.llm_enabled)
        current.llm_auto_ctx = bool(current.llm_auto_ctx)
        current.vision_custom_model = bool(current.vision_custom_model)
        current.llm_custom_model = bool(current.llm_custom_model)
        current.auto_tag_on_ocr = bool(current.auto_tag_on_ocr)
        current.auto_rename_on_ocr = bool(current.auto_rename_on_ocr)
        current.auto_organize_on_ocr = bool(current.auto_organize_on_ocr)
        current.auto_organize_allow_new_folders = bool(current.auto_organize_allow_new_folders)
        current.auto_summarize_on_ocr = bool(current.auto_summarize_on_ocr)
        if not current.vision_model.strip():
            current.vision_model = "glm-ocr"
        current.default_model = current.vision_model
        if not current.archive_dir.strip():
            current.archive_dir = str(AppSettings().archive_dir)
        save_settings(current, settings_path)
        _clear_role_error()
        data = mask_settings_for_api(current)
        data["default_system_prompt"] = DEFAULT_SYSTEM_PROMPT
        return data

    @app.get("/api/providers")
    def get_providers() -> dict:
        cfg = get_settings()
        masked = mask_settings_for_api(cfg)
        vision_ep = resolve_vision_endpoint(cfg)
        llm_ep = resolve_llm_endpoint(cfg)
        llm_cred = find_credential(cfg, cfg.llm_credential_id)
        local_cred = find_credential(cfg, OLLAMA_LOCAL_CRED_ID)
        preview_ep = llm_ep or ProviderEndpoint(
            provider_id=str(llm_cred["provider"]) if llm_cred else (cfg.llm_provider or "ollama_local"),
            base_url=effective_base_url(cfg, cfg.llm_provider, cfg.llm_credential_id),
            api_key="",
            model=cfg.llm_model or "qwen3:1.7b",
            num_ctx=cfg.num_ctx,
            auto_ctx=cfg.llm_auto_ctx,
            credential_id=str(llm_cred["id"]) if llm_cred else OLLAMA_LOCAL_CRED_ID,
            credential_name=str(llm_cred["name"]) if llm_cred else "Ollama Local",
        )
        llm_cname = str(llm_cred["name"]) if llm_cred else (PROVIDER_REGISTRY.get(cfg.llm_provider, PROVIDER_REGISTRY["ollama_local"]).short_label)
        return {
            "registry": list_providers_metadata(),
            "credentials": masked["credentials"],
            "providers": masked["providers"],
            "ollama_local": {
                "id": OLLAMA_LOCAL_CRED_ID,
                "enabled": bool(local_cred.get("enabled", True)) if local_cred else True,
                "base_url": str(local_cred.get("base_url") or cfg.ollama_url) if local_cred else cfg.ollama_url,
            },
            "vision": {
                "credential_id": vision_ep.credential_id or cfg.vision_credential_id,
                "credential_name": vision_ep.credential_name,
                "provider": vision_ep.provider_id,
                "model": vision_ep.model,
                "display_label": _display_label(vision_ep.model, vision_ep.credential_name),
                "custom": bool(cfg.vision_custom_model),
            },
            "llm": {
                "enabled": bool(cfg.llm_enabled),
                "available": llm_ep is not None,
                "credential_id": cfg.llm_credential_id,
                "credential_name": llm_cname,
                "provider": str(llm_cred["provider"]) if llm_cred else cfg.llm_provider,
                "model": cfg.llm_model,
                "display_label": _display_label(cfg.llm_model, llm_cname) if (cfg.llm_enabled and cfg.llm_model.strip()) else "Disabled",
                "custom": bool(cfg.llm_custom_model),
                "auto_ctx": bool(cfg.llm_auto_ctx),
                "vram_plan": calculate_vram_plan(preview_ep).to_dict(),
            },
        }

    @app.get("/api/providers/status")
    def get_providers_status(refresh: bool = Query(False)) -> dict:
        if refresh:
            _clear_role_error()
        cfg = get_settings()
        local_cred = find_credential(cfg, OLLAMA_LOCAL_CRED_ID)
        local_enabled = bool(local_cred.get("enabled", True)) if local_cred else True

        vision_ep = resolve_vision_endpoint(cfg)
        vis_cred = find_credential(cfg, vision_ep.credential_id)
        vis_enabled = bool(vis_cred.get("enabled", True)) if vis_cred else True
        vis_probe = probe_endpoint_status(vision_ep, enabled=vis_enabled, timeout=3.0)

        # Overlay recent runtime error (e.g. Out of Quota 429 during inference) when /models probe still passed.
        with lock:
            rt_vis = dict(runtime_role_status.get("vision") or {})
            rt_llm = dict(runtime_role_status.get("llm") or {})

        if vis_probe["status"] == "ok" and rt_vis and rt_vis.get("credential_id") == vision_ep.credential_id:
            vis_probe["status"] = rt_vis["status"]
            vis_probe["status_text"] = rt_vis["status_text"]
            vis_probe["message"] = rt_vis.get("message") or vis_probe["message"]

        llm_cred = find_credential(cfg, cfg.llm_credential_id)
        llm_cname = str(llm_cred["name"]) if llm_cred else "Ollama Local"
        if not cfg.llm_enabled:
            llm_info = {
                "enabled": False,
                "model": "",
                "credential_id": cfg.llm_credential_id,
                "credential_name": llm_cname,
                "label": "Disabled",
                "status": "disabled",
                "status_text": "Disabled",
                "online": False,
                "models": [],
                "message": "Assistant LLM is disabled.",
            }
        else:
            l_enabled = bool(llm_cred.get("enabled", True)) if llm_cred else True
            if not cfg.llm_model.strip():
                llm_info = {
                    "enabled": True,
                    "model": "",
                    "credential_id": cfg.llm_credential_id,
                    "credential_name": llm_cname,
                    "label": f"(Unset) ({llm_cname})",
                    "status": "error",
                    "status_text": "No Model",
                    "online": False,
                    "models": [],
                    "message": "Select an Assistant LLM model.",
                }
            else:
                llm_ep = ProviderEndpoint(
                    provider_id=str(llm_cred["provider"]) if llm_cred else cfg.llm_provider,
                    base_url=effective_base_url(cfg, cfg.llm_provider, cfg.llm_credential_id),
                    api_key=effective_api_key(cfg, cfg.llm_provider, cfg.llm_credential_id),
                    model=cfg.llm_model.strip(),
                    num_ctx=cfg.num_ctx,
                    auto_ctx=cfg.llm_auto_ctx,
                    credential_id=cfg.llm_credential_id,
                    credential_name=llm_cname,
                )
                if llm_ep.credential_id == vision_ep.credential_id and l_enabled == vis_enabled and vis_probe["status"] != "model_missing":
                    llm_probe = dict(vis_probe)
                    spec_l = PROVIDER_REGISTRY.get(llm_ep.provider_id)
                    if llm_probe["status"] == "ok" and spec_l and spec_l.protocol == "ollama":
                        names = set(llm_probe.get("models") or [])
                        if llm_ep.model not in names and f"{llm_ep.model}:latest" not in names:
                            llm_probe["status"] = "model_missing"
                            llm_probe["status_text"] = STATUS_LABELS["model_missing"]
                            llm_probe["online"] = False
                            llm_probe["message"] = f"Model '{llm_ep.model}' is not pulled on {llm_cname}."
                else:
                    llm_probe = probe_endpoint_status(llm_ep, enabled=l_enabled, timeout=3.0)

                if llm_probe["status"] == "ok" and rt_llm and rt_llm.get("credential_id") == cfg.llm_credential_id:
                    llm_probe["status"] = rt_llm["status"]
                    llm_probe["status_text"] = rt_llm["status_text"]
                    llm_probe["message"] = rt_llm.get("message") or llm_probe["message"]

                llm_info = {
                    "enabled": True,
                    "model": llm_ep.model,
                    "credential_id": llm_ep.credential_id,
                    "credential_name": llm_cname,
                    "label": _display_label(llm_ep.model, llm_cname),
                    **llm_probe,
                }

        # Reuse Ollama Local probe if Vision or LLM already probed cred_ollama_local
        if vision_ep.credential_id == OLLAMA_LOCAL_CRED_ID:
            ollama_local_status = {
                "enabled": local_enabled,
                "online": bool(vis_probe.get("online")) or vis_probe.get("status") == "model_missing",
                "installed": bool(vis_probe.get("installed")) or (find_ollama_executable() is not None),
                "url": str(local_cred.get("base_url") or cfg.ollama_url) if local_cred else cfg.ollama_url,
                "models": list(vis_probe.get("models") or []),
                "detailed": list(vis_probe.get("detailed") or []),
            }
        else:
            local_url = str(local_cred.get("base_url") or cfg.ollama_url) if local_cred else cfg.ollama_url
            local_ep = ProviderEndpoint("ollama_local", local_url, "", "", credential_id=OLLAMA_LOCAL_CRED_ID, credential_name="Ollama Local")
            lp = probe_endpoint_status(local_ep, enabled=local_enabled, timeout=2.5)
            ollama_local_status = {
                "enabled": local_enabled,
                "online": bool(lp.get("online")),
                "installed": bool(lp.get("installed")) or (find_ollama_executable() is not None),
                "url": local_url,
                "models": list(lp.get("models") or []),
                "detailed": list(lp.get("detailed") or []),
            }

        return {
            "vision": {
                "provider": vision_ep.provider_id,
                "model": vision_ep.model,
                "credential_id": vision_ep.credential_id,
                "credential_name": vision_ep.credential_name,
                "label": _display_label(vision_ep.model, vision_ep.credential_name),
                **vis_probe,
            },
            "llm": llm_info,
            "ollama_local": ollama_local_status,
        }

    @app.post("/api/credentials")
    def create_credential_endpoint(payload: dict = Body(...)) -> dict:
        cfg = get_settings()
        try:
            created = upsert_credential(cfg, payload)
        except DuplicateCredentialError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_settings(cfg, settings_path)
        _clear_role_error()
        masked = mask_settings_for_api(cfg)["credentials"]
        return next(c for c in masked if c["id"] == created["id"])

    @app.put("/api/credentials/{cred_id}")
    def update_credential_endpoint(cred_id: str, payload: dict = Body(...)) -> dict:
        cfg = get_settings()
        try:
            updated = upsert_credential(cfg, payload, cred_id=cred_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DuplicateCredentialError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_settings(cfg, settings_path)
        _clear_role_error()
        masked = mask_settings_for_api(cfg)["credentials"]
        return next(c for c in masked if c["id"] == updated["id"])

    @app.delete("/api/credentials/{cred_id}")
    def delete_credential_endpoint(cred_id: str) -> dict:
        cfg = get_settings()
        try:
            delete_credential(cfg, cred_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_settings(cfg, settings_path)
        _clear_role_error()
        return {"ok": True, "deleted": cred_id}

    @app.post("/api/llm/vram-estimate")
    def estimate_llm_vram(payload: dict = Body(default={})) -> dict:
        cfg = get_settings()
        cid = str(payload.get("credential_id") or "").strip() or None
        cred = find_credential(cfg, cid) if cid else None
        pid = str(payload.get("provider") or (cred["provider"] if cred else cfg.llm_provider) or "ollama_local").strip().lower()
        model = str(payload.get("model") or cfg.llm_model or "qwen3:1.7b").strip()
        model_size_gb = payload.get("model_size_gb")
        param_size = str(payload.get("parameter_size") or "").strip()
        quant = str(payload.get("quantization") or "").strip()
        probe_ollama = bool(payload.get("probe_ollama", False))
        ep = ProviderEndpoint(
            provider_id=pid,
            base_url=effective_base_url(cfg, pid, cid),
            api_key=effective_api_key(cfg, pid, cid),
            model=model,
            num_ctx=cfg.num_ctx,
            auto_ctx=bool(payload.get("auto_ctx", cfg.llm_auto_ctx)),
            credential_id=cid or "",
            credential_name=str(cred["name"]) if cred else "",
        )
        return calculate_vram_plan(
            ep,
            model_size_gb=float(model_size_gb) if isinstance(model_size_gb, (int, float)) else None,
            parameter_size=param_size,
            quantization=quant,
            probe_ollama=probe_ollama,
        ).to_dict()

    @app.post("/api/providers/test")
    def probe_provider(payload: dict = Body(...)) -> dict:
        cfg = get_settings()
        cid = str(payload.get("credential_id") or "").strip() or None
        cred = find_credential(cfg, cid) if cid else None
        pid = str(payload.get("provider") or (cred["provider"] if cred else "ollama_local")).strip().lower()
        base_url = str(payload.get("base_url") or effective_base_url(cfg, pid, cid)).strip()
        raw_key = str(payload.get("api_key") or "").strip()
        if raw_key == "••••••••":
            raw_key = ""
        api_key = raw_key or effective_api_key(cfg, pid, cid)
        try:
            res = test_provider_connection(pid, base_url=base_url, api_key=api_key)
            _clear_role_error()
            return res
        except ProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/updates/check")
    def check_for_updates() -> dict:
        try:
            return fetch_github_releases()
        except UpdateError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/updates/install")
    def install_github_update(payload: dict = Body(...)) -> dict:
        tag = str(payload.get("tag", "")).strip()
        download_url = str(payload.get("download_url", "")).strip()
        asset_name = str(payload.get("asset_name", "")).strip()
        if not download_url or not asset_name:
            raise HTTPException(status_code=400, detail="download_url and asset_name are required.")
        with lock:
            if update_state["active"]:
                raise HTTPException(status_code=409, detail="An update download is already in progress.")
            update_state.update(
                {
                    "active": True,
                    "tag": tag,
                    "status": "downloading",
                    "percent": 0.0,
                    "completed_mb": 0.0,
                    "total_mb": 0.0,
                    "done": False,
                    "error": "",
                    "message": "",
                    "path": "",
                }
            )

        def on_progress(downloaded: int, total: int) -> None:
            with lock:
                update_state["completed_mb"] = round(downloaded / (1024 * 1024), 1)
                update_state["total_mb"] = round(total / (1024 * 1024), 1)
                if total > 0:
                    update_state["percent"] = round((downloaded / total) * 100.0, 1)

        def worker() -> None:
            try:
                res = download_and_install_update(download_url, asset_name, on_progress=on_progress)
                with lock:
                    update_state["active"] = False
                    update_state["done"] = True
                    update_state["percent"] = 100.0
                    update_state["status"] = "completed"
                    update_state["message"] = str(res.get("message", ""))
                    update_state["path"] = str(res.get("path", ""))
            except Exception as exc:
                with lock:
                    update_state["active"] = False
                    update_state["done"] = True
                    update_state["status"] = "failed"
                    update_state["error"] = str(exc)

        threading.Thread(target=worker, daemon=True).start()
        with lock:
            return dict(update_state)

    @app.get("/api/updates/install-status")
    def get_update_install_status() -> dict:
        with lock:
            return dict(update_state)

    @app.get("/api/models")
    def list_models(credential_id: str | None = Query(None)) -> dict:
        cfg = get_settings()
        target_url = _resolve_ollama_url_for_cred(cfg, credential_id)
        installed = find_ollama_executable() is not None
        try:
            detailed = list_models_detailed(target_url, timeout=3.0)
            return {
                "online": True,
                "installed": True,
                "models": [m["name"] for m in detailed],
                "detailed": detailed,
                "recommended": RECOMMENDED_MODELS,
                "default_model": cfg.default_model,
                "ollama_url": target_url,
            }
        except httpx.HTTPError as exc:
            return {
                "online": False,
                "installed": installed,
                "models": [],
                "detailed": [],
                "recommended": RECOMMENDED_MODELS,
                "default_model": cfg.default_model,
                "ollama_url": target_url,
                "error": str(exc),
            }

    @app.post("/api/ollama/start")
    def start_ollama() -> dict:
        cfg = get_settings()
        return start_ollama_service(cfg.ollama_url)

    @app.post("/api/models/pull")
    def pull_ollama_model(model: str = Form(...), credential_id: str = Form("")) -> dict:
        cfg = get_settings()
        target_url = _resolve_ollama_url_for_cred(cfg, credential_id)
        target = model.strip()
        if not target:
            raise HTTPException(status_code=400, detail="Model name is required.")
        try:
            pull_model(target_url, target)
            return {"ok": True, "model": target}
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/models/pull-async")
    def start_model_pull(model: str = Form(...), credential_id: str = Form("")) -> dict:
        cfg = get_settings()
        target_url = _resolve_ollama_url_for_cred(cfg, credential_id)
        target = model.strip()
        if not target:
            raise HTTPException(status_code=400, detail="Model name is required.")
        with lock:
            if pull_state["active"]:
                raise HTTPException(status_code=409, detail=f"Already pulling '{pull_state['model']}'.")
            pull_state.update(
                {
                    "active": True,
                    "model": target,
                    "status": "starting",
                    "percent": 0.0,
                    "completed_mb": 0.0,
                    "total_mb": 0.0,
                    "done": False,
                    "error": "",
                }
            )

        def on_progress(ev: dict) -> None:
            with lock:
                pull_state["status"] = ev.get("status", "pulling")
                completed = ev.get("completed") or 0
                total = ev.get("total") or 0
                pull_state["completed_mb"] = round(completed / (1024 * 1024), 1)
                pull_state["total_mb"] = round(total / (1024 * 1024), 1)
                if ev.get("percent") is not None:
                    pull_state["percent"] = float(ev["percent"])

        def worker() -> None:
            try:
                pull_model_stream(target_url, target, on_progress=on_progress)
                with lock:
                    pull_state["active"] = False
                    pull_state["done"] = True
                    pull_state["percent"] = 100.0
                    pull_state["status"] = "completed"
            except Exception as exc:
                with lock:
                    pull_state["active"] = False
                    pull_state["done"] = True
                    pull_state["status"] = "failed"
                    pull_state["error"] = str(exc)

        threading.Thread(target=worker, daemon=True).start()
        with lock:
            return dict(pull_state)

    @app.get("/api/models/pull-status")
    def get_model_pull_status() -> dict:
        with lock:
            return dict(pull_state)

    @app.delete("/api/models/{model_name:path}")
    def remove_ollama_model(model_name: str, credential_id: str | None = Query(None)) -> dict:
        cfg = get_settings()
        target_url = _resolve_ollama_url_for_cred(cfg, credential_id)
        try:
            delete_model(target_url, model_name)
            return {"ok": True, "deleted": model_name}
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/folders")
    def list_folders() -> dict:
        cfg = get_settings()
        data = scan_artifacts(Path(cfg.archive_dir))
        return {"archive_dir": data["archive_dir"], "folders": data["folders"]}

    @app.get("/api/artifacts")
    def list_artifacts(
        q: str = Query(""),
        folder: str = Query(""),
        tag: str = Query(""),
        sort: str = Query("newest"),
        cwd: str | None = Query(None),
    ) -> dict:
        cfg = get_settings()
        result = scan_artifacts(Path(cfg.archive_dir), query=q, folder=folder, tag=tag, sort=sort, cwd=cwd)
        result["obsidian_mode"] = cfg.obsidian_mode
        return result

    @app.post("/api/artifacts/folder")
    def create_artifact_folder(payload: dict = Body(...)) -> dict:
        cfg = get_settings()
        rel = sanitize_rel_folder(str(payload.get("folder", "")))
        if not rel.parts:
            raise HTTPException(status_code=400, detail="Folder name is required.")
        root_dir = Path(cfg.archive_dir).expanduser().resolve()
        target = (root_dir / rel).resolve()
        if not target.is_relative_to(root_dir):
            raise HTTPException(status_code=400, detail="Invalid folder path.")
        target.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "folder": target.relative_to(root_dir).as_posix()}

    @app.patch("/api/artifacts/folder")
    def rename_artifact_folder(payload: dict = Body(...)) -> dict:
        cfg = get_settings()
        try:
            new_rel = rename_folder(
                Path(cfg.archive_dir),
                str(payload.get("folder", "")),
                str(payload.get("new_name", "")),
            )
            return {"ok": True, "folder": new_rel}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/artifacts/folder")
    def delete_artifact_folder(folder: str = Query(...), recursive: bool = Query(False)) -> dict:
        cfg = get_settings()
        try:
            delete_folder(Path(cfg.archive_dir), folder, recursive=recursive)
            return {"ok": True}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=409, detail="Folder is not empty.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/artifacts/content")
    def get_artifact_content(path: str = Query(...), download: bool = False) -> FileResponse:
        cfg = get_settings()
        target = _resolve_artifact_path(Path(cfg.archive_dir), path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        if download:
            return FileResponse(target, filename=target.name, media_type="text/markdown")
        return FileResponse(target, media_type="text/plain; charset=utf-8")

    @app.patch("/api/artifacts/move")
    @app.patch("/api/artifacts/rename")
    def move_or_rename_artifact(payload: dict = Body(...)) -> dict:
        cfg = get_settings()
        archive_dir = Path(cfg.archive_dir).expanduser().resolve()
        old_rel = str(payload.get("path", "")).strip()
        if not old_rel:
            raise HTTPException(status_code=400, detail="path is required.")
        src = _resolve_artifact_path(archive_dir, old_rel)
        if not src.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")

        new_name = str(payload.get("new_name", "")).strip() or src.name
        if "/" in new_name or "\\" in new_name or ".." in new_name:
            raise HTTPException(status_code=400, detail="Invalid filename.")
        if not new_name.lower().endswith(".md"):
            new_name += ".md"

        if "target_folder" in payload:
            rel_folder = sanitize_rel_folder(str(payload.get("target_folder") or ""))
            dst_dir = (archive_dir / rel_folder).resolve()
        else:
            dst_dir = src.parent
        if not dst_dir.is_relative_to(archive_dir):
            raise HTTPException(status_code=400, detail="Invalid target folder.")
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = (dst_dir / new_name).resolve()
        if dst != src and dst.exists():
            raise HTTPException(status_code=409, detail="An artifact with that name already exists there.")
        src.rename(dst)
        move_artifact_sidecar(src, dst)
        return {"ok": True, "path": dst.relative_to(archive_dir).as_posix()}

    @app.patch("/api/artifacts/tags")
    def patch_artifact_tags(payload: dict = Body(...)) -> dict:
        cfg = get_settings()
        rel = str(payload.get("path", "")).strip()
        if not rel:
            raise HTTPException(status_code=400, detail="path is required.")
        target = _resolve_artifact_path(Path(cfg.archive_dir), rel)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        raw_tags = payload.get("tags", [])
        tags = normalize_tags(raw_tags if isinstance(raw_tags, (list, str)) else [])
        updated = update_artifact_tags(target, tags)
        return {"ok": True, "tags": updated}

    @app.post("/api/artifacts/obsidian")
    def open_in_obsidian(payload: dict = Body(default={})) -> dict:
        from urllib.parse import quote
        import webbrowser

        cfg = get_settings()
        archive_dir = Path(cfg.archive_dir).expanduser().resolve()
        ensure_obsidian_vault(archive_dir)
        rel = str(payload.get("path", "")).strip()
        uri = f"obsidian://open?vault={quote(archive_dir.name)}"
        if rel:
            uri += f"&file={quote(rel)}"
        webbrowser.open(uri)
        return {"ok": True, "uri": uri}

    @app.delete("/api/artifacts")
    def delete_artifact(path: str = Query(...)) -> dict:
        cfg = get_settings()
        target = _resolve_artifact_path(Path(cfg.archive_dir), path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        target.unlink()
        delete_artifact_sidecar(target)
        return {"ok": True}

    @app.post("/api/artifacts/reveal")
    def reveal_artifact(payload: dict = Body(default={})) -> dict:
        cfg = get_settings()
        archive_dir = Path(cfg.archive_dir).expanduser().resolve()
        archive_dir.mkdir(parents=True, exist_ok=True)
        rel = str(payload.get("path", "")).strip()
        target = _resolve_artifact_path(archive_dir, rel).parent if rel else archive_dir
        open_in_file_manager(target)
        return {"ok": True}

    def _require_llm() -> tuple[AppSettings, object]:
        cfg = get_settings()
        endpoint = resolve_llm_endpoint(cfg)
        if endpoint is None:
            raise HTTPException(
                status_code=400,
                detail="Optional Assistant LLM is not configured. Enable and select an LLM in Models & Providers.",
            )
        return cfg, endpoint

    def _run_llm_call(endpoint: object, fn):
        cid = str(getattr(endpoint, "credential_id", "") or "")
        try:
            out = fn()
            _clear_role_error("llm")
            return out
        except (ProviderError, ValueError) as exc:
            _record_role_error("llm", cid, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.put("/api/artifacts/content")
    def save_artifact_content(payload: dict = Body(...)) -> dict:
        cfg = get_settings()
        rel = str(payload.get("path", "")).strip()
        if not rel:
            raise HTTPException(status_code=400, detail="path is required.")
        target = _resolve_artifact_path(Path(cfg.archive_dir), rel)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        content = str(payload.get("content", ""))
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": rel}

    @app.post("/api/artifacts/ai/tags")
    def ai_artifact_tags(payload: dict = Body(...)) -> dict:
        cfg, endpoint = _require_llm()
        rel = str(payload.get("path", "")).strip()
        target = _resolve_artifact_path(Path(cfg.archive_dir), rel)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        index = scan_artifacts(Path(cfg.archive_dir))
        known_tags = [str(t["name"]) for t in (index.get("tags") or [])]
        raw_text = target.read_text(encoding="utf-8", errors="replace")
        tags = _run_llm_call(endpoint, lambda: suggest_tags(raw_text, known_tags, endpoint))  # type: ignore[arg-type]
        if payload.get("apply", True):
            if not tags:
                raise HTTPException(status_code=502, detail="Assistant LLM returned no valid tags for this document.")
            tags = update_artifact_tags(target, tags)
        return {"ok": True, "tags": tags, "applied": bool(payload.get("apply", True))}

    @app.post("/api/artifacts/ai/rename")
    def ai_artifact_rename(payload: dict = Body(...)) -> dict:
        cfg, endpoint = _require_llm()
        rel = str(payload.get("path", "")).strip()
        target = _resolve_artifact_path(Path(cfg.archive_dir), rel)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        raw_text = target.read_text(encoding="utf-8", errors="replace")
        suggested = _run_llm_call(endpoint, lambda: suggest_filename(raw_text, target.name, endpoint))  # type: ignore[arg-type]
        return {"ok": True, "suggested_name": suggested}

    @app.post("/api/artifacts/ai/summarize")
    def ai_artifact_summarize(payload: dict = Body(...)) -> dict:
        cfg, endpoint = _require_llm()
        rel = str(payload.get("path", "")).strip()
        target = _resolve_artifact_path(Path(cfg.archive_dir), rel)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        raw_text = target.read_text(encoding="utf-8", errors="replace")
        summary = _run_llm_call(endpoint, lambda: summarize_document(raw_text, endpoint))  # type: ignore[arg-type]
        saved = update_artifact_summary(target, summary)
        return {"ok": True, "summary": saved}

    @app.post("/api/artifacts/ai/cleanup")
    def ai_artifact_cleanup(payload: dict = Body(...)) -> dict:
        cfg, endpoint = _require_llm()
        rel = str(payload.get("path", "")).strip()
        target = _resolve_artifact_path(Path(cfg.archive_dir), rel)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        raw_text = target.read_text(encoding="utf-8", errors="replace")
        cleaned = _run_llm_call(endpoint, lambda: cleanup_ocr_markdown(raw_text, endpoint))  # type: ignore[arg-type]
        return {"ok": True, "original": raw_text, "cleaned_markdown": cleaned}

    @app.post("/api/artifacts/ai/folder-summary")
    def ai_folder_summary(payload: dict = Body(default={})) -> dict:
        cfg, endpoint = _require_llm()
        cwd = sanitize_rel_folder(str(payload.get("cwd", ""))).as_posix()
        if cwd == ".":
            cwd = ""
        result = _run_llm_call(
            endpoint,
            lambda: summarize_folder_contents(Path(cfg.archive_dir), cwd, endpoint),  # type: ignore[arg-type]
        )
        return {"ok": True, **result}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        with lock:
            ordered = list(reversed(jobs.values()))
        return [j.to_dict() for j in ordered]

    @app.post("/api/jobs")
    def create_job(
        source_mode: str = Form("upload"),
        local_input_path: str = Form(""),
        local_output_path: str = Form(""),
        job_name: str = Form(""),
        folder: str = Form(""),
        tags: str = Form(""),
        credential_id: str = Form(""),
        provider: str = Form(""),
        model: str = Form(""),
        dpi: int = Form(0),
        max_dim: int = Form(0),
        series_mode: bool = Form(False),
        force: bool = Form(False),
        auto_tag: bool | None = Form(None),
        auto_rename: bool | None = Form(None),
        auto_organize: bool | None = Form(None),
        auto_summarize: bool | None = Form(None),
        selected_paths: str = Form(""),
        files: list[UploadFile] = File(default=[]),
    ) -> dict:
        settings = get_settings()
        job_id = uuid.uuid4().hex[:10]
        job_dir = root / job_id
        archive_base = Path(settings.archive_dir).expanduser()
        rel_folder = sanitize_rel_folder(folder)
        clean_name = job_name.strip().replace("/", "_").replace("\\", "_")
        parsed_tags = normalize_tags(tags)

        if source_mode == "local":
            if not local_input_path.strip():
                raise HTTPException(status_code=400, detail="Local input path is required.")
            input_path = Path(local_input_path.strip()).expanduser()
            default_sub = archive_base / rel_folder if rel_folder.parts else archive_base / f"{input_path.stem or 'batch'}"
            output_dir = Path(local_output_path.strip()).expanduser() if local_output_path.strip() else default_sub
            label = clean_name or input_path.name or str(input_path)
        else:
            valid_files = [f for f in files if f.filename]
            input_path = job_dir / "input"
            output_dir = archive_base / rel_folder if rel_folder.parts else archive_base
            staged_names: list[str] = []
            if selected_paths.strip():
                staged_names.extend(_stage_local_files(selected_paths, input_path))
            if valid_files:
                staged_names.extend(_save_uploads(valid_files, input_path))
            if not staged_names:
                raise HTTPException(status_code=400, detail="Please select at least one PDF or image.")
            label = clean_name or (staged_names[0] if len(staged_names) == 1 else f"{len(staged_names)} uploaded files")

        vision_ep = resolve_vision_endpoint(
            settings,
            provider_id=provider.strip() or None,
            model=model.strip() or None,
            credential_id=credential_id.strip() or None,
        )
        config = OcrConfig(
            model=vision_ep.model,
            provider=vision_ep.provider_id,
            base_url=vision_ep.base_url,
            api_key=vision_ep.api_key,
            ollama_url=vision_ep.base_url if vision_ep.provider_id in ("ollama", "ollama_local") else settings.ollama_url,
            dpi=max(72, min(dpi or settings.dpi, 600)),
            max_image_dim=max(512, min(max_dim or settings.max_image_dim, 4096)),
            num_ctx=settings.num_ctx,
            obsidian_mode=settings.obsidian_mode,
            tags=tuple(parsed_tags),
            system_prompt=settings.system_prompt,
            force=force,
        )

        try:
            tasks: list[DocumentTask] = discover_documents(input_path, output_dir, series_mode=series_mode)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not tasks:
            raise HTTPException(status_code=400, detail="No PDF or image files found in input.")

        if clean_name and len(tasks) == 1:
            t0 = tasks[0]
            stem = clean_name[:-3] if clean_name.lower().endswith(".md") else clean_name
            tasks[0] = DocumentTask(
                kind=t0.kind,
                source_paths=t0.source_paths,
                output_path=t0.output_path.with_name(f"{stem}.md"),
                display_name=clean_name,
            )

        if source_mode == "upload" and not force:
            tasks = [
                DocumentTask(
                    kind=t.kind,
                    source_paths=t.source_paths,
                    output_path=_unique_output_path(t.output_path),
                    display_name=t.display_name,
                )
                for t in tasks
            ]

        if ocr_fn is None:
            vis_cred = find_credential(settings, vision_ep.credential_id)
            if vis_cred is not None and not vis_cred.get("enabled", True):
                raise HTTPException(
                    status_code=503,
                    detail=f"Credential '{vis_cred['name']}' is disabled. Enable it in Providers & Models.",
                )
            try:
                check_endpoint_ready(vision_ep, require_vision=True)
            except ProviderError as exc:
                _record_role_error("vision", vision_ep.credential_id, exc)
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        post_llm = {
            "auto_tag": settings.auto_tag_on_ocr if auto_tag is None else auto_tag,
            "auto_rename": settings.auto_rename_on_ocr if auto_rename is None else auto_rename,
            "auto_organize": settings.auto_organize_on_ocr if auto_organize is None else auto_organize,
            "auto_summarize": settings.auto_summarize_on_ocr if auto_summarize is None else auto_summarize,
            "explicit_name": bool(clean_name),
            "explicit_folder": bool(rel_folder.parts),
        }

        state = JobState(
            job_id=job_id,
            label=label,
            input_dir=input_path,
            output_dir=output_dir.resolve(),
            archive_root=archive_base.resolve() if source_mode == "upload" else None,
            work_dir=job_dir.resolve(),
            provider=config.provider,
            model=config.model,
            dpi=config.dpi,
            folder=rel_folder.as_posix() if rel_folder.parts else "",
            tags=parsed_tags,
            expected_files=[t.output_path.resolve() for t in tasks],
            post_llm=post_llm,
            total_docs=len(tasks),
        )
        with lock:
            jobs[job_id] = state
        job_queue.put((state, tasks, config))
        return state.to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        with lock:
            state = jobs.get(job_id)
            if state is None:
                raise HTTPException(status_code=404, detail="Job not found.")
            if state.status in ("completed", "failed", "cancelled"):
                return state.to_dict()
            state.cancel_event.set()
            state.status = "cancelled" if state.status == "waiting" else "cancelling"
            client = state.active_client
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        with lock:
            return state.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        with lock:
            state = jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return state.to_dict()

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def get_job_log(job_id: str) -> PlainTextResponse:
        with lock:
            state = jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        log_path = state.work_dir / "ocr_pipeline.log"
        if not log_path.exists():
            return PlainTextResponse("Waiting for log output...")
        return PlainTextResponse(log_path.read_text(encoding="utf-8", errors="replace"))

    @app.get("/api/jobs/{job_id}/files/{file_path:path}")
    def get_job_file(job_id: str, file_path: str, download: bool = False) -> FileResponse:
        with lock:
            state = jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        target = state.resolve_output_file(file_path)
        if target is None:
            raise HTTPException(status_code=404, detail="Output file not found.")
        if download:
            return FileResponse(target, filename=target.name, media_type="text/markdown")
        return FileResponse(target, media_type="text/plain; charset=utf-8")

    @app.get("/api/jobs/{job_id}/download-all")
    def download_all(job_id: str) -> StreamingResponse:
        with lock:
            state = jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for md_file in sorted(state.expected_files):
                if md_file.is_file():
                    zf.write(md_file, arcname=state._rel_path(md_file))
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="ocr_{job_id}.zip"'},
        )

    return app
