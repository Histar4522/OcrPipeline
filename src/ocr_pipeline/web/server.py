"""FastAPI backend server for OcrPipeline Desktop & Web App."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import io
import mimetypes
from pathlib import Path
import platform
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
from .. import __version__
from ..artifacts import (
    ensure_obsidian_vault,
    normalize_tags,
    sanitize_rel_folder,
    scan_artifacts,
    update_artifact_tags,
)
from ..config import DEFAULT_SYSTEM_PROMPT, OcrConfig
from ..discovery import DocumentTask, discover_documents
from ..ollama_client import (
    OllamaError,
    check_ollama_ready,
    delete_model,
    find_ollama_executable,
    list_models_detailed,
    ocr_image_b64,
    pull_model,
    pull_model_stream,
    start_ollama_service,
)
from ..platform_utils import generate_app_icon_image, get_system_info, open_in_file_manager
from ..runner import OcrFn, ProgressEvent, run_pipeline
from ..settings import AppSettings, default_jobs_dir, load_settings, save_settings

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
    model: str
    dpi: int
    status: str = "queued"
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
    error: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    active_client: httpx.Client | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        files = []
        if self.output_dir.exists():
            files = sorted(
                p.relative_to(self.output_dir).as_posix()
                for p in self.output_dir.rglob("*.md")
                if ".ocr_cache" not in p.parts
            )
        return {
            "job_id": self.job_id,
            "label": self.label,
            "output_dir": str(self.output_dir),
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


def _save_uploads(files: list[UploadFile], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for upload in files:
        if not upload.filename:
            continue
        rel = Path(upload.filename.replace("\\", "/"))
        safe_parts = [p for p in rel.parts if p not in ("", ".", "..", "/", "\\")]
        if not safe_parts:
            continue
        target = dest_dir.joinpath(*safe_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            while chunk := upload.file.read(1024 * 1024):
                out.write(chunk)


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
    app = FastAPI(title="OcrPipeline", version="0.2.0")
    root = (jobs_root or default_jobs_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, JobState] = {}
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
    effective_ocr_fn: OcrFn = ocr_fn or ocr_image_b64

    def get_settings() -> AppSettings:
        cfg = load_settings(settings_path)
        if ollama_url:
            cfg.ollama_url = ollama_url
        return cfg

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
            "version": __version__,
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
        data = cfg.to_dict()
        data["default_system_prompt"] = DEFAULT_SYSTEM_PROMPT
        return data

    @app.put("/api/settings")
    def update_settings(payload: dict = Body(...)) -> dict:
        current = get_settings()
        if payload.get("reset_prompt"):
            payload["system_prompt"] = DEFAULT_SYSTEM_PROMPT
        allowed = current.to_dict()
        for key in allowed:
            if key in payload:
                setattr(current, key, payload[key])
        current.dpi = max(72, min(int(current.dpi), 600))
        current.max_image_dim = max(512, min(int(current.max_image_dim), 4096))
        current.num_ctx = max(1024, min(int(current.num_ctx), 32768))
        if not current.archive_dir.strip():
            current.archive_dir = str(AppSettings().archive_dir)
        save_settings(current, settings_path)
        data = current.to_dict()
        data["default_system_prompt"] = DEFAULT_SYSTEM_PROMPT
        return data

    @app.get("/api/models")
    def list_models() -> dict:
        cfg = get_settings()
        installed = find_ollama_executable() is not None
        try:
            detailed = list_models_detailed(cfg.ollama_url, timeout=3.0)
            return {
                "online": True,
                "installed": True,
                "models": [m["name"] for m in detailed],
                "detailed": detailed,
                "recommended": RECOMMENDED_MODELS,
                "default_model": cfg.default_model,
                "ollama_url": cfg.ollama_url,
            }
        except httpx.HTTPError as exc:
            return {
                "online": False,
                "installed": installed,
                "models": [],
                "detailed": [],
                "recommended": RECOMMENDED_MODELS,
                "default_model": cfg.default_model,
                "ollama_url": cfg.ollama_url,
                "error": str(exc),
            }

    @app.post("/api/ollama/start")
    def start_ollama() -> dict:
        cfg = get_settings()
        return start_ollama_service(cfg.ollama_url)

    @app.post("/api/models/pull")
    def pull_ollama_model(model: str = Form(...)) -> dict:
        cfg = get_settings()
        target = model.strip()
        if not target:
            raise HTTPException(status_code=400, detail="Model name is required.")
        try:
            pull_model(cfg.ollama_url, target)
            return {"ok": True, "model": target}
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/models/pull-async")
    def start_model_pull(model: str = Form(...)) -> dict:
        cfg = get_settings()
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
                pull_model_stream(cfg.ollama_url, target, on_progress=on_progress)
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
    def remove_ollama_model(model_name: str) -> dict:
        cfg = get_settings()
        try:
            delete_model(cfg.ollama_url, model_name)
            return {"ok": True, "deleted": model_name}
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/artifacts")
    def list_artifacts(
        q: str = Query(""),
        folder: str = Query(""),
        tag: str = Query(""),
        sort: str = Query("newest"),
    ) -> dict:
        cfg = get_settings()
        result = scan_artifacts(Path(cfg.archive_dir), query=q, folder=folder, tag=tag, sort=sort)
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
        model: str = Form(""),
        dpi: int = Form(0),
        max_dim: int = Form(0),
        series_mode: bool = Form(False),
        force: bool = Form(False),
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
            default_sub = archive_base / rel_folder if rel_folder.parts else archive_base / f"{input_path.stem or 'batch'}_{job_id[:6]}"
            output_dir = Path(local_output_path.strip()).expanduser() if local_output_path.strip() else default_sub
            label = clean_name or input_path.name or str(input_path)
        else:
            valid_files = [f for f in files if f.filename]
            if not valid_files:
                raise HTTPException(status_code=400, detail="Please select at least one PDF or image.")
            input_path = job_dir / "input"
            if rel_folder.parts:
                output_dir = archive_base / rel_folder
            elif clean_name:
                output_dir = archive_base
            else:
                output_dir = archive_base / f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id[:4]}"
            _save_uploads(valid_files, input_path)
            label = clean_name or f"{len(valid_files)} uploaded file(s)"

        config = OcrConfig(
            model=model.strip() or settings.default_model,
            ollama_url=settings.ollama_url,
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

        if ocr_fn is None:
            try:
                check_ollama_ready(config.ollama_url, config.model)
            except OllamaError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        state = JobState(
            job_id=job_id,
            label=label,
            input_dir=input_path,
            output_dir=output_dir.resolve(),
            model=config.model,
            dpi=config.dpi,
            folder=rel_folder.as_posix() if rel_folder.parts else "",
            tags=parsed_tags,
            total_docs=len(tasks),
        )
        with lock:
            jobs[job_id] = state

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

        def worker() -> None:
            with lock:
                if state.cancel_event.is_set():
                    state.status = "cancelled"
                    return
                state.status = "running"
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
                )
                with lock:
                    state.active_client = None
                    state.completed_docs = summary.processed_docs + summary.skipped_docs
                    state.skipped_docs = summary.skipped_docs
                    state.failed_pages = summary.failed_pages
                    state.status = "cancelled" if summary.cancelled else "completed"
            except Exception as exc:
                with lock:
                    state.active_client = None
                    if state.cancel_event.is_set():
                        state.status = "cancelled"
                    else:
                        state.status = "failed"
                        state.error = str(exc)
            if on_job_finished is not None:
                on_job_finished(state)

        threading.Thread(target=worker, daemon=True).start()
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
            state.status = "cancelling"
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
        log_path = state.output_dir / "ocr_pipeline.log"
        if not log_path.exists():
            return PlainTextResponse("Waiting for log output...")
        return PlainTextResponse(log_path.read_text(encoding="utf-8", errors="replace"))

    @app.get("/api/jobs/{job_id}/files/{file_path:path}")
    def get_job_file(job_id: str, file_path: str, download: bool = False) -> FileResponse:
        with lock:
            state = jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        target = (state.output_dir / file_path).resolve()
        if not target.is_relative_to(state.output_dir) or not target.is_file():
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
            for md_file in sorted(state.output_dir.rglob("*.md")):
                if ".ocr_cache" not in md_file.parts:
                    zf.write(md_file, arcname=md_file.relative_to(state.output_dir).as_posix())
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="ocr_{job_id}.zip"'},
        )

    return app
