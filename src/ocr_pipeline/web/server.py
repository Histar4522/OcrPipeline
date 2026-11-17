"""FastAPI backend server for OcrPipeline Desktop & Web App."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import io
import mimetypes
from pathlib import Path
import platform
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
    start_ollama_service,
)
from ..platform_utils import generate_app_icon_image, open_in_file_manager
from ..runner import OcrFn, ProgressEvent, run_pipeline
from ..settings import AppSettings, default_jobs_dir, load_settings, save_settings

# Ensure .js/.css serve valid MIME types even on Windows where registry maps .js to text/plain
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

STATIC_DIR = Path(__file__).parent / "static"

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

    @app.delete("/api/models/{model_name:path}")
    def remove_ollama_model(model_name: str) -> dict:
        cfg = get_settings()
        try:
            delete_model(cfg.ollama_url, model_name)
            return {"ok": True, "deleted": model_name}
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/artifacts")
    def list_artifacts() -> dict:
        cfg = get_settings()
        archive_dir = Path(cfg.archive_dir).expanduser().resolve()
        items: list[dict] = []
        if archive_dir.exists():
            for md_file in sorted(archive_dir.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
                if ".ocr_cache" in md_file.parts:
                    continue
                stat = md_file.stat()
                snippet = md_file.read_text(encoding="utf-8", errors="replace")[:180].strip()
                items.append(
                    {
                        "path": md_file.relative_to(archive_dir).as_posix(),
                        "name": md_file.name,
                        "size_bytes": stat.st_size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                        "preview": snippet,
                    }
                )
        return {"archive_dir": str(archive_dir), "artifacts": items}

    @app.get("/api/artifacts/content")
    def get_artifact_content(path: str = Query(...), download: bool = False) -> FileResponse:
        cfg = get_settings()
        target = _resolve_artifact_path(Path(cfg.archive_dir), path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        if download:
            return FileResponse(target, filename=target.name, media_type="text/markdown")
        return FileResponse(target, media_type="text/plain; charset=utf-8")

    @app.patch("/api/artifacts/rename")
    def rename_artifact(payload: dict = Body(...)) -> dict:
        cfg = get_settings()
        archive_dir = Path(cfg.archive_dir)
        old_rel = str(payload.get("path", "")).strip()
        new_name = str(payload.get("new_name", "")).strip()
        if not old_rel or not new_name:
            raise HTTPException(status_code=400, detail="Both path and new_name are required.")
        if "/" in new_name or "\\" in new_name or ".." in new_name:
            raise HTTPException(status_code=400, detail="Invalid filename.")
        if not new_name.lower().endswith(".md"):
            new_name += ".md"
        src = _resolve_artifact_path(archive_dir, old_rel)
        if not src.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found.")
        dst = (src.parent / new_name).resolve()
        if dst.exists():
            raise HTTPException(status_code=409, detail="An artifact with that name already exists.")
        src.rename(dst)
        return {"ok": True, "path": dst.relative_to(archive_dir.expanduser().resolve()).as_posix()}

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

        if source_mode == "local":
            if not local_input_path.strip():
                raise HTTPException(status_code=400, detail="Local input path is required.")
            input_path = Path(local_input_path.strip()).expanduser()
            output_dir = (
                Path(local_output_path.strip()).expanduser()
                if local_output_path.strip()
                else archive_base / f"{input_path.stem or 'batch'}_{job_id[:6]}"
            )
            label = input_path.name or str(input_path)
        else:
            valid_files = [f for f in files if f.filename]
            if not valid_files:
                raise HTTPException(status_code=400, detail="Please select at least one PDF or image.")
            input_path = job_dir / "input"
            output_dir = archive_base / f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id[:4]}"
            _save_uploads(valid_files, input_path)
            label = f"{len(valid_files)} uploaded file(s)"

        config = OcrConfig(
            model=model.strip() or settings.default_model,
            ollama_url=settings.ollama_url,
            dpi=max(72, min(dpi or settings.dpi, 600)),
            max_image_dim=max(512, min(max_dim or settings.max_image_dim, 4096)),
            num_ctx=settings.num_ctx,
            system_prompt=settings.system_prompt,
            force=force,
        )

        try:
            tasks: list[DocumentTask] = discover_documents(input_path, output_dir, series_mode=series_mode)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not tasks:
            raise HTTPException(status_code=400, detail="No PDF or image files found in input.")

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
