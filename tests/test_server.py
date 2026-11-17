"""Tests for the FastAPI Desktop & Web UI server."""

import io
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image

from ocr_pipeline.config import OcrConfig
from ocr_pipeline.settings import AppSettings, save_settings
from ocr_pipeline.web.server import create_app


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (120, 60), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _make_test_app(tmp_path: Path, ocr_fn: object = None, on_job_finished: object = None) -> TestClient:
    settings_path = tmp_path / "settings.json"
    archive_dir = tmp_path / "archive"
    save_settings(AppSettings(archive_dir=str(archive_dir), ollama_url="http://127.0.0.1:59999"), settings_path)
    app = create_app(
        jobs_root=tmp_path / "jobs",
        settings_path=settings_path,
        ocr_fn=ocr_fn,  # type: ignore[arg-type]
        on_job_finished=on_job_finished,  # type: ignore[arg-type]
    )
    return TestClient(app)


def test_web_ui_serves_html_health_and_models(tmp_path: Path) -> None:
    client = _make_test_app(tmp_path)

    home = client.get("/")
    assert home.status_code == 200
    assert "OcrPipeline" in home.text
    assert "bootstrap" in home.text.lower()
    assert "katex" in home.text.lower()

    for asset in ("/static/app.js", "/static/markdown.js"):
        js = client.get(asset)
        assert js.status_code == 200
        assert js.headers["content-type"].startswith(("application/javascript", "text/javascript"))

    icon = client.get("/api/icon.png")
    assert icon.status_code == 200
    assert icon.headers["content-type"] == "image/png"

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["setup_completed"] is False
    assert health.json()["version"] == "0.1.0"
    assert health.json()["license"] == "MIT OR Apache-2.0"
    assert "Weien Xu" in health.json()["copyright"]

    models = client.get("/api/models")
    assert models.status_code == 200
    assert models.json()["online"] is False
    assert isinstance(models.json()["installed"], bool)
    assert len(models.json()["recommended"]) >= 3


def test_ollama_start_endpoint_when_missing(tmp_path: Path, monkeypatch: object) -> None:
    import ocr_pipeline.web.server as srv

    monkeypatch.setattr(srv, "start_ollama_service", lambda url: {"online": False, "installed": False, "started": False, "message": "not found"})  # type: ignore[attr-defined]
    client = _make_test_app(tmp_path)
    res = client.post("/api/ollama/start")
    assert res.status_code == 200
    assert res.json()["installed"] is False


def test_settings_and_artifact_management(tmp_path: Path) -> None:
    client = _make_test_app(tmp_path)

    updated = client.put(
        "/api/settings",
        json={"dpi": 300, "setup_completed": True, "default_model": "glm-ocr", "obsidian_mode": True},
    )
    assert updated.status_code == 200
    assert updated.json()["dpi"] == 300
    assert updated.json()["setup_completed"] is True
    assert updated.json()["obsidian_mode"] is True

    archive_dir = tmp_path / "archive"
    assert (archive_dir / ".obsidian" / "app.json").exists()

    folder_resp = client.post("/api/artifacts/folder", json={"folder": "Physics/Mechanics"})
    assert folder_resp.status_code == 200
    assert folder_resp.json()["folder"] == "Physics/Mechanics"

    (archive_dir / "note1.md").write_text("# Sample Note\nQuantum mechanics #physics", encoding="utf-8")

    listed = client.get("/api/artifacts", params={"q": "quantum", "tag": "physics"})
    assert listed.status_code == 200
    assert [a["path"] for a in listed.json()["artifacts"]] == ["note1.md"]
    assert listed.json()["obsidian_mode"] is True

    tagged = client.patch("/api/artifacts/tags", json={"path": "note1.md", "tags": ["lecture", "exam"]})
    assert tagged.status_code == 200
    assert tagged.json()["tags"] == ["lecture", "exam"]

    moved = client.patch(
        "/api/artifacts/move",
        json={"path": "note1.md", "new_name": "renamed_note", "target_folder": "Physics/Mechanics"},
    )
    assert moved.status_code == 200
    assert moved.json()["path"] == "Physics/Mechanics/renamed_note.md"

    filtered = client.get("/api/artifacts", params={"folder": "Physics", "tag": "exam"})
    assert [a["path"] for a in filtered.json()["artifacts"]] == ["Physics/Mechanics/renamed_note.md"]

    content = client.get("/api/artifacts/content", params={"path": "Physics/Mechanics/renamed_note.md"})
    assert content.status_code == 200
    assert "Quantum mechanics" in content.text

    deleted = client.delete("/api/artifacts", params={"path": "Physics/Mechanics/renamed_note.md"})
    assert deleted.status_code == 200
    assert client.get("/api/artifacts").json()["artifacts"] == []


def test_job_naming_folder_tags_obsidian_and_eta(tmp_path: Path) -> None:
    seen_eta: list[float | None] = []

    def fake_ocr(image_b64: str, cfg: OcrConfig, client: object) -> str:
        time.sleep(0.02)
        return "Integral calculus notes"

    client = _make_test_app(tmp_path, ocr_fn=fake_ocr)
    client.put("/api/settings", json={"obsidian_mode": True})

    resp = client.post(
        "/api/jobs",
        data={
            "source_mode": "upload",
            "job_name": "Calculus_Week1",
            "folder": "Math/Notes",
            "tags": "calculus, handwritten",
            "series_mode": "true",
        },
        files=[
            ("files", ("p1.png", _png_bytes(), "image/png")),
            ("files", ("p2.png", _png_bytes(), "image/png")),
        ],
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    deadline = time.time() + 5.0
    job_data = resp.json()
    while job_data["status"] not in ("completed", "failed") and time.time() < deadline:
        time.sleep(0.02)
        job_data = client.get(f"/api/jobs/{job_id}").json()
        seen_eta.append(job_data.get("eta_seconds"))

    assert job_data["status"] == "completed"
    assert job_data["files"] == ["Calculus_Week1.md"]
    assert job_data["eta_seconds"] == 0.0
    assert job_data["elapsed_seconds"] >= 0.0

    md_text = (tmp_path / "archive" / "Math" / "Notes" / "Calculus_Week1.md").read_text(encoding="utf-8")
    assert 'title: "Calculus_Week1"' in md_text
    assert "ocr_pipeline: true" in md_text
    assert "  - calculus" in md_text
    assert "Integral calculus notes" in md_text


def test_web_ui_upload_job_completes_and_triggers_callback(tmp_path: Path) -> None:
    finished_statuses: list[str] = []

    def fake_ocr(image_b64: str, cfg: OcrConfig, client: object) -> str:
        assert len(image_b64) > 20
        return "# Handwritten Note\n\n[Figure: simple diagram]"

    client = _make_test_app(tmp_path, ocr_fn=fake_ocr, on_job_finished=lambda j: finished_statuses.append(j.status))

    resp = client.post(
        "/api/jobs",
        data={"source_mode": "upload", "model": "glm-ocr", "dpi": "150", "series_mode": "true"},
        files=[("files", ("page_1.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    deadline = time.time() + 5.0
    job_data = resp.json()
    while job_data["status"] not in ("completed", "failed") and time.time() < deadline:
        time.sleep(0.05)
        job_data = client.get(f"/api/jobs/{job_id}").json()

    assert job_data["status"] == "completed"
    assert job_data["completed_docs"] == 1
    assert job_data["files"] == ["input.md"]
    assert finished_statuses == ["completed"]

    md_resp = client.get(f"/api/jobs/{job_id}/files/input.md")
    assert md_resp.status_code == 200
    assert "Handwritten Note" in md_resp.text

    log_resp = client.get(f"/api/jobs/{job_id}/log")
    assert log_resp.status_code == 200
    assert "SAVED:" in log_resp.text

    zip_resp = client.get(f"/api/jobs/{job_id}/download-all")
    assert zip_resp.status_code == 200
    assert zip_resp.headers["content-type"] == "application/zip"


def test_web_ui_cancel_job(tmp_path: Path) -> None:
    started = threading.Event()

    def slow_ocr(image_b64: str, cfg: OcrConfig, client: object, cancel_event: threading.Event | None = None) -> str:
        started.set()
        if cancel_event is not None:
            cancel_event.wait(timeout=3.0)
        return "done"

    client = _make_test_app(tmp_path, ocr_fn=slow_ocr)
    resp = client.post(
        "/api/jobs",
        data={"source_mode": "upload", "model": "glm-ocr"},
        files=[("files", ("page_1.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert started.wait(timeout=2.0)

    cancel_resp = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200

    deadline = time.time() + 3.0
    job_data = cancel_resp.json()
    while job_data["status"] not in ("cancelled", "completed", "failed") and time.time() < deadline:
        time.sleep(0.05)
        job_data = client.get(f"/api/jobs/{job_id}").json()

    assert job_data["status"] == "cancelled"
