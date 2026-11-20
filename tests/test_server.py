"""Tests for the FastAPI Desktop & Web UI server."""

import io
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image

from ocr_pipeline import __version__
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
    assert 'id="wizardView"' in home.text
    assert 'id="wizardModal"' not in home.text
    assert 'id="setAutoUpdate"' not in home.text
    for elem_id in (
        "navVisualBadge",
        "navLlmBadge",
        "selectFilesBtn",
        "selectFolderBtn",
        "clearSelectedFilesBtn",
        "selectedFilesList",
        "settingsUpdateCard",
        "checkUpdatesBtn",
        "installUpdateBtn",
        "setAutoCheckUpdates",
        "releaseHistorySelect",
        "newReleaseNotes",
        "releaseHistoryPreview",
        "jobVisualModelSelect",
        "jobAiOptionsBox",
        "artifactAiBtn",
        "aiFolderSummaryBtn",
        "visualConfigPane",
        "llmConfigPane",
        "credentialsPane",
        "localOllamaPane",
        "visualModelSelect",
        "llmModelSelect",
        "roleLlmEnabled",
        "addCredentialBtn",
        "credOllamaLocalToggle",
    ):
        assert f'id="{elem_id}"' in home.text
    for removed_id in ("aiOrganizeFolderBtn", "artifactMoveBtn", "aiOrganizeBtn", "aiOrganizeModal", "setOllamaUrl", "setDefaultModel"):
        assert f'id="{removed_id}"' not in home.text
    for val in ("always", "not_in_focus", "minimized", "never"):
        assert f'value="{val}"' in home.text

    app_js = client.get("/static/app.js")
    assert app_js.status_code == 200
    assert "STATUS_BADGE_MAP" in app_js.text
    assert 'waiting: "text-bg-info"' in app_js.text

    sys_info = client.get("/api/system-info")
    assert sys_info.status_code == 200
    assert sys_info.json()["ram_gb"] > 0
    assert sys_info.json()["cpu_count"] >= 1

    dialogs_js = client.get("/static/dialogs.js")
    assert dialogs_js.status_code == 200
    assert "runModal" in dialogs_js.text
    assert "hidden.bs.modal" in dialogs_js.text

    for asset in (
        "/static/app.js",
        "/static/markdown.js",
        "/static/dialogs.js",
        "/static/tag_input.js",
        "/static/updates.js",
        "/static/providers_ui.js",
        "/static/model_dropdown.js",
    ):
        js = client.get(asset)
        assert js.status_code == 200
        assert js.headers["content-type"].startswith(("application/javascript", "text/javascript"))

    icon = client.get("/api/icon.png")
    assert icon.status_code == 200
    assert icon.headers["content-type"] == "image/png"

    bad_url = client.post("/api/open-url", json={"url": "file:///C:/Windows/win.ini"})
    assert bad_url.status_code == 400

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["setup_completed"] is False
    assert health.json()["base_version"] == __version__
    assert health.json()["version"].startswith(f"{__version__}-devrun-")
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


def test_async_model_pull_progress(tmp_path: Path, monkeypatch: object) -> None:
    import ocr_pipeline.web.server as srv

    def fake_stream(url: str, model: str, on_progress: object) -> None:
        on_progress({"status": "downloading", "completed": 50 * 1024 * 1024, "total": 100 * 1024 * 1024, "percent": 50.0})  # type: ignore[operator]

    monkeypatch.setattr(srv, "pull_model_stream", fake_stream)  # type: ignore[attr-defined]
    client = _make_test_app(tmp_path)
    start = client.post("/api/models/pull-async", data={"model": "glm-ocr"})
    assert start.status_code == 200

    deadline = time.time() + 3.0
    st = start.json()
    while not st["done"] and time.time() < deadline:
        time.sleep(0.05)
        st = client.get("/api/models/pull-status").json()

    assert st["done"] is True
    assert st["status"] == "completed"
    assert st["percent"] == 100.0


def test_settings_and_artifact_management(tmp_path: Path) -> None:
    client = _make_test_app(tmp_path)

    updated = client.put(
        "/api/settings",
        json={"dpi": 300, "setup_completed": True, "default_model": "glm-ocr", "obsidian_mode": True, "theme": "dark"},
    )
    assert updated.status_code == 200
    assert updated.json()["dpi"] == 300
    assert updated.json()["setup_completed"] is True
    assert updated.json()["obsidian_mode"] is True
    assert updated.json()["theme"] == "dark"

    archive_dir = tmp_path / "archive"
    assert (archive_dir / ".obsidian" / "app.json").exists()

    folder_resp = client.post("/api/artifacts/folder", json={"folder": "Physics/Mechanics"})
    assert folder_resp.status_code == 200
    assert folder_resp.json()["folder"] == "Physics/Mechanics"

    (archive_dir / "note1.md").write_text("# Sample Note\nQuantum mechanics #physics #1 #2", encoding="utf-8")

    listed = client.get("/api/artifacts", params={"q": "quantum", "tag": "physics"})
    assert listed.status_code == 200
    assert [a["path"] for a in listed.json()["artifacts"]] == ["note1.md"]
    assert listed.json()["artifacts"][0]["tags"] == ["physics"]
    assert listed.json()["obsidian_mode"] is True

    cleared = client.patch("/api/artifacts/tags", json={"path": "note1.md", "tags": []})
    assert cleared.status_code == 200
    assert cleared.json()["tags"] == []
    assert client.get("/api/artifacts", params={"q": "quantum"}).json()["artifacts"][0]["tags"] == []

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

    root_browse = client.get("/api/artifacts", params={"cwd": ""})
    assert root_browse.json()["artifacts"] == []
    assert root_browse.json()["child_folders"] == [{"name": "Physics", "path": "Physics", "doc_count": 1}]

    sub_browse = client.get("/api/artifacts", params={"cwd": "Physics/Mechanics"})
    assert [b["name"] for b in sub_browse.json()["breadcrumbs"]] == ["Archive", "Physics", "Mechanics"]
    assert [a["name"] for a in sub_browse.json()["artifacts"]] == ["renamed_note.md"]

    ren_dir = client.patch("/api/artifacts/folder", json={"folder": "Physics/Mechanics", "new_name": "Optics"})
    assert ren_dir.status_code == 200
    assert ren_dir.json()["folder"] == "Physics/Optics"
    assert "Physics/Optics" in client.get("/api/folders").json()["folders"]

    content = client.get("/api/artifacts/content", params={"path": "Physics/Optics/renamed_note.md"})
    assert content.status_code == 200
    assert "Quantum mechanics" in content.text

    non_empty_del = client.delete("/api/artifacts/folder", params={"folder": "Physics/Optics"})
    assert non_empty_del.status_code == 409

    deleted = client.delete("/api/artifacts", params={"path": "Physics/Optics/renamed_note.md"})
    assert deleted.status_code == 200
    assert not (archive_dir / "Physics" / "Optics" / "renamed_note.md.meta").exists()
    empty_del = client.delete("/api/artifacts/folder", params={"folder": "Physics", "recursive": "true"})
    assert empty_del.status_code == 200
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

    md_path = tmp_path / "archive" / "Math" / "Notes" / "Calculus_Week1.md"
    md_text = md_path.read_text(encoding="utf-8")
    assert 'title: "Calculus_Week1"' in md_text
    assert "ocr_pipeline: true" in md_text
    assert "tags:" not in md_text
    assert "summary:" not in md_text
    assert "Integral calculus notes" in md_text
    meta_path = tmp_path / "archive" / "Math" / "Notes" / "Calculus_Week1.md.meta"
    assert meta_path.is_file()
    import json
    assert json.loads(meta_path.read_text(encoding="utf-8"))["tags"] == ["calculus", "handwritten"]
    # Ensure internal cache and logs stay in work_dir, keeping archive_dir clean
    assert not (tmp_path / "archive" / "Math" / "Notes" / "ocr_pipeline.log").exists()
    assert not (tmp_path / "archive" / "Math" / "Notes" / ".ocr_cache").exists()


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


def test_job_with_selected_paths_and_duplicate_names(tmp_path: Path) -> None:
    import json

    ocr_calls = 0

    def fake_ocr(image_b64: str, cfg: OcrConfig, client: object) -> str:
        nonlocal ocr_calls
        ocr_calls += 1
        return f"Page {ocr_calls}"

    client = _make_test_app(tmp_path, ocr_fn=fake_ocr)
    dir_a = tmp_path / "src_a"
    dir_b = tmp_path / "src_b"
    dir_a.mkdir()
    dir_b.mkdir()
    img_a = dir_a / "page.png"
    img_b = dir_b / "page.png"
    img_a.write_bytes(_png_bytes())
    img_b.write_bytes(_png_bytes())

    missing_resp = client.post(
        "/api/jobs",
        data={
            "source_mode": "upload",
            "selected_paths": json.dumps([{"path": str(tmp_path / "missing.png"), "rel": "missing.png"}]),
        },
    )
    assert missing_resp.status_code == 400

    resp = client.post(
        "/api/jobs",
        data={
            "source_mode": "upload",
            "series_mode": "true",
            "selected_paths": json.dumps([
                {"path": str(img_a), "rel": "page.png"},
                {"path": str(img_b), "rel": "page.png"},
            ]),
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    deadline = time.time() + 5.0
    job_data = resp.json()
    while job_data["status"] not in ("completed", "failed") and time.time() < deadline:
        time.sleep(0.05)
        job_data = client.get(f"/api/jobs/{job_id}").json()

    assert job_data["status"] == "completed"
    assert ocr_calls == 2


def test_desktop_file_selection_and_platform_open(tmp_path: Path, monkeypatch: object) -> None:
    import os
    from webview.util import parse_file_type

    from ocr_pipeline import desktop, platform_utils

    for spec in desktop._PICK_FILE_TYPES:
        desc, exts = parse_file_type(spec)
        assert desc and exts

    folder = tmp_path / "notes"
    hidden = folder / ".git"
    nested = folder / "sub"
    folder.mkdir()
    hidden.mkdir()
    nested.mkdir()
    f1 = folder / "p1.png"
    f2 = nested / "p2.pdf"
    f_hidden = hidden / "secret.png"
    f_ignored = folder / "readme.txt"
    for p in (f1, f2, f_hidden):
        p.write_bytes(_png_bytes())
    f_ignored.write_text("ignore", encoding="utf-8")

    collected = desktop.collect_supported_files([str(f1), str(folder)])
    assert [c["rel"] for c in collected] == ["p1.png", "notes/sub/p2.pdf"]

    opened: list[Path] = []
    revealed: list[Path] = []
    monkeypatch.setattr(desktop, "open_with_default_app", lambda p: opened.append(p))  # type: ignore[attr-defined]
    monkeypatch.setattr(desktop, "open_in_file_manager", lambda p: revealed.append(p))  # type: ignore[attr-defined]

    api = desktop.DesktopApi("http://127.0.0.1:8000", {"win": None})
    assert api.open_file(str(f1))["ok"] is False
    assert api.reveal_file(str(f1))["ok"] is False

    api.register_paths([str(f1)])
    assert api.open_file(str(f1)) == {"ok": True}
    assert api.reveal_file(str(f1)) == {"ok": True}
    assert opened == [f1.resolve()]
    assert revealed == [f1.resolve()]

    # Verify Windows explorer /select and os.startfile invocation
    monkeypatch.undo()
    popen_calls: list[list[str]] = []
    startfile_calls: list[str] = []
    monkeypatch.setattr(platform_utils.platform, "system", lambda: "Windows")  # type: ignore[attr-defined]
    monkeypatch.setattr(platform_utils.subprocess, "Popen", lambda args, **kw: popen_calls.append(args))  # type: ignore[attr-defined]
    monkeypatch.setattr(os, "startfile", lambda s: startfile_calls.append(s), raising=False)  # type: ignore[attr-defined]

    platform_utils.open_in_file_manager(f1)
    platform_utils.open_with_default_app(f1)
    assert popen_calls == [["explorer", "/select,", str(f1.resolve())]]
    assert startfile_calls == [str(f1.resolve())]


def test_job_queue_waiting_classification_and_cancel(tmp_path: Path) -> None:
    job1_started = threading.Event()
    release_job1 = threading.Event()
    processed_labels: list[str] = []

    def gated_ocr(image_b64: str, cfg: OcrConfig, client: object) -> str:
        if not job1_started.is_set():
            job1_started.set()
            release_job1.wait(timeout=4.0)
        processed_labels.append("page")
        return "done"

    client = _make_test_app(tmp_path, ocr_fn=gated_ocr)

    r1 = client.post(
        "/api/jobs",
        data={"source_mode": "upload", "job_name": "job_one"},
        files=[("files", ("p1.png", _png_bytes(), "image/png"))],
    )
    assert r1.status_code == 200
    j1 = r1.json()["job_id"]
    assert job1_started.wait(timeout=2.0)

    r2 = client.post(
        "/api/jobs",
        data={"source_mode": "upload", "job_name": "job_two"},
        files=[("files", ("p2.png", _png_bytes(), "image/png"))],
    )
    r3 = client.post(
        "/api/jobs",
        data={"source_mode": "upload", "job_name": "job_three"},
        files=[("files", ("p3.png", _png_bytes(), "image/png"))],
    )
    assert r2.status_code == 200 and r3.status_code == 200
    j2 = r2.json()["job_id"]
    j3 = r3.json()["job_id"]

    assert client.get(f"/api/jobs/{j2}").json()["status"] == "waiting"
    assert client.get(f"/api/jobs/{j3}").json()["status"] == "waiting"
    assert processed_labels == []

    cancel_j3 = client.post(f"/api/jobs/{j3}/cancel")
    assert cancel_j3.status_code == 200
    assert cancel_j3.json()["status"] == "cancelled"

    release_job1.set()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        s1 = client.get(f"/api/jobs/{j1}").json()["status"]
        s2 = client.get(f"/api/jobs/{j2}").json()["status"]
        if s1 == "completed" and s2 == "completed":
            break
        time.sleep(0.05)

    assert client.get(f"/api/jobs/{j1}").json()["status"] == "completed"
    assert client.get(f"/api/jobs/{j2}").json()["status"] == "completed"
    assert client.get(f"/api/jobs/{j3}").json()["status"] == "cancelled"
    assert len(processed_labels) == 2


def test_updater_version_asset_selection_and_endpoints(tmp_path: Path, monkeypatch: object) -> None:
    import httpx
    from ocr_pipeline import updater
    import ocr_pipeline.web.server as srv

    assert updater.parse_version("v0.2.10") > updater.parse_version("v0.2.2") > updater.parse_version("0.2.1")

    sample_assets = [
        {"name": "OcrPipeline-0.3.0-x64.msi", "size": 1024, "browser_download_url": "https://github.com/a/b/OcrPipeline-0.3.0-x64.msi"},
        {"name": "OcrPipeline-0.3.0-linux-x86_64.tar.gz", "size": 2048, "browser_download_url": "https://github.com/a/b/OcrPipeline-0.3.0-linux-x86_64.tar.gz"},
        {"name": "OcrPipeline-0.3.0-macOS-arm64.dmg", "size": 3072, "browser_download_url": "https://github.com/a/b/OcrPipeline-0.3.0-macOS-arm64.dmg"},
    ]
    assert updater.select_release_asset(sample_assets, system="Windows", machine="AMD64")["name"].endswith(".msi")  # type: ignore[index]
    assert updater.select_release_asset(sample_assets, system="Linux", machine="x86_64")["name"].endswith(".tar.gz")  # type: ignore[index]
    assert updater.select_release_asset(sample_assets, system="Darwin", machine="arm64")["name"].endswith(".dmg")  # type: ignore[index]

    fake_releases = [
        {
            "tag_name": "v9.9.9-draft",
            "draft": True,
            "prerelease": False,
            "body": "hidden",
            "assets": [],
        },
        {
            "tag_name": "v0.3.0",
            "name": "OcrPipeline v0.3.0",
            "body": "### New Features\n- Auto update test",
            "published_at": "2026-11-18T12:00:00Z",
            "html_url": "https://github.com/Histar4522/OcrPipeline/releases/tag/v0.3.0",
            "draft": False,
            "prerelease": False,
            "assets": sample_assets,
        },
        {
            "tag_name": "v0.2.2",
            "name": "OcrPipeline v0.2.2",
            "body": "Current release notes",
            "published_at": "2026-11-18T10:00:00Z",
            "html_url": "https://github.com/Histar4522/OcrPipeline/releases/tag/v0.2.2",
            "draft": False,
            "prerelease": False,
            "assets": [],
        },
    ]

    class FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> object:
            return fake_releases

    monkeypatch.setattr(updater.httpx, "get", lambda *a, **k: FakeResp())  # type: ignore[attr-defined]

    checked = updater.fetch_github_releases(current_version="0.2.2")
    assert checked["update_available"] is True
    assert checked["latest"]["tag_name"] == "v0.3.0"
    assert [r["tag_name"] for r in checked["releases"]] == ["v0.3.0", "v0.2.2"]
    assert checked["releases"][1]["is_current"] is True

    client = _make_test_app(tmp_path)
    saved = client.put("/api/settings", json={"auto_check_updates": False})
    assert saved.status_code == 200
    assert saved.json()["auto_check_updates"] is False
    assert "auto_update" not in saved.json()

    api_check = client.get("/api/updates/check")
    assert api_check.status_code == 200
    assert api_check.json()["latest"]["tag_name"] == "v0.3.0"

    bad_install = client.post("/api/updates/install", json={"tag": "v0.3.0"})
    assert bad_install.status_code == 400

    # Test untrusted URL rejection and valid download/install flow with mocked stream + Popen
    launched: list[list[str]] = []
    monkeypatch.setattr(updater.subprocess, "Popen", lambda args, **k: launched.append(list(args)))  # type: ignore[attr-defined]
    monkeypatch.setattr(updater.platform, "system", lambda: "Windows")  # type: ignore[attr-defined]

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"fake-msi-bytes", headers={"content-length": "14"})

    orig_stream = updater.httpx.stream
    monkeypatch.setattr(
        updater.httpx,
        "stream",
        lambda method, url, **kw: httpx.Client(transport=httpx.MockTransport(mock_handler)).stream(method, url),
    )  # type: ignore[attr-defined]
    res = updater.download_and_install_update(
        "https://github.com/Histar4522/OcrPipeline/releases/download/v0.3.0/OcrPipeline-0.3.0-x64.msi",
        "OcrPipeline-0.3.0-x64.msi",
        download_dir=tmp_path / "dl",
    )
    monkeypatch.setattr(updater.httpx, "stream", orig_stream)  # type: ignore[attr-defined]
    assert res["installed"] is True
    assert launched and launched[0][0] == "msiexec"
    assert (tmp_path / "dl" / "OcrPipeline-0.3.0-x64.msi").read_bytes() == b"fake-msi-bytes"
    assert callable(srv.fetch_github_releases)


def test_providers_endpoints_and_ai_artifact_gating(tmp_path: Path, monkeypatch: object) -> None:
    import json
    from ocr_pipeline import llm_features

    client = _make_test_app(tmp_path)
    prov_res = client.get("/api/providers")
    assert prov_res.status_code == 200
    prov_ids = {r["id"] for r in prov_res.json()["registry"]}
    assert {"ollama", "llama_cpp", "openai", "anthropic", "google", "deepseek", "xai", "qwen", "custom"}.issubset(prov_ids)

    archive_dir = tmp_path / "archive"
    (archive_dir / "Math").mkdir(parents=True)
    note = archive_dir / "note.md"
    note.write_text("# Linear Algebra\nEigenvalues and eigenvectors", encoding="utf-8")

    # Disabled LLM must reject all AI endpoints with 400
    for ep in ("/api/artifacts/ai/tags", "/api/artifacts/ai/rename", "/api/artifacts/ai/summarize", "/api/artifacts/ai/cleanup"):
        assert client.post(ep, json={"path": "note.md"}).status_code == 400
    assert client.post("/api/artifacts/ai/folder-summary", json={"cwd": ""}).status_code == 400

    # Enable optional Assistant LLM
    upd = client.put(
        "/api/settings",
        json={"llm_enabled": True, "llm_provider": "ollama", "llm_model": "qwen3:1.7b"},
    )
    assert upd.status_code == 200
    assert upd.json()["llm_available"] is True

    def fake_chat(endpoint, sys_prompt, user_prompt, **kw):
        if kw.get("json_mode"):
            if '"tags"' in sys_prompt:
                return json.dumps({"tags": ["linear-algebra", "math"]})
            if '"filename"' in sys_prompt or '"title"' in sys_prompt:
                return json.dumps({"filename": "Linear_Algebra_Notes.md"})
            if '"summary"' in sys_prompt:
                return json.dumps({"summary": "Covers eigenvalues cleanly."})
        if "Synthesize the document summaries inside this archive folder" in sys_prompt:
            return "### Overview\nArchive contains Linear Algebra notes."
        return "# Linear Algebra\nClean eigenvalues and eigenvectors."

    monkeypatch.setattr(llm_features, "chat_text", fake_chat)  # type: ignore[attr-defined]

    tags_res = client.post("/api/artifacts/ai/tags", json={"path": "note.md"})
    assert tags_res.status_code == 200
    assert tags_res.json()["tags"] == ["linear-algebra", "math"]

    ren_res = client.post("/api/artifacts/ai/rename", json={"path": "note.md"})
    assert ren_res.status_code == 200
    assert ren_res.json()["suggested_name"] == "Linear_Algebra_Notes.md"

    sum_res = client.post("/api/artifacts/ai/summarize", json={"path": "note.md"})
    assert sum_res.status_code == 200
    assert sum_res.json()["summary"] == "Covers eigenvalues cleanly."
    note_raw = note.read_text(encoding="utf-8")
    assert "tags:" not in note_raw
    assert "summary:" not in note_raw
    assert (archive_dir / "note.md.meta").is_file()

    clean_res = client.post("/api/artifacts/ai/cleanup", json={"path": "note.md"})
    assert clean_res.status_code == 200
    assert "Clean eigenvalues" in clean_res.json()["cleaned_markdown"]

    save_res = client.put("/api/artifacts/content", json={"path": "note.md", "content": clean_res.json()["cleaned_markdown"]})
    assert save_res.status_code == 200

    folder_sum = client.post("/api/artifacts/ai/folder-summary", json={"cwd": ""})
    assert folder_sum.status_code == 200
    assert folder_sum.json()["doc_count"] == 1
    assert "Linear Algebra notes" in folder_sum.json()["summary"]
    assert (archive_dir / ".folder.meta").is_file()

    moved = client.patch(
        "/api/artifacts/move",
        json={"path": "note.md", "new_name": "Linear_Algebra_Notes.md", "target_folder": "Math"},
    )
    assert moved.status_code == 200
    moved_meta = archive_dir / "Math" / "Linear_Algebra_Notes.md.meta"
    assert moved_meta.is_file()
    assert not (archive_dir / "note.md.meta").exists()
    assert json.loads(moved_meta.read_text(encoding="utf-8"))["summary"] == "Covers eigenvalues cleanly."
    # Verify summary & tag search indexes sidecar metadata cleanly
    found = client.get("/api/artifacts", params={"q": "eigenvalues cleanly"})
    assert [a["path"] for a in found.json()["artifacts"]] == ["Math/Linear_Algebra_Notes.md"]


def test_post_ocr_llm_automation_and_nonfatal_error(tmp_path: Path, monkeypatch: object) -> None:
    import ocr_pipeline.web.server as srv

    archive_dir = tmp_path / "archive"
    (archive_dir / "Biology").mkdir(parents=True)

    def fake_ocr(image_b64: str, cfg: OcrConfig, client: object) -> str:
        return "# Cellular Respiration\nATP synthesis inside mitochondria."

    client = _make_test_app(tmp_path, ocr_fn=fake_ocr)
    client.put(
        "/api/settings",
        json={"llm_enabled": True, "llm_provider": "ollama", "llm_model": "qwen3:1.7b"},
    )

    monkeypatch.setattr(
        srv,
        "analyze_artifact_all",
        lambda *a, **kw: {
            "tags": ["biology", "atp"],
            "filename": "Cellular_Respiration.md",
            "summary": "Explains ATP synthesis in mitochondria.",
        },
    )  # type: ignore[attr-defined]

    resp = client.post(
        "/api/jobs",
        data={
            "source_mode": "upload",
            "auto_tag": "true",
            "auto_summarize": "true",
            "auto_rename": "true",
        },
        files=[("files", ("scan.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    deadline = time.time() + 5.0
    job_data = resp.json()
    while job_data["status"] not in ("completed", "failed") and time.time() < deadline:
        time.sleep(0.05)
        job_data = client.get(f"/api/jobs/{job_id}").json()

    assert job_data["status"] == "completed"
    moved_file = archive_dir / "Cellular_Respiration.md"
    moved_meta = archive_dir / "Cellular_Respiration.md.meta"
    assert moved_file.is_file()
    assert moved_meta.is_file()
    content = moved_file.read_text(encoding="utf-8")
    # Tags and summaries must live in the .meta JSON sidecar, not inside Markdown
    assert "tags:" not in content
    assert "Explains ATP synthesis" not in content
    import json
    meta_json = json.loads(moved_meta.read_text(encoding="utf-8"))
    assert meta_json["tags"] == ["biology", "atp"]
    assert meta_json["summary"] == "Explains ATP synthesis in mitochondria."
    assert job_data["files"] == ["Cellular_Respiration.md"]
    assert client.get(f"/api/jobs/{job_id}/files/Cellular_Respiration.md").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/download-all").status_code == 200

    # Ensure an LLM exception does not fail the OCR job itself
    def failing_llm(*a: object, **kw: object) -> dict:
        raise RuntimeError("LLM timed out")

    monkeypatch.setattr(srv, "analyze_artifact_all", failing_llm)  # type: ignore[attr-defined]
    resp2 = client.post(
        "/api/jobs",
        data={"source_mode": "upload", "auto_tag": "true"},
        files=[("files", ("scan2.png", _png_bytes(), "image/png"))],
    )
    job2 = resp2.json()["job_id"]
    deadline = time.time() + 5.0
    j2 = resp2.json()
    while j2["status"] not in ("completed", "failed") and time.time() < deadline:
        time.sleep(0.05)
        j2 = client.get(f"/api/jobs/{job2}").json()
    assert j2["status"] == "completed"
    assert "LLM WARN" in client.get(f"/api/jobs/{job2}/log").text


def test_server_vram_estimate_endpoint(tmp_path: Path) -> None:
    client = _make_test_app(tmp_path)
    prov = client.get("/api/providers")
    assert prov.status_code == 200
    assert "vram_plan" in prov.json()["llm"]
    assert prov.json()["llm"]["auto_ctx"] is True

    est_local = client.post(
        "/api/llm/vram-estimate",
        json={"provider": "ollama", "model": "qwen3:1.7b"},
    )
    assert est_local.status_code == 200
    assert est_local.json()["is_local"] is True
    assert est_local.json()["recommended_max_ctx"] >= 2048
    assert est_local.json()["split_threshold_tokens"] >= 1200

    est_cloud = client.post(
        "/api/llm/vram-estimate",
        json={"provider": "openai", "model": "gpt-4.1-mini"},
    )
    assert est_cloud.status_code == 200
    assert est_cloud.json()["is_local"] is False
    assert est_cloud.json()["vram_at_recommended_gb"] == 0.0


def test_credentials_crud_and_provider_status(tmp_path: Path) -> None:
    seen_providers: list[str] = []

    def fake_ocr(image_b64: str, cfg: OcrConfig, client: object) -> str:
        seen_providers.append(f"{cfg.provider}@{cfg.base_url}")
        return "# OCR Output"

    client = _make_test_app(tmp_path, ocr_fn=fake_ocr)

    # Create two custom credentials: DeepSeek (with Vision OCR) and a second Ollama instance
    c1 = client.post(
        "/api/credentials",
        json={"name": "DeepSeek Account", "provider": "deepseek", "api_key": "sk-ds-test"},
    )
    assert c1.status_code == 200
    assert c1.json()["api_key"] == "••••••••"
    ds_id = c1.json()["id"]

    c2 = client.post(
        "/api/credentials",
        json={"name": "Ollama Garage Server", "provider": "ollama", "base_url": "http://192.168.1.99:11434"},
    )
    assert c2.status_code == 200
    garage_id = c2.json()["id"]

    # Unique credential names & singleton Ollama Local are enforced
    dup = client.post(
        "/api/credentials",
        json={"name": "ollama garage server", "provider": "ollama", "base_url": "http://10.0.0.1:11434"},
    )
    assert dup.status_code == 409
    dup_local = client.post(
        "/api/credentials",
        json={"name": "Another Local", "provider": "ollama_local"},
    )
    assert dup_local.status_code == 400
    assert client.delete("/api/credentials/cred_ollama_local").status_code == 400

    # Status endpoint reports offline/disabled accurately cleanly
    client.put(
        "/api/settings",
        json={
            "vision_credential_id": "cred_ollama_local",
            "vision_model": "glm-ocr",
            "llm_enabled": False,
        },
    )
    st1 = client.get("/api/providers/status")
    assert st1.status_code == 200
    assert st1.json()["vision"]["label"] == "glm-ocr (Ollama Local)"
    assert st1.json()["vision"]["status"] == "offline"
    assert st1.json()["vision"]["status_text"] == "Offline"
    assert st1.json()["llm"]["label"] == "Disabled"

    # Toggle Ollama Local off and confirm status reflects Disabled
    toggled = client.put("/api/credentials/cred_ollama_local", json={"provider": "ollama_local", "enabled": False})
    assert toggled.status_code == 200
    assert toggled.json()["enabled"] is False
    st2 = client.get("/api/providers/status")
    assert st2.json()["vision"]["status"] == "disabled"
    assert st2.json()["vision"]["status_text"] == "Disabled"

    # Route a job explicitly to DeepSeek credential by credential_id
    job = client.post(
        "/api/jobs",
        data={"source_mode": "upload", "credential_id": ds_id, "model": "deepseek-ocr"},
        files=[("files", ("p1.png", _png_bytes(), "image/png"))],
    )
    assert job.status_code == 200
    job_id = job.json()["job_id"]
    deadline = time.time() + 4.0
    while client.get(f"/api/jobs/{job_id}").json()["status"] not in ("completed", "failed") and time.time() < deadline:
        time.sleep(0.05)
    assert seen_providers == ["deepseek@https://api.deepseek.com/v1"]

    # Deleting Garage Server credential succeeds cleanly
    assert client.delete(f"/api/credentials/{garage_id}").status_code == 200


def test_recent_tags_ordering_and_suggestion_limit(tmp_path: Path) -> None:
    import os

    client = _make_test_app(tmp_path)
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("alpha.md", ["zebra", "ancient"], 1_000_000.0),
        ("beta.md", ["middle"], 2_000_000.0),
        ("gamma.md", ["fresh-one", "fresh-two"], 3_000_000.0),
    ]
    for name, tags, ts in specs:
        md = archive_dir / name
        md.write_text("# Content\n", encoding="utf-8")
        meta = archive_dir / f"{name}.meta"
        meta.write_text(f'{{"tags": {tags!r}}}'.replace("'", '"'), encoding="utf-8")
        os.utime(md, (ts, ts))
        os.utime(meta, (ts, ts))

    res = client.get("/api/artifacts")
    assert res.status_code == 200
    data = res.json()
    assert data["recent_tags"] == ["fresh-one", "fresh-two", "middle", "ancient", "zebra"]
    assert [t["name"] for t in data["tags"]] == ["ancient", "fresh-one", "fresh-two", "middle", "zebra"]
    assert all("last_used" in t for t in data["tags"])

    # Updating alpha.md.meta via API bumps its tags to the front of recent_tags
    upd = client.patch("/api/artifacts/tags", json={"path": "alpha.md", "tags": ["zebra"]})
    assert upd.status_code == 200
    res2 = client.get("/api/artifacts")
    assert res2.json()["recent_tags"][0] == "zebra"

    # Frontend tag_input defaults maxSuggestions to 3 rather than 12
    tag_js = client.get("/static/tag_input.js")
    assert tag_js.status_code == 200
    assert "maxSuggestions = 3" in tag_js.text
    assert "available.slice(0, maxSuggestions)" in tag_js.text
    assert "slice(0, 12)" not in tag_js.text
