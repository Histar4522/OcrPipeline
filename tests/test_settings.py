"""Unit tests for cross-platform settings and OS utilities."""

from pathlib import Path

from ocr_pipeline.ollama_client import start_ollama_service
from ocr_pipeline.platform_utils import ensure_app_icon_file, ensure_window_icon_file, generate_app_icon_image
from ocr_pipeline.settings import AppSettings, load_settings, save_settings


def test_settings_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg" / "settings.json"
    archive_dir = tmp_path / "archive"
    settings = AppSettings(
        setup_completed=True,
        default_model="glm-ocr",
        archive_dir=str(archive_dir),
        dpi=300,
        minimize_to_tray=False,
    )
    save_settings(settings, cfg_path)

    loaded = load_settings(cfg_path)
    assert loaded.setup_completed is True
    assert loaded.default_model == "glm-ocr"
    assert loaded.dpi == 300
    assert loaded.minimize_to_tray is False
    assert archive_dir.exists()


def test_app_icon_generation(tmp_path: Path) -> None:
    img = generate_app_icon_image(64)
    assert img.size == (64, 64)
    icon_file = ensure_app_icon_file(tmp_path)
    assert icon_file.exists()
    assert icon_file.stat().st_size > 100
    win_icon = ensure_window_icon_file(tmp_path)
    assert win_icon.exists()
    assert win_icon.stat().st_size > 100


def test_start_ollama_skips_duplicate_spawn(monkeypatch: object) -> None:
    import ocr_pipeline.ollama_client as oc

    spawned: list[list[str]] = []
    monkeypatch.setattr(oc, "find_ollama_executable", lambda: "ollama")  # type: ignore[attr-defined]
    monkeypatch.setattr(oc, "_is_ollama_process_running", lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setattr(oc, "_poll_ollama_ready", lambda url, wait: True)  # type: ignore[attr-defined]
    monkeypatch.setattr(oc.subprocess, "Popen", lambda *a, **k: spawned.append(list(a[0])))  # type: ignore[attr-defined]

    res = start_ollama_service("http://127.0.0.1:59999", wait_seconds=0.1)
    assert res["online"] is True
    assert res["started"] is False
    assert spawned == []
