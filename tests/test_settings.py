"""Unit tests for cross-platform settings and OS utilities."""

from pathlib import Path

from ocr_pipeline.desktop import DesktopApi
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


def test_desktop_api_save_url_writes_chosen_file(tmp_path: Path, monkeypatch: object) -> None:
    import ocr_pipeline.desktop as desk

    dest = tmp_path / "exported.md"

    class FakeWin:
        def create_file_dialog(self, *args: object, **kwargs: object) -> tuple[str, ...]:
            return (str(dest),)

    class FakeResp:
        content = b"# Saved Markdown\n"

        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(desk.httpx, "get", lambda url, timeout=60.0: FakeResp())  # type: ignore[attr-defined]
    api = DesktopApi("http://127.0.0.1:9999", {"win": FakeWin()})  # type: ignore[dict-item]
    res = api.save_url("/api/artifacts/content?path=a.md&download=true", "a.md")
    assert res["saved"] is True
    assert dest.read_bytes() == b"# Saved Markdown\n"


def test_notification_mode_normalization_and_conditions(tmp_path: Path) -> None:
    import json

    from ocr_pipeline.desktop import should_send_notification
    from ocr_pipeline.settings import normalize_notification_mode

    assert normalize_notification_mode(True) == "always"
    assert normalize_notification_mode(False) == "never"
    assert normalize_notification_mode("not_in_focus") == "not_in_focus"
    assert normalize_notification_mode("minimized") == "minimized"
    assert normalize_notification_mode("unknown") == "always"

    cfg_path = tmp_path / "legacy_settings.json"
    cfg_path.write_text(json.dumps({"desktop_notifications": False}), encoding="utf-8")
    assert load_settings(cfg_path).desktop_notifications == "never"

    assert should_send_notification("always", focused=True, in_tray=False, minimized=False) is True
    assert should_send_notification("never", focused=False, in_tray=True, minimized=True) is False
    assert should_send_notification("not_in_focus", focused=True, in_tray=False, minimized=False) is False
    assert should_send_notification("not_in_focus", focused=False, in_tray=False, minimized=False) is True
    assert should_send_notification("minimized", focused=False, in_tray=False, minimized=False) is False
    assert should_send_notification("minimized", focused=False, in_tray=True, minimized=False) is True
    assert should_send_notification("minimized", focused=False, in_tray=False, minimized=True) is True

    win_state = {"focused": True, "in_tray": False, "minimized": False}
    api = DesktopApi("http://127.0.0.1:9999", {"win": None}, window_state=win_state)
    api.set_focus(False)
    assert win_state["focused"] is False
    api.set_focus(True)
    assert win_state["focused"] is True


def test_legacy_settings_migration_and_api_key_masking(tmp_path: Path, monkeypatch: object) -> None:
    import json
    from ocr_pipeline.settings import (
        MASKED_SECRET,
        apply_provider_updates,
        mask_settings_for_api,
        resolve_llm_endpoint,
        resolve_vision_endpoint,
    )

    cfg_path = tmp_path / "old_settings.json"
    cfg_path.write_text(
        json.dumps({"ollama_url": "http://10.0.0.5:11434", "default_model": "deepseek-ocr"}),
        encoding="utf-8",
    )
    migrated = load_settings(cfg_path)
    assert migrated.vision_provider == "ollama"
    assert migrated.vision_model == "deepseek-ocr"
    assert migrated.providers["ollama"]["base_url"] == "http://10.0.0.5:11434"
    assert resolve_llm_endpoint(migrated) is None

    apply_provider_updates(migrated, {"openai": {"base_url": "https://api.openai.com/v1", "api_key": "sk-secret-999"}})
    masked = mask_settings_for_api(migrated)
    assert masked["providers"]["openai"]["api_key"] == MASKED_SECRET
    assert masked["providers"]["openai"]["has_key"] is True

    # Sending masked placeholder back must preserve the real key
    apply_provider_updates(migrated, {"openai": {"api_key": MASKED_SECRET}})
    assert migrated.providers["openai"]["api_key"] == "sk-secret-999"

    migrated.vision_provider = "openai"
    migrated.vision_model = "gpt-4.1-mini"
    vis_ep = resolve_vision_endpoint(migrated)
    assert vis_ep.provider_id == "openai"
    assert vis_ep.api_key == "sk-secret-999"

    migrated.llm_enabled = True
    migrated.llm_provider = "ollama"
    migrated.llm_model = "qwen3:1.7b"
    llm_ep = resolve_llm_endpoint(migrated)
    assert llm_ep is not None and llm_ep.model == "qwen3:1.7b"
    assert resolve_vision_endpoint(migrated).provider_id == "openai"
