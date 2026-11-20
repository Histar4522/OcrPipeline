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
    assert migrated.vision_provider in ("ollama", "ollama_local")
    assert migrated.vision_credential_id == "cred_ollama_local"
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


def test_multi_credentials_unique_names_and_singleton_local(tmp_path: Path) -> None:
    import pytest
    from ocr_pipeline.settings import (
        DuplicateCredentialError,
        MASKED_SECRET,
        OLLAMA_LOCAL_CRED_ID,
        delete_credential,
        mask_settings_for_api,
        resolve_llm_endpoint,
        resolve_vision_endpoint,
        upsert_credential,
    )

    s = AppSettings(archive_dir=str(tmp_path / "archive"))
    garage = upsert_credential(
        s,
        {"name": "Ollama Garage Server", "provider": "ollama", "base_url": "http://192.168.1.50:11434"},
    )
    ds = upsert_credential(
        s,
        {"name": "DeepSeek Primary", "provider": "deepseek", "api_key": "sk-ds-123"},
    )

    # Duplicate credential names (case-insensitive) and duplicate ollama_local must fail
    with pytest.raises(DuplicateCredentialError):
        upsert_credential(s, {"name": "ollama garage server", "provider": "ollama", "base_url": "http://10.0.0.2:11434"})
    with pytest.raises(ValueError, match="already exists"):
        upsert_credential(s, {"name": "Second Local", "provider": "ollama_local"})

    s.vision_credential_id = str(garage["id"])
    s.vision_provider = "ollama"
    s.vision_model = "glm-ocr"
    vis = resolve_vision_endpoint(s)
    assert vis.base_url == "http://192.168.1.50:11434"
    assert vis.credential_name == "Ollama Garage Server"

    s.llm_enabled = True
    s.llm_credential_id = str(ds["id"])
    s.llm_provider = "deepseek"
    s.llm_model = "deepseek-chat"
    llm = resolve_llm_endpoint(s)
    assert llm is not None and llm.api_key == "sk-ds-123" and llm.credential_name == "DeepSeek Primary"

    masked = mask_settings_for_api(s)
    ds_masked = next(c for c in masked["credentials"] if c["id"] == ds["id"])
    assert ds_masked["api_key"] == MASKED_SECRET

    # Disabling Ollama Local works; deleting Ollama Local is blocked
    upsert_credential(s, {"enabled": False}, cred_id=OLLAMA_LOCAL_CRED_ID)
    assert any(c["id"] == OLLAMA_LOCAL_CRED_ID and c["enabled"] is False for c in s.credentials)
    with pytest.raises(ValueError, match="cannot be deleted"):
        delete_credential(s, OLLAMA_LOCAL_CRED_ID)

    # Deleting active LLM credential cleans up role binding safely
    delete_credential(s, str(ds["id"]))
    assert s.llm_enabled is False
    assert s.llm_credential_id == OLLAMA_LOCAL_CRED_ID


def test_single_instance_lock_activation_and_duplicate_exit(tmp_path: Path, monkeypatch: object) -> None:
    from fastapi.testclient import TestClient

    import ocr_pipeline.desktop as desk
    from ocr_pipeline.settings import AppSettings, save_settings
    from ocr_pipeline.web.server import create_app

    lock_dir = tmp_path / "instance"
    g1 = desk.SingleInstanceGuard(lock_dir)
    g2 = desk.SingleInstanceGuard(lock_dir)
    assert g1.acquire() is True
    assert g2.acquire() is False

    g1.write_info(45123, "tok-secret")
    assert g2.read_info() == {"pid": g2.read_info()["pid"], "port": 45123, "token": "tok-secret"}  # type: ignore[index]

    # Verify /api/desktop/activate auth and callback behavior
    archive_dir = tmp_path / "archive"
    cfg_path = tmp_path / "settings.json"
    save_settings(AppSettings(archive_dir=str(archive_dir)), cfg_path)
    web_only = TestClient(create_app(jobs_root=tmp_path / "jobs1", settings_path=cfg_path))
    assert web_only.post("/api/desktop/activate", json={"token": "tok-secret"}).status_code == 404

    activations: list[tuple[str, str, str]] = []
    desk_client = TestClient(
        create_app(
            jobs_root=tmp_path / "jobs2",
            settings_path=cfg_path,
            on_activate=lambda t, m, v: activations.append((t, m, v)) is None,
            activate_token="tok-secret",
        )
    )
    assert desk_client.post("/api/desktop/activate", json={"token": "wrong"}).status_code == 403

    monkeypatch.setattr(
        desk.httpx,
        "post",
        lambda url, json=None, timeout=2.0: desk_client.post("/api/desktop/activate", json=json),
    )  # type: ignore[attr-defined]
    assert desk.notify_existing_instance(g2, wait_seconds=0.3) is True
    assert len(activations) == 1
    assert activations[0][0] == "Already Running"
    assert activations[0][2] == "danger"

    # launch_desktop_app returns 1 immediately when lock is already held & notifies existing window
    fallback_alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(desk, "_show_fallback_error_alert", lambda t, m: fallback_alerts.append((t, m)))  # type: ignore[attr-defined]
    assert desk.launch_desktop_app(lock_dir=lock_dir) == 1
    assert len(activations) == 2
    assert fallback_alerts == []

    # When existing instance does not respond, launch_desktop_app still exits 1 with fallback alert
    monkeypatch.setattr(desk, "notify_existing_instance", lambda *a, **k: False)  # type: ignore[attr-defined]
    assert desk.launch_desktop_app(lock_dir=lock_dir) == 1
    assert len(fallback_alerts) == 1

    # Releasing allows acquiring again and removes instance info file
    g1.release()
    assert not g1.info_file.exists()
    assert g2.acquire() is True
    g2.release()
