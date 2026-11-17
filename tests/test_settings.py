"""Unit tests for cross-platform settings and OS utilities."""

from pathlib import Path

from ocr_pipeline.platform_utils import ensure_app_icon_file, generate_app_icon_image
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
