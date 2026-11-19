"""Tests for cross-platform installer asset generation and packaging."""

from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))

from build_installer import linux_desktop_entry, package_linux_archive, resolve_installer_version  # noqa: E402
from ocr_pipeline import __version__  # noqa: E402
from prepare_assets import prepare_assets  # noqa: E402


def test_build_version_formats() -> None:
    beta_ver = resolve_installer_version(release=False)
    rel_ver = resolve_installer_version(release=True)
    assert rel_ver == __version__
    assert re.match(rf"^{re.escape(__version__)}-dev-[0-9a-z]+-\d{{14}}$", beta_ver)


def test_prepare_assets_creates_cross_platform_icons_and_rtf(tmp_path: Path) -> None:
    out = prepare_assets(tmp_path / "assets")
    assert (out / "app_icon.png").is_file()
    assert (out / "app_icon.ico").is_file()
    assert (out / "app_icon.icns").is_file()
    rtf = (out / "License.rtf").read_text(encoding="utf-8")
    assert rtf.startswith("{\\rtf1")
    assert "Weien Xu" in rtf


def test_linux_archive_packaging_creates_desktop_and_scripts(tmp_path: Path) -> None:
    assets = prepare_assets(tmp_path / "assets")
    bundle = tmp_path / "dist" / "OcrPipeline"
    bundle.mkdir(parents=True)
    (bundle / "OcrPipeline").write_text("#!/bin/sh\necho ok\n", encoding="utf-8")

    tar_path = package_linux_archive(bundle_dir=bundle, assets_dir=assets)
    assert tar_path.is_file()
    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()
    assert any(n.endswith("/install.sh") for n in names)
    assert any(n.endswith("/uninstall.sh") for n in names)
    assert any(n.endswith("/OcrPipeline.desktop.in") for n in names)
    assert any(n.endswith("/LICENSE") for n in names)

    desktop = linux_desktop_entry("/opt/ocr/OcrPipeline", "/opt/ocr/icon.png")
    assert "Exec=/opt/ocr/OcrPipeline" in desktop
    assert "Icon=/opt/ocr/icon.png" in desktop
