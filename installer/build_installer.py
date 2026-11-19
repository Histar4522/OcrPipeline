"""Cross-platform installer builder for OcrPipeline (Windows MSI, macOS DMG, Linux tarball)."""

import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "installer"))

from ocr_pipeline import __version__  # noqa: E402
from prepare_assets import prepare_assets  # noqa: E402


def _run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _find_wix() -> str:
    found = shutil.which("wix")
    if found:
        return found
    fallback = Path(r"C:\Program Files\WiX Toolset v6.0\bin\wix.exe")
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError("WiX Toolset v6 (`wix`) was not found on PATH.")


def build_pyinstaller_bundle() -> Path:
    """Build the standalone PyInstaller bundle into dist/."""
    spec = ROOT / "installer" / "OcrPipeline.spec"
    _run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec)])
    return ROOT / "dist" / ("OcrPipeline.app" if sys.platform == "darwin" else "OcrPipeline")


def package_windows_msi() -> Path:
    """Compile the Windows MSI installation wizard via WiX Toolset v6."""
    wix = _find_wix()
    ver_out = subprocess.check_output([wix, "--version"], text=True).strip()
    wix_ver = ver_out.split("+", 1)[0].strip()

    ext_out = subprocess.run([wix, "extension", "list", "-g"], capture_output=True, text=True)
    if "WixToolset.UI.wixext" not in ext_out.stdout:
        _run([wix, "extension", "add", "-g", f"WixToolset.UI.wixext/{wix_ver}"])

    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    msi_out = ROOT / "dist" / f"OcrPipeline-{__version__}-{arch}.msi"
    _run(
        [
            wix,
            "build",
            "-arch",
            arch,
            "-ext",
            "WixToolset.UI.wixext",
            "-d",
            f"AppVersion={__version__}",
            "-o",
            str(msi_out),
            str(ROOT / "installer" / "windows" / "OcrPipeline.wxs"),
        ]
    )
    return msi_out


def package_macos_dmg() -> Path:
    """Package dist/OcrPipeline.app into a drag-to-Applications DMG on macOS."""
    app_bundle = ROOT / "dist" / "OcrPipeline.app"
    if not app_bundle.exists():
        raise FileNotFoundError(f"Missing macOS bundle: {app_bundle}")

    arch = platform.machine().lower()
    dmg_out = ROOT / "dist" / f"OcrPipeline-{__version__}-macOS-{arch}.dmg"
    if dmg_out.exists():
        dmg_out.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "dmg_stage"
        stage.mkdir()
        shutil.copytree(app_bundle, stage / "OcrPipeline.app", symlinks=True)
        (stage / "Applications").symlink_to("/Applications")
        _run(
            [
                "hdiutil",
                "create",
                "-volname",
                "OcrPipeline",
                "-srcfolder",
                str(stage),
                "-ov",
                "-format",
                "UDZO",
                str(dmg_out),
            ]
        )
    return dmg_out


def linux_desktop_entry(exec_path: str, icon_path: str) -> str:
    """Return a freedesktop .desktop file entry for Linux."""
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Version={__version__}\n"
        "Name=OcrPipeline\n"
        "Comment=Local AI OCR Pipeline powered by Ollama\n"
        f"Exec={exec_path}\n"
        f"Icon={icon_path}\n"
        "Terminal=false\n"
        "Categories=Office;Utility;\n"
    )


def package_linux_archive(bundle_dir: Path | None = None, assets_dir: Path | None = None) -> Path:
    """Package the Linux PyInstaller bundle with .desktop entry and install/uninstall scripts."""
    src_bundle = bundle_dir or (ROOT / "dist" / "OcrPipeline")
    src_assets = assets_dir or (ROOT / "installer" / "build_assets")
    if not src_bundle.is_dir():
        raise FileNotFoundError(f"Missing Linux bundle directory: {src_bundle}")

    arch = platform.machine().lower()
    tar_out = src_bundle.parent / f"OcrPipeline-{__version__}-linux-{arch}.tar.gz"

    install_sh = (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        'HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"\n'
        'PREFIX="${XDG_DATA_HOME:-$HOME/.local/share}/OcrPipeline"\n'
        'APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"\n'
        'BIN_DIR="$HOME/.local/bin"\n'
        'mkdir -p "$PREFIX" "$APPS_DIR" "$BIN_DIR"\n'
        'rm -rf "$PREFIX/app"\n'
        'cp -a "$HERE/OcrPipeline" "$PREFIX/app"\n'
        'cp "$HERE/app_icon.png" "$PREFIX/app_icon.png"\n'
        'ln -sf "$PREFIX/app/OcrPipeline" "$BIN_DIR/ocr-pipeline-app"\n'
        'sed "s|@EXEC@|$PREFIX/app/OcrPipeline|g; s|@ICON@|$PREFIX/app_icon.png|g" '
        '"$HERE/OcrPipeline.desktop.in" > "$APPS_DIR/OcrPipeline.desktop"\n'
        'echo "Installed OcrPipeline to $PREFIX/app"\n'
    )
    uninstall_sh = (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        'PREFIX="${XDG_DATA_HOME:-$HOME/.local/share}/OcrPipeline"\n'
        'APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"\n'
        'rm -rf "$PREFIX" "$APPS_DIR/OcrPipeline.desktop" "$HOME/.local/bin/ocr-pipeline-app"\n'
        'echo "Uninstalled OcrPipeline."\n'
    )

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / f"OcrPipeline-{__version__}-linux-{arch}"
        stage.mkdir()
        shutil.copytree(src_bundle, stage / "OcrPipeline", symlinks=True)
        shutil.copy2(src_assets / "app_icon.png", stage / "app_icon.png")
        shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")
        (stage / "OcrPipeline.desktop.in").write_text(linux_desktop_entry("@EXEC@", "@ICON@"), encoding="utf-8")

        for name, script in (("install.sh", install_sh), ("uninstall.sh", uninstall_sh)):
            path = stage / name
            path.write_text(script, encoding="utf-8", newline="\n")
            path.chmod(0o755)

        with tarfile.open(tar_out, "w:gz") as tf:
            tf.add(stage, arcname=stage.name)

    return tar_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build OcrPipeline desktop bundle and OS installer.")
    parser.add_argument("--bundle-only", action="store_true", help="Only build PyInstaller bundle.")
    parser.add_argument("--skip-pyinstaller", action="store_true", help="Package an existing dist/ bundle.")
    args = parser.parse_args(argv)

    assets = prepare_assets()
    print(f"Prepared installer assets in {assets}")

    if not args.skip_pyinstaller:
        build_pyinstaller_bundle()

    if args.bundle_only:
        print("Bundle build finished.")
        return 0

    if sys.platform == "win32":
        artifact = package_windows_msi()
    elif sys.platform == "darwin":
        artifact = package_macos_dmg()
    else:
        artifact = package_linux_archive(assets_dir=assets)

    print(f"Installer ready: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
