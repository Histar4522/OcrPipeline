"""Generate cross-platform build assets (.ico, .png, License.rtf) for installers."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ocr_pipeline.platform_utils import generate_app_icon_image  # noqa: E402


def _to_rtf(plain_text: str) -> str:
    escaped = (
        plain_text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\r\n", "\n")
        .replace("\n", "\\par\n")
    )
    return "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Segoe UI;}}\\fs18\n" + escaped + "\n}"


def prepare_assets(assets_dir: Path | None = None) -> Path:
    """Generate PNG, ICO, ICNS icons and License.rtf into build_assets."""
    target = assets_dir or (ROOT / "installer" / "build_assets")
    target.mkdir(parents=True, exist_ok=True)

    icon = generate_app_icon_image(256)
    icon.save(target / "app_icon.png", format="PNG")
    icon.save(
        target / "app_icon.ico",
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    icon.save(target / "app_icon.icns", format="ICNS")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    (target / "License.rtf").write_text(_to_rtf(license_text), encoding="utf-8")
    return target


def main() -> None:
    assets_dir = prepare_assets()
    print(f"Prepared installer assets in {assets_dir}")


if __name__ == "__main__":
    main()
