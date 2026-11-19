"""GitHub release update checking, release notes preview, and cross-platform installer download/apply."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
import platform
import re
import subprocess
import tarfile
from urllib.parse import urlparse

import httpx
from platformdirs import user_data_dir

from . import __version__, get_display_version
from .platform_utils import open_in_file_manager

GITHUB_REPO = "Histar4522/OcrPipeline"
_ALLOWED_ASSET_HOSTS = (
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "codeload.github.com",
)
_VERSION_RE = re.compile(r"(\d+(?:\.\d+)*)")


class UpdateError(RuntimeError):
    """Raised when checking, downloading, or applying an update fails."""


def parse_version(tag: str) -> tuple[int, ...]:
    """Extract numeric version tuple (e.g. 'v0.2.2' -> (0, 2, 2)) for comparison."""
    match = _VERSION_RE.search(str(tag or "").strip())
    if not match:
        return (0,)
    return tuple(int(part) for part in match.group(1).split("."))


def select_release_asset(
    assets: list[dict],
    system: str | None = None,
    machine: str | None = None,
) -> dict | None:
    """Pick the best release asset dict for the current OS and architecture."""
    if not assets:
        return None
    sys_name = (system or platform.system()).lower()
    mach = (machine or platform.machine()).lower()
    is_arm = "arm" in mach or "aarch" in mach

    if sys_name == "windows":
        ext, arch_hint = ".msi", ("arm64" if is_arm else "x64")
    elif sys_name == "darwin":
        ext, arch_hint = ".dmg", ("arm64" if is_arm else "x86_64")
    else:
        ext, arch_hint = ".tar.gz", ("aarch64" if is_arm else "x86_64")

    matching_ext = [a for a in assets if str(a.get("name", "")).lower().endswith(ext)]
    if not matching_ext:
        return None
    preferred = next((a for a in matching_ext if arch_hint in str(a.get("name", "")).lower()), None)
    picked = preferred or matching_ext[0]
    return {
        "name": str(picked.get("name", "")),
        "size": int(picked.get("size") or 0),
        "download_url": str(picked.get("browser_download_url") or picked.get("download_url") or ""),
    }


def _validate_download_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https" or not any(
        host == h or host.endswith(f".{h}") for h in _ALLOWED_ASSET_HOSTS
    ):
        raise UpdateError(f"Untrusted release download URL: {url}")


def fetch_github_releases(
    repo: str = GITHUB_REPO,
    current_version: str = __version__,
    timeout: float = 10.0,
) -> dict:
    """Query GitHub Releases API for current, newer, and historical releases."""
    url = f"https://api.github.com/repos/{repo}/releases"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": f"OcrPipeline/{current_version}",
    }
    try:
        resp = httpx.get(url, params={"per_page": 20}, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        raw_releases = resp.json()
    except httpx.HTTPStatusError as exc:
        detail = f"GitHub API returned HTTP {exc.response.status_code}"
        if exc.response.status_code in (403, 429):
            detail += " (rate limited — try again later)"
        raise UpdateError(detail) from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise UpdateError(f"Failed to check GitHub releases: {exc}") from exc

    if not isinstance(raw_releases, list):
        raise UpdateError("Unexpected GitHub Releases response format.")

    cur_tuple = parse_version(current_version)
    releases: list[dict] = []
    for item in raw_releases:
        if not isinstance(item, dict) or item.get("draft"):
            continue
        tag = str(item.get("tag_name") or "").strip()
        if not tag:
            continue
        ver_tuple = parse_version(tag)
        asset = select_release_asset(list(item.get("assets") or []))
        releases.append(
            {
                "tag_name": tag,
                "name": str(item.get("name") or tag),
                "body": str(item.get("body") or "").strip(),
                "published_at": str(item.get("published_at") or "")[:10],
                "html_url": str(item.get("html_url") or f"https://github.com/{repo}/releases/tag/{tag}"),
                "prerelease": bool(item.get("prerelease")),
                "is_current": ver_tuple == cur_tuple,
                "is_newer": ver_tuple > cur_tuple,
                "asset": asset,
            }
        )

    latest = next((r for r in releases if not r["prerelease"]), releases[0] if releases else None)
    return {
        "repo": repo,
        "current_version": current_version,
        "display_version": get_display_version(),
        "latest": latest,
        "update_available": bool(latest and latest["is_newer"]),
        "releases": releases,
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _apply_downloaded_installer(dest: Path, system: str | None = None) -> dict:
    sys_name = (system or platform.system()).lower()
    suffix = dest.name.lower()

    if sys_name == "windows" and suffix.endswith(".msi"):
        subprocess.Popen(["msiexec", "/i", str(dest)])
        return {
            "installed": True,
            "launched_installer": True,
            "path": str(dest),
            "message": f"Launched Windows installer ({dest.name}). Follow the wizard to finish upgrading.",
        }

    if sys_name == "darwin" and suffix.endswith(".dmg"):
        subprocess.Popen(["open", str(dest)])
        return {
            "installed": True,
            "launched_installer": True,
            "path": str(dest),
            "message": f"Opened disk image ({dest.name}). Drag OcrPipeline.app into Applications.",
        }

    if sys_name == "linux" and suffix.endswith(".tar.gz"):
        stage_dir = dest.parent / "extracted"
        stage_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "r:gz") as tf:
            tf.extractall(stage_dir, filter="data")
        install_scripts = list(stage_dir.glob("*/install.sh"))
        if install_scripts:
            proc = subprocess.run(["sh", str(install_scripts[0])], capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                raise UpdateError(proc.stderr.strip() or f"install.sh exited with {proc.returncode}")
            return {
                "installed": True,
                "launched_installer": False,
                "path": str(dest),
                "message": "Updated Linux installation via install.sh. Restart OcrPipeline to apply.",
            }

    open_in_file_manager(dest)
    return {
        "installed": False,
        "launched_installer": True,
        "path": str(dest),
        "message": f"Downloaded {dest.name} to {dest.parent}.",
    }


def download_and_install_update(
    asset_url: str,
    asset_name: str,
    download_dir: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    timeout: float = 120.0,
) -> dict:
    """Download a GitHub release asset to local staging and launch/apply it for this OS."""
    _validate_download_url(asset_url)
    clean_name = Path(asset_name).name
    if not clean_name:
        raise UpdateError("Release asset filename is required.")

    target_dir = (download_dir or Path(user_data_dir("OcrPipeline", appauthor=False)) / "updates").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = (target_dir / clean_name).resolve()
    if not dest.is_relative_to(target_dir):
        raise UpdateError("Invalid asset filename.")

    headers = {"User-Agent": f"OcrPipeline/{__version__}"}
    try:
        with httpx.stream("GET", asset_url, headers=headers, timeout=timeout, follow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0)
            downloaded = 0
            with dest.open("wb") as out:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    out.write(chunk)
                    downloaded += len(chunk)
                    if on_progress is not None:
                        on_progress(downloaded, total)
    except httpx.HTTPError as exc:
        raise UpdateError(f"Failed to download update asset: {exc}") from exc

    return _apply_downloaded_installer(dest)
