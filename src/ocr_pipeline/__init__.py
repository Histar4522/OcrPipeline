"""Local Ollama-powered AI OCR pipeline."""

from datetime import datetime
from functools import lru_cache
import os
from pathlib import Path
import subprocess
import sys

__version__ = "0.2.2"


@lru_cache(maxsize=1)
def get_display_version() -> str:
    """Return 'x.x.x' for release builds, 'x.x.x-dev-...' for beta installers, or 'x.x.x-devrun-...' for uv run."""
    try:
        from ._build_version import BUILD_VERSION  # type: ignore[import-not-found]

        if BUILD_VERSION:
            return str(BUILD_VERSION)
    except ImportError:
        pass

    if getattr(sys, "frozen", False):
        return __version__

    commit = "nogit"
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / ".git").exists():
        kwargs: dict = {"cwd": repo_root, "capture_output": True, "text": True, "timeout": 1.5}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        try:
            proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kwargs)
            if proc.returncode == 0 and proc.stdout.strip():
                commit = proc.stdout.strip()
        except Exception:
            pass

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{__version__}-devrun-{commit}-{stamp}"
