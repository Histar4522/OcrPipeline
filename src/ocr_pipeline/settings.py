"""Cross-platform persistent application settings."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

from .artifacts import ensure_obsidian_vault
from .config import DEFAULT_SYSTEM_PROMPT

APP_NAME = "OcrPipeline"


def default_config_path() -> Path:
    """Return the cross-platform settings JSON file path."""
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "settings.json"


def default_archive_dir() -> Path:
    """Return the cross-platform default archive output directory."""
    return Path(user_data_dir(APP_NAME, appauthor=False)) / "archive"


def default_jobs_dir() -> Path:
    """Return the cross-platform directory for uploaded job inputs/logs."""
    return Path(user_data_dir(APP_NAME, appauthor=False)) / "jobs"


@dataclass(slots=True)
class AppSettings:
    """Persistent user preferences for the OcrPipeline desktop/web app."""

    setup_completed: bool = False
    ollama_url: str = "http://localhost:11434"
    default_model: str = "glm-ocr"
    archive_dir: str = ""
    dpi: int = 220
    max_image_dim: int = 1536
    num_ctx: int = 4096
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    minimize_to_tray: bool = True
    desktop_notifications: bool = True
    obsidian_mode: bool = False

    def __post_init__(self) -> None:
        if not self.archive_dir:
            self.archive_dir = str(default_archive_dir())

    def to_dict(self) -> dict:
        return asdict(self)


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Load settings from disk, falling back to defaults for missing fields."""
    path = config_path or default_config_path()
    if not path.exists():
        return AppSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return AppSettings()

    defaults = AppSettings().to_dict()
    merged = {key: raw.get(key, val) for key, val in defaults.items()}
    return AppSettings(**merged)


def save_settings(settings: AppSettings, config_path: Path | None = None) -> Path:
    """Persist settings atomically to disk and ensure archive directory exists."""
    path = config_path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if settings.archive_dir:
        archive_path = Path(settings.archive_dir).expanduser()
        archive_path.mkdir(parents=True, exist_ok=True)
        if settings.obsidian_mode:
            ensure_obsidian_vault(archive_path)
    path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    return path
