"""Cross-platform persistent application settings and provider/role resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

from .artifacts import ensure_obsidian_vault
from .config import DEFAULT_SYSTEM_PROMPT
from .providers import (
    PROVIDER_REGISTRY,
    ProviderEndpoint,
    default_provider_configs,
    resolve_api_key,
    resolve_base_url,
)

APP_NAME = "OcrPipeline"
NOTIFICATION_MODES = ("always", "not_in_focus", "minimized", "never")
MASKED_SECRET = "••••••••"


def normalize_notification_mode(value: object) -> str:
    """Coerce legacy boolean or string notification setting into a valid NOTIFICATION_MODES value."""
    if isinstance(value, bool):
        return "always" if value else "never"
    text = str(value or "").strip().lower()
    if text in NOTIFICATION_MODES:
        return text
    if text in ("false", "off", "0", "none"):
        return "never"
    return "always"


def default_config_path() -> Path:
    """Return the cross-platform settings JSON file path."""
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "settings.json"


def default_archive_dir() -> Path:
    """Return the cross-platform default archive output directory."""
    return Path(user_data_dir(APP_NAME, appauthor=False)) / "archive"


def default_jobs_dir() -> Path:
    """Return the cross-platform directory for uploaded job inputs/logs."""
    return Path(user_data_dir(APP_NAME, appauthor=False)) / "jobs"


def _merge_provider_configs(raw_providers: object, legacy_ollama_url: str = "") -> dict[str, dict[str, str]]:
    merged = default_provider_configs()
    if isinstance(raw_providers, dict):
        for pid, cfg in raw_providers.items():
            if pid in merged and isinstance(cfg, dict):
                merged[pid]["base_url"] = str(cfg.get("base_url") or merged[pid]["base_url"]).strip()
                merged[pid]["api_key"] = str(cfg.get("api_key") or "").strip()
    default_ollama_url = PROVIDER_REGISTRY["ollama"].default_base_url
    existing_ollama_url = (
        str(raw_providers["ollama"].get("base_url") or "").strip()
        if isinstance(raw_providers, dict) and isinstance(raw_providers.get("ollama"), dict)
        else ""
    )
    if legacy_ollama_url.strip() and (
        not existing_ollama_url
        or (existing_ollama_url == default_ollama_url and legacy_ollama_url.strip() != default_ollama_url)
    ):
        merged["ollama"]["base_url"] = legacy_ollama_url.strip()
    return merged


@dataclass(slots=True)
class AppSettings:
    """Persistent user preferences for the OcrPipeline desktop/web app."""

    setup_completed: bool = False
    ollama_url: str = "http://localhost:11434"
    default_model: str = "glm-ocr"
    providers: dict[str, dict[str, str]] = field(default_factory=default_provider_configs)
    vision_provider: str = "ollama"
    vision_model: str = "glm-ocr"
    llm_enabled: bool = False
    llm_provider: str = "ollama"
    llm_model: str = ""
    auto_tag_on_ocr: bool = True
    auto_rename_on_ocr: bool = False
    auto_organize_on_ocr: bool = False
    auto_organize_allow_new_folders: bool = False
    auto_summarize_on_ocr: bool = True
    archive_dir: str = ""
    dpi: int = 220
    max_image_dim: int = 1536
    num_ctx: int = 4096
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    minimize_to_tray: bool = True
    desktop_notifications: str = "always"
    obsidian_mode: bool = False
    theme: str = "auto"
    auto_check_updates: bool = True

    def __post_init__(self) -> None:
        if not self.archive_dir:
            self.archive_dir = str(default_archive_dir())
        self.providers = _merge_provider_configs(self.providers, self.ollama_url)
        if self.vision_provider not in PROVIDER_REGISTRY:
            self.vision_provider = "ollama"
        if self.llm_provider not in PROVIDER_REGISTRY:
            self.llm_provider = "ollama"
        if not self.vision_model.strip():
            self.vision_model = self.default_model.strip() or "glm-ocr"
        self.default_model = self.vision_model
        self.ollama_url = self.providers["ollama"]["base_url"] or "http://localhost:11434"
        self.desktop_notifications = normalize_notification_mode(self.desktop_notifications)
        self.auto_check_updates = bool(self.auto_check_updates)
        self.llm_enabled = bool(self.llm_enabled)
        self.auto_tag_on_ocr = bool(self.auto_tag_on_ocr)
        self.auto_rename_on_ocr = bool(self.auto_rename_on_ocr)
        self.auto_organize_on_ocr = bool(self.auto_organize_on_ocr)
        self.auto_organize_allow_new_folders = bool(self.auto_organize_allow_new_folders)
        self.auto_summarize_on_ocr = bool(self.auto_summarize_on_ocr)

    def to_dict(self) -> dict:
        return asdict(self)


def effective_base_url(settings: AppSettings, provider_id: str) -> str:
    """Return resolved base URL for provider_id using saved config, legacy field, or default."""
    pid = (provider_id or "ollama").strip().lower()
    cfg = settings.providers.get(pid) or {}
    configured = str(cfg.get("base_url") or "")
    if pid == "ollama" and not configured and settings.ollama_url:
        configured = settings.ollama_url
    return resolve_base_url(pid, configured)


def effective_api_key(settings: AppSettings, provider_id: str) -> str:
    """Return resolved API key from saved settings or matching environment variable."""
    pid = (provider_id or "ollama").strip().lower()
    cfg = settings.providers.get(pid) or {}
    return resolve_api_key(pid, str(cfg.get("api_key") or ""))


def resolve_vision_endpoint(
    settings: AppSettings,
    provider_id: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 180.0,
) -> ProviderEndpoint:
    """Build ProviderEndpoint for mandatory Vision OCR."""
    pid = (provider_id or settings.vision_provider or "ollama").strip().lower()
    if pid not in PROVIDER_REGISTRY:
        pid = "ollama"
    chosen_model = (model or settings.vision_model or settings.default_model or "glm-ocr").strip()
    return ProviderEndpoint(
        provider_id=pid,
        base_url=effective_base_url(settings, pid),
        api_key=effective_api_key(settings, pid),
        model=chosen_model,
        num_ctx=settings.num_ctx,
        timeout_seconds=timeout_seconds,
    )


def resolve_llm_endpoint(
    settings: AppSettings,
    timeout_seconds: float = 120.0,
) -> ProviderEndpoint | None:
    """Return ProviderEndpoint for the optional Assistant LLM, or None when disabled/unset."""
    if not settings.llm_enabled or not settings.llm_model.strip():
        return None
    pid = (settings.llm_provider or "ollama").strip().lower()
    if pid not in PROVIDER_REGISTRY:
        return None
    return ProviderEndpoint(
        provider_id=pid,
        base_url=effective_base_url(settings, pid),
        api_key=effective_api_key(settings, pid),
        model=settings.llm_model.strip(),
        num_ctx=settings.num_ctx,
        timeout_seconds=timeout_seconds,
    )


def apply_provider_updates(current: AppSettings, raw_providers: object) -> None:
    """Update provider base_url/api_key fields while preserving secrets when masked."""
    if not isinstance(raw_providers, dict):
        return
    for pid, cfg in raw_providers.items():
        if pid not in PROVIDER_REGISTRY or not isinstance(cfg, dict):
            continue
        target = current.providers.setdefault(
            pid, {"base_url": PROVIDER_REGISTRY[pid].default_base_url, "api_key": ""}
        )
        if "base_url" in cfg:
            target["base_url"] = str(cfg.get("base_url") or "").strip()
        if "api_key" in cfg:
            incoming_key = str(cfg.get("api_key") or "").strip()
            if incoming_key != MASKED_SECRET:
                target["api_key"] = incoming_key
    if "ollama" in current.providers and current.providers["ollama"]["base_url"]:
        current.ollama_url = current.providers["ollama"]["base_url"]


def mask_settings_for_api(settings: AppSettings) -> dict:
    """Return settings dictionary with raw API keys masked for browser serialization."""
    data = settings.to_dict()
    masked_providers: dict[str, dict[str, object]] = {}
    for pid, cfg in settings.providers.items():
        saved_key = str(cfg.get("api_key") or "").strip()
        env_key = resolve_api_key(pid, "")
        masked_providers[pid] = {
            "base_url": str(cfg.get("base_url") or ""),
            "api_key": MASKED_SECRET if saved_key else "",
            "has_saved_key": bool(saved_key),
            "has_key": bool(saved_key or env_key),
        }
    data["providers"] = masked_providers
    data["llm_available"] = resolve_llm_endpoint(settings) is not None
    return data


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Load settings from disk, migrating legacy Ollama fields cleanly."""
    path = config_path or default_config_path()
    if not path.exists():
        return AppSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return AppSettings()
    if not isinstance(raw, dict):
        return AppSettings()

    defaults = AppSettings().to_dict()
    merged = {key: raw.get(key, val) for key, val in defaults.items()}

    # Migrate legacy single-provider settings seamlessly
    if "vision_model" not in raw and raw.get("default_model"):
        merged["vision_model"] = str(raw["default_model"]).strip()
    if "ollama_url" in raw and isinstance(raw.get("ollama_url"), str):
        merged["providers"] = _merge_provider_configs(raw.get("providers"), raw["ollama_url"])

    return AppSettings(**merged)


def save_settings(settings: AppSettings, config_path: Path | None = None) -> Path:
    """Persist settings to disk (chmod 0600 on POSIX) and ensure archive directory exists."""
    path = config_path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if settings.archive_dir:
        archive_path = Path(settings.archive_dir).expanduser()
        archive_path.mkdir(parents=True, exist_ok=True)
        if settings.obsidian_mode:
            ensure_obsidian_vault(archive_path)
    # Keep legacy fields synchronized
    if settings.providers.get("ollama", {}).get("base_url"):
        settings.ollama_url = settings.providers["ollama"]["base_url"]
    if settings.vision_model:
        settings.default_model = settings.vision_model
    path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return path
