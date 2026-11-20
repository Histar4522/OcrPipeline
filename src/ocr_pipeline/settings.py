"""Cross-platform persistent application settings, multi-credential storage, and role resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import uuid

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
OLLAMA_LOCAL_CRED_ID = "cred_ollama_local"


class DuplicateCredentialError(ValueError):
    """Raised when attempting to save a credential whose name already exists."""


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


def default_credentials(ollama_url: str = "http://localhost:11434") -> list[dict[str, object]]:
    """Return initial credential list containing singleton Ollama Local."""
    return [
        {
            "id": OLLAMA_LOCAL_CRED_ID,
            "name": "Ollama Local",
            "provider": "ollama_local",
            "base_url": ollama_url.strip() or PROVIDER_REGISTRY["ollama_local"].default_base_url,
            "api_key": "",
            "enabled": True,
        }
    ]


def _merge_provider_configs(raw_providers: object, legacy_ollama_url: str = "") -> dict[str, dict[str, str]]:
    merged = default_provider_configs()
    if isinstance(raw_providers, dict):
        for pid, cfg in raw_providers.items():
            if pid in merged and isinstance(cfg, dict):
                merged[pid]["base_url"] = str(cfg.get("base_url") or merged[pid]["base_url"]).strip()
                merged[pid]["api_key"] = str(cfg.get("api_key") or "").strip()
    default_ollama_url = PROVIDER_REGISTRY["ollama_local"].default_base_url
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
        merged["ollama_local"]["base_url"] = legacy_ollama_url.strip()
    elif merged["ollama"]["base_url"]:
        merged["ollama_local"]["base_url"] = merged["ollama"]["base_url"]
    return merged


def _normalize_credentials(
    raw_credentials: object,
    providers_map: dict[str, dict[str, str]],
    legacy_ollama_url: str,
) -> list[dict[str, object]]:
    default_local_url = PROVIDER_REGISTRY["ollama_local"].default_base_url
    local_entry: dict[str, object] = {
        "id": OLLAMA_LOCAL_CRED_ID,
        "name": "Ollama Local",
        "provider": "ollama_local",
        "base_url": providers_map["ollama_local"]["base_url"] or default_local_url,
        "api_key": "",
        "enabled": True,
    }
    others: list[dict[str, object]] = []
    seen_ids: set[str] = {OLLAMA_LOCAL_CRED_ID}
    seen_names: set[str] = {"ollama local"}

    if isinstance(raw_credentials, list):
        for item in raw_credentials:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("provider") or "").strip().lower()
            if pid not in PROVIDER_REGISTRY:
                continue
            cid = str(item.get("id") or "").strip()
            if pid == "ollama_local" or cid == OLLAMA_LOCAL_CRED_ID:
                cand_url = str(item.get("base_url") or "").strip()
                if legacy_ollama_url.strip() and legacy_ollama_url.strip() != default_local_url and cand_url == default_local_url:
                    cand_url = legacy_ollama_url.strip()
                local_entry["base_url"] = cand_url or str(local_entry["base_url"]) or default_local_url
                local_entry["enabled"] = bool(item.get("enabled", True))
                continue

            spec = PROVIDER_REGISTRY[pid]
            if not cid or cid in seen_ids:
                cid = f"cred_{uuid.uuid4().hex[:8]}"
            base_name = str(item.get("name") or spec.short_label or spec.label).strip() or spec.label
            unique_name = base_name
            suffix = 2
            while unique_name.lower() in seen_names:
                unique_name = f"{base_name} ({suffix})"
                suffix += 1
            seen_ids.add(cid)
            seen_names.add(unique_name.lower())
            others.append(
                {
                    "id": cid,
                    "name": unique_name,
                    "provider": pid,
                    "base_url": str(item.get("base_url") or spec.default_base_url).strip(),
                    "api_key": str(item.get("api_key") or "").strip(),
                    "enabled": bool(item.get("enabled", True)),
                }
            )
    return [local_entry, *others]


@dataclass(slots=True)
class AppSettings:
    """Persistent user preferences and multi-provider credentials for OcrPipeline."""

    setup_completed: bool = False
    ollama_url: str = "http://localhost:11434"
    default_model: str = "glm-ocr"
    providers: dict[str, dict[str, str]] = field(default_factory=default_provider_configs)
    credentials: list[dict[str, object]] = field(default_factory=default_credentials)
    vision_credential_id: str = OLLAMA_LOCAL_CRED_ID
    vision_provider: str = "ollama_local"
    vision_model: str = "glm-ocr"
    vision_custom_model: bool = False
    llm_enabled: bool = False
    llm_credential_id: str = OLLAMA_LOCAL_CRED_ID
    llm_provider: str = "ollama_local"
    llm_model: str = ""
    llm_custom_model: bool = False
    llm_auto_ctx: bool = True
    auto_tag_on_ocr: bool = True
    auto_rename_on_ocr: bool = False
    auto_organize_on_ocr: bool = False
    auto_organize_allow_new_folders: bool = False
    auto_summarize_on_ocr: bool = True
    auto_cleanup_on_ocr: bool = False
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
        self.credentials = _normalize_credentials(self.credentials, self.providers, self.ollama_url)

        local_cred = find_credential(self, OLLAMA_LOCAL_CRED_ID)
        if local_cred is not None:
            self.ollama_url = str(local_cred.get("base_url") or "http://localhost:11434")
            self.providers["ollama_local"]["base_url"] = self.ollama_url
            self.providers["ollama"]["base_url"] = self.ollama_url

        if self.vision_provider not in PROVIDER_REGISTRY:
            self.vision_provider = "ollama_local"
        if self.llm_provider not in PROVIDER_REGISTRY:
            self.llm_provider = "ollama_local"

        v_cred = find_credential(self, self.vision_credential_id)
        if v_cred is None:
            v_cred = find_credential_for_provider(self, self.vision_provider) or local_cred
        if v_cred is not None:
            self.vision_credential_id = str(v_cred["id"])
            if self.vision_provider not in PROVIDER_REGISTRY or self.vision_credential_id == OLLAMA_LOCAL_CRED_ID:
                self.vision_provider = str(v_cred["provider"])

        l_cred = find_credential(self, self.llm_credential_id)
        if l_cred is None:
            l_cred = find_credential_for_provider(self, self.llm_provider) or local_cred
        if l_cred is not None:
            self.llm_credential_id = str(l_cred["id"])
            if self.llm_provider not in PROVIDER_REGISTRY or self.llm_credential_id == OLLAMA_LOCAL_CRED_ID:
                self.llm_provider = str(l_cred["provider"])

        if not self.vision_model.strip():
            self.vision_model = self.default_model.strip() or "glm-ocr"
        self.default_model = self.vision_model
        self.desktop_notifications = normalize_notification_mode(self.desktop_notifications)
        self.auto_check_updates = bool(self.auto_check_updates)
        self.llm_enabled = bool(self.llm_enabled)
        self.llm_auto_ctx = bool(self.llm_auto_ctx)
        self.vision_custom_model = bool(self.vision_custom_model)
        self.llm_custom_model = bool(self.llm_custom_model)
        self.auto_tag_on_ocr = bool(self.auto_tag_on_ocr)
        self.auto_rename_on_ocr = bool(self.auto_rename_on_ocr)
        self.auto_organize_on_ocr = bool(self.auto_organize_on_ocr)
        self.auto_organize_allow_new_folders = bool(self.auto_organize_allow_new_folders)
        self.auto_summarize_on_ocr = bool(self.auto_summarize_on_ocr)
        self.auto_cleanup_on_ocr = bool(self.auto_cleanup_on_ocr)

    def to_dict(self) -> dict:
        return asdict(self)


def find_credential(settings: AppSettings, cred_id: str | None) -> dict[str, object] | None:
    """Return credential dict matching `cred_id` if present."""
    cid = (cred_id or "").strip()
    if not cid:
        return None
    for cred in settings.credentials:
        if str(cred.get("id") or "") == cid:
            return cred
    return None


def find_credential_for_provider(settings: AppSettings, provider_id: str | None) -> dict[str, object] | None:
    """Find best matching credential for `provider_id` (mapping legacy 'ollama' -> 'ollama_local' when needed)."""
    pid = (provider_id or "").strip().lower()
    if not pid:
        return None
    for cred in settings.credentials:
        if str(cred.get("provider") or "").lower() == pid and cred.get("enabled", True):
            return cred
    for cred in settings.credentials:
        if str(cred.get("provider") or "").lower() == pid:
            return cred
    if pid == "ollama":
        return find_credential(settings, OLLAMA_LOCAL_CRED_ID)
    return None


def upsert_credential(settings: AppSettings, payload: dict, cred_id: str | None = None) -> dict[str, object]:
    """Create or update a named credential while enforcing unique names and singleton Ollama Local."""
    target_id = (cred_id or payload.get("id") or "").strip()
    pid = str(payload.get("provider") or "").strip().lower()
    if not pid and target_id == OLLAMA_LOCAL_CRED_ID:
        pid = "ollama_local"
    if pid not in PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported provider '{pid}'.")

    spec = PROVIDER_REGISTRY[pid]
    if target_id == OLLAMA_LOCAL_CRED_ID or (pid == "ollama_local" and not target_id):
        existing_local = find_credential(settings, OLLAMA_LOCAL_CRED_ID)
        if not target_id and existing_local is not None:
            raise ValueError("Ollama Local already exists and can only be added once.")
        assert existing_local is not None
        if "base_url" in payload:
            existing_local["base_url"] = str(payload.get("base_url") or spec.default_base_url).strip()
        if "enabled" in payload:
            existing_local["enabled"] = bool(payload["enabled"])
        settings.ollama_url = str(existing_local["base_url"] or spec.default_base_url)
        settings.providers["ollama_local"]["base_url"] = settings.ollama_url
        settings.providers["ollama"]["base_url"] = settings.ollama_url
        return existing_local

    if pid == "ollama_local":
        raise ValueError("Cannot change provider type to Ollama Local.")

    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Credential name is required.")

    for other in settings.credentials:
        if str(other.get("id")) != target_id and str(other.get("name") or "").strip().lower() == name.lower():
            raise DuplicateCredentialError(f"A credential named '{name}' already exists. Choose a unique name.")

    existing = find_credential(settings, target_id) if target_id else None
    if target_id and existing is None:
        raise LookupError(f"Credential '{target_id}' not found.")

    base_url = str(payload["base_url"]).strip() if "base_url" in payload else (
        str(existing.get("base_url") or "") if existing else spec.default_base_url
    )
    if "api_key" in payload:
        raw_key = str(payload.get("api_key") or "").strip()
        api_key = str(existing.get("api_key") or "") if (raw_key == MASKED_SECRET and existing) else raw_key
    else:
        api_key = str(existing.get("api_key") or "") if existing else ""
    enabled = bool(payload.get("enabled", existing.get("enabled", True) if existing else True))

    if existing is not None:
        existing["name"] = name
        existing["provider"] = pid
        existing["base_url"] = base_url
        existing["api_key"] = api_key
        existing["enabled"] = enabled
        updated = existing
    else:
        updated = {
            "id": f"cred_{uuid.uuid4().hex[:8]}",
            "name": name,
            "provider": pid,
            "base_url": base_url,
            "api_key": api_key,
            "enabled": enabled,
        }
        settings.credentials.append(updated)

    settings.providers[pid] = {"base_url": base_url, "api_key": api_key}
    return updated


def delete_credential(settings: AppSettings, cred_id: str) -> None:
    """Remove a non-singleton credential and cleanly rebind affected roles."""
    cid = (cred_id or "").strip()
    if cid == OLLAMA_LOCAL_CRED_ID:
        raise ValueError("Ollama Local cannot be deleted; disable it with the toggle instead.")
    existing = find_credential(settings, cid)
    if existing is None:
        raise LookupError(f"Credential '{cid}' not found.")

    settings.credentials = [c for c in settings.credentials if str(c.get("id")) != cid]
    if settings.vision_credential_id == cid:
        settings.vision_credential_id = OLLAMA_LOCAL_CRED_ID
        settings.vision_provider = "ollama_local"
        settings.vision_custom_model = False
    if settings.llm_credential_id == cid:
        settings.llm_enabled = False
        settings.llm_credential_id = OLLAMA_LOCAL_CRED_ID
        settings.llm_provider = "ollama_local"
        settings.llm_custom_model = False


def effective_base_url(settings: AppSettings, provider_id: str, credential_id: str | None = None) -> str:
    """Return resolved base URL for credential_id or provider_id."""
    if credential_id:
        cred = find_credential(settings, credential_id)
        if cred is not None:
            return resolve_base_url(str(cred["provider"]), str(cred.get("base_url") or ""))
    pid = (provider_id or "ollama_local").strip().lower()
    if pid in PROVIDER_REGISTRY:
        cfg = settings.providers.get(pid) or {}
        configured = str(cfg.get("base_url") or "")
        if pid in ("ollama", "ollama_local") and not configured and settings.ollama_url:
            configured = settings.ollama_url
        return resolve_base_url(pid, configured)
    return resolve_base_url("ollama_local", settings.ollama_url)


def effective_api_key(settings: AppSettings, provider_id: str, credential_id: str | None = None) -> str:
    """Return resolved API key for credential_id or provider_id (with env var fallback)."""
    if credential_id:
        cred = find_credential(settings, credential_id)
        if cred is not None:
            return resolve_api_key(str(cred["provider"]), str(cred.get("api_key") or ""))
    pid = (provider_id or "ollama_local").strip().lower()
    cfg = settings.providers.get(pid) or {}
    return resolve_api_key(pid, str(cfg.get("api_key") or ""))


def _provider_family_matches(selected_pid: str, cred_pid: str) -> bool:
    if selected_pid == cred_pid:
        return True
    return selected_pid == "ollama" and cred_pid == "ollama_local"


def resolve_vision_endpoint(
    settings: AppSettings,
    provider_id: str | None = None,
    model: str | None = None,
    *,
    credential_id: str | None = None,
    timeout_seconds: float = 180.0,
) -> ProviderEndpoint:
    """Build ProviderEndpoint for mandatory Vision OCR."""
    cred: dict[str, object] | None = find_credential(settings, credential_id) if credential_id else None
    if cred is None:
        role_cred = find_credential(settings, settings.vision_credential_id)
        req_pid = (provider_id or "").strip().lower()
        if req_pid:
            if role_cred is not None and _provider_family_matches(req_pid, str(role_cred["provider"])):
                cred = role_cred
            else:
                cred = find_credential_for_provider(settings, req_pid)
        else:
            if role_cred is not None and (
                not settings.vision_provider or _provider_family_matches(settings.vision_provider, str(role_cred["provider"]))
            ):
                cred = role_cred
            else:
                cred = find_credential_for_provider(settings, settings.vision_provider) or role_cred

    if cred is not None:
        pid = str(cred["provider"])
        base_url = resolve_base_url(pid, str(cred.get("base_url") or ""))
        api_key = resolve_api_key(pid, str(cred.get("api_key") or ""))
        cid = str(cred["id"])
        cname = str(cred["name"])
    else:
        pid = (provider_id or settings.vision_provider or "ollama_local").strip().lower()
        if pid not in PROVIDER_REGISTRY:
            pid = "ollama_local"
        base_url = effective_base_url(settings, pid)
        api_key = effective_api_key(settings, pid)
        cid = ""
        cname = PROVIDER_REGISTRY[pid].short_label or PROVIDER_REGISTRY[pid].label

    chosen_model = (model or settings.vision_model or settings.default_model or "glm-ocr").strip()
    return ProviderEndpoint(
        provider_id=pid,
        base_url=base_url,
        api_key=api_key,
        model=chosen_model,
        num_ctx=settings.num_ctx,
        timeout_seconds=timeout_seconds,
        auto_ctx=settings.llm_auto_ctx,
        credential_id=cid,
        credential_name=cname,
    )


def resolve_llm_endpoint(
    settings: AppSettings,
    timeout_seconds: float = 120.0,
) -> ProviderEndpoint | None:
    """Return ProviderEndpoint for the optional Assistant LLM, or None when disabled/unset."""
    if not settings.llm_enabled or not settings.llm_model.strip():
        return None
    role_cred = find_credential(settings, settings.llm_credential_id)
    if role_cred is not None and _provider_family_matches(settings.llm_provider, str(role_cred["provider"])):
        cred = role_cred
    else:
        cred = find_credential_for_provider(settings, settings.llm_provider) or role_cred

    if cred is not None:
        if not cred.get("enabled", True):
            return None
        pid = str(cred["provider"])
        base_url = resolve_base_url(pid, str(cred.get("base_url") or ""))
        api_key = resolve_api_key(pid, str(cred.get("api_key") or ""))
        cid = str(cred["id"])
        cname = str(cred["name"])
    else:
        pid = (settings.llm_provider or "ollama_local").strip().lower()
        if pid not in PROVIDER_REGISTRY:
            return None
        base_url = effective_base_url(settings, pid)
        api_key = effective_api_key(settings, pid)
        cid = ""
        cname = PROVIDER_REGISTRY[pid].short_label or PROVIDER_REGISTRY[pid].label

    return ProviderEndpoint(
        provider_id=pid,
        base_url=base_url,
        api_key=api_key,
        model=settings.llm_model.strip(),
        num_ctx=settings.num_ctx,
        timeout_seconds=timeout_seconds,
        auto_ctx=settings.llm_auto_ctx,
        credential_id=cid,
        credential_name=cname,
    )


def apply_provider_updates(current: AppSettings, raw_providers: object) -> None:
    """Legacy helper: update provider base_url/api_key and sync matching credential entry."""
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

        if pid in ("ollama", "ollama_local"):
            local = find_credential(current, OLLAMA_LOCAL_CRED_ID)
            if local is not None:
                local["base_url"] = target["base_url"] or local["base_url"]
                current.ollama_url = str(local["base_url"] or "http://localhost:11434")
                current.providers["ollama_local"]["base_url"] = current.ollama_url
                current.providers["ollama"]["base_url"] = current.ollama_url
        else:
            cred = find_credential_for_provider(current, pid)
            if cred is None:
                spec = PROVIDER_REGISTRY[pid]
                cred_name = spec.short_label or spec.label
                if any(str(c.get("name") or "").lower() == cred_name.lower() for c in current.credentials):
                    cred_name = f"{cred_name} ({pid})"
                cred = {
                    "id": f"cred_{pid}",
                    "name": cred_name,
                    "provider": pid,
                    "base_url": target["base_url"],
                    "api_key": target["api_key"],
                    "enabled": True,
                }
                current.credentials.append(cred)
            else:
                if "base_url" in cfg:
                    cred["base_url"] = target["base_url"]
                if "api_key" in cfg:
                    cred["api_key"] = target["api_key"]


def _mask_credential(cred: dict[str, object]) -> dict[str, object]:
    pid = str(cred.get("provider") or "custom")
    saved_key = str(cred.get("api_key") or "").strip()
    env_key = resolve_api_key(pid, "")
    return {
        "id": str(cred.get("id") or ""),
        "name": str(cred.get("name") or ""),
        "provider": pid,
        "base_url": str(cred.get("base_url") or ""),
        "api_key": MASKED_SECRET if saved_key else "",
        "has_saved_key": bool(saved_key),
        "has_key": bool(saved_key or env_key),
        "enabled": bool(cred.get("enabled", True)),
    }


def mask_settings_for_api(settings: AppSettings) -> dict:
    """Return settings dictionary with all credential API keys masked for browser serialization."""
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
    data["credentials"] = [_mask_credential(c) for c in settings.credentials]
    data["llm_available"] = resolve_llm_endpoint(settings) is not None
    return data


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Load settings from disk, migrating legacy Ollama and v0.3.0 provider maps into credentials."""
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

    if "vision_model" not in raw and raw.get("default_model"):
        merged["vision_model"] = str(raw["default_model"]).strip()

    raw_ollama_url = str(raw.get("ollama_url") or "http://localhost:11434").strip()
    providers_map = _merge_provider_configs(raw.get("providers"), raw_ollama_url)
    merged["providers"] = providers_map

    if "credentials" not in raw:
        # Seamless migration from legacy provider map -> named multi-credential list
        creds = default_credentials(providers_map["ollama_local"]["base_url"])
        vis_p = str(raw.get("vision_provider") or "ollama_local").strip().lower()
        llm_p = str(raw.get("llm_provider") or "ollama_local").strip().lower()
        if vis_p == "ollama":
            vis_p = "ollama_local"
        if llm_p == "ollama":
            llm_p = "ollama_local"

        for pid, cfg in providers_map.items():
            if pid in ("ollama", "ollama_local"):
                continue
            spec = PROVIDER_REGISTRY[pid]
            has_custom = (
                bool(cfg.get("api_key"))
                or (cfg.get("base_url") and cfg.get("base_url") != spec.default_base_url)
                or pid in (vis_p, llm_p)
            )
            if has_custom:
                creds.append(
                    {
                        "id": f"cred_{pid}",
                        "name": spec.short_label or spec.label,
                        "provider": pid,
                        "base_url": str(cfg.get("base_url") or spec.default_base_url),
                        "api_key": str(cfg.get("api_key") or ""),
                        "enabled": True,
                    }
                )
        merged["credentials"] = creds
        merged["vision_provider"] = vis_p if vis_p in PROVIDER_REGISTRY else "ollama_local"
        merged["llm_provider"] = llm_p if llm_p in PROVIDER_REGISTRY else "ollama_local"
        merged["vision_credential_id"] = (
            OLLAMA_LOCAL_CRED_ID if merged["vision_provider"] == "ollama_local" else f"cred_{merged['vision_provider']}"
        )
        merged["llm_credential_id"] = (
            OLLAMA_LOCAL_CRED_ID if merged["llm_provider"] == "ollama_local" else f"cred_{merged['llm_provider']}"
        )

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

    local = find_credential(settings, OLLAMA_LOCAL_CRED_ID)
    if local is not None:
        settings.ollama_url = str(local.get("base_url") or "http://localhost:11434")
        settings.providers["ollama_local"]["base_url"] = settings.ollama_url
        settings.providers["ollama"]["base_url"] = settings.ollama_url
    if settings.vision_model:
        settings.default_model = settings.vision_model

    path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return path
