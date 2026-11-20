"""Unified zero-dependency multi-provider client for Vision OCR and Text LLM tasks."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import os
import threading
import time

import httpx

from .config import OcrConfig
from .ollama_client import (
    JobCancelledError,
    ProviderError,
    check_ollama_ready,
    clean_ocr_output,
    find_ollama_executable,
    list_models_detailed,
    ocr_image_b64,
)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Static definition of a supported local or cloud AI model provider."""

    id: str
    label: str
    protocol: str  # "ollama" | "openai_compat" | "anthropic" | "google"
    default_base_url: str
    requires_api_key: bool
    is_local: bool
    supports_vision: bool
    supports_llm: bool
    env_var: str
    short_label: str = ""
    singleton: bool = False
    suggested_vision_models: tuple[str, ...] = ()
    suggested_llm_models: tuple[str, ...] = ()


@dataclass(slots=True)
class ProviderEndpoint:
    """Resolved connection & model parameters for a single inference role."""

    provider_id: str
    base_url: str
    api_key: str
    model: str
    num_ctx: int = 4096
    timeout_seconds: float = 180.0
    max_retries: int = 2
    auto_ctx: bool = True
    credential_id: str = ""
    credential_name: str = ""


STATUS_LABELS: dict[str, str] = {
    "ok": "",
    "connecting": "Connecting",
    "offline": "Offline",
    "out_of_quota": "Out of Quota",
    "invalid_key": "Invalid Key",
    "no_key": "No Key",
    "no_url": "No URL",
    "model_missing": "Not Pulled",
    "disabled": "Disabled",
    "error": "Error",
}


DEEPSEEK_OCR_TOOL_MODELS: frozenset[str] = frozenset(
    {"deepseek-ocr", "deepseek-ocr-rich", "deepseek-ocr-text", "deepseek-ai/deepseek-ocr"}
)
DEEPSEEK_TEXT_ONLY_MODELS: frozenset[str] = frozenset({"deepseek-v4-pro", "deepseek-reasoner"})


PROVIDER_SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="ollama_local",
        label="Ollama Local",
        short_label="Ollama Local",
        protocol="ollama",
        default_base_url="http://localhost:11434",
        requires_api_key=False,
        is_local=True,
        singleton=True,
        supports_vision=True,
        supports_llm=True,
        env_var="OLLAMA_HOST",
        suggested_vision_models=("glm-ocr", "deepseek-ocr", "qwen3-vl:4b", "qwen3-vl:8b"),
        suggested_llm_models=("qwen3:1.7b", "qwen2.5:3b", "llama3.2:3b", "gemma3:4b"),
    ),
    ProviderSpec(
        id="ollama",
        label="Ollama (Remote / Custom)",
        short_label="Ollama",
        protocol="ollama",
        default_base_url="http://localhost:11434",
        requires_api_key=False,
        is_local=True,
        singleton=False,
        supports_vision=True,
        supports_llm=True,
        env_var="",
        suggested_vision_models=("glm-ocr", "deepseek-ocr", "qwen3-vl:4b", "qwen3-vl:8b"),
        suggested_llm_models=("qwen3:1.7b", "qwen2.5:3b", "llama3.2:3b", "gemma3:4b"),
    ),
    ProviderSpec(
        id="llama_cpp",
        label="llama.cpp Server (Local)",
        short_label="llama.cpp",
        protocol="openai_compat",
        default_base_url="http://localhost:8080/v1",
        requires_api_key=False,
        is_local=True,
        supports_vision=True,
        supports_llm=True,
        env_var="LLAMA_CPP_API_KEY",
        suggested_vision_models=("Qwen2.5-VL-7B-Instruct", "Qwen3-VL-4B-Instruct"),
        suggested_llm_models=("Qwen2.5-3B-Instruct", "Qwen3-1.7B", "Llama-3.2-3B-Instruct"),
    ),
    ProviderSpec(
        id="openai",
        label="OpenAI",
        short_label="OpenAI",
        protocol="openai_compat",
        default_base_url="https://api.openai.com/v1",
        requires_api_key=True,
        is_local=False,
        supports_vision=True,
        supports_llm=True,
        env_var="OPENAI_API_KEY",
        suggested_vision_models=("gpt-5-mini", "gpt-4.1-mini", "gpt-5", "gpt-4.1", "gpt-4o-mini"),
        suggested_llm_models=("gpt-5-nano", "gpt-5-mini", "gpt-4.1-nano", "gpt-4.1-mini"),
    ),
    ProviderSpec(
        id="anthropic",
        label="Anthropic",
        short_label="Anthropic",
        protocol="anthropic",
        default_base_url="https://api.anthropic.com/v1",
        requires_api_key=True,
        is_local=False,
        supports_vision=True,
        supports_llm=True,
        env_var="ANTHROPIC_API_KEY",
        suggested_vision_models=("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5", "claude-sonnet-4-5"),
        suggested_llm_models=("claude-haiku-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"),
    ),
    ProviderSpec(
        id="google",
        label="Google Gemini",
        short_label="Google",
        protocol="google",
        default_base_url="https://generativelanguage.googleapis.com/v1beta",
        requires_api_key=True,
        is_local=False,
        supports_vision=True,
        supports_llm=True,
        env_var="GEMINI_API_KEY",
        suggested_vision_models=("gemini-3.8-flash", "gemini-4-argon", "gemini-2.5-flash"),
        suggested_llm_models=("gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"),
    ),
    ProviderSpec(
        id="deepseek",
        label="DeepSeek",
        short_label="DeepSeek",
        protocol="openai_compat",
        default_base_url="https://api.deepseek.com/v1",
        requires_api_key=True,
        is_local=False,
        supports_vision=True,
        supports_llm=True,
        env_var="DEEPSEEK_API_KEY",
        suggested_vision_models=("deepseek-ocr", "deepseek-ocr-text", "deepseek-flash"),
        suggested_llm_models=("deepseek-flash", "deepseek-v4-pro"),
    ),
    ProviderSpec(
        id="xai",
        label="Grok (xAI)",
        short_label="Grok",
        protocol="openai_compat",
        default_base_url="https://api.x.ai/v1",
        requires_api_key=True,
        is_local=False,
        supports_vision=True,
        supports_llm=True,
        env_var="XAI_API_KEY",
        suggested_vision_models=("grok-4.7", "grok-4.6", "grok-4.20-non-reasoning", "grok-4.20-reasoning"),
        suggested_llm_models=("grok-4.20-non-reasoning", "grok-4.6", "grok-4.7"),
    ),
    ProviderSpec(
        id="qwen",
        label="Qwen (DashScope)",
        short_label="Qwen",
        protocol="openai_compat",
        default_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        requires_api_key=True,
        is_local=False,
        supports_vision=True,
        supports_llm=True,
        env_var="DASHSCOPE_API_KEY",
        suggested_vision_models=("qwen-vl-ocr", "qwen3.5-ocr", "qwen3-vl-plus", "qwen-vl-plus"),
        suggested_llm_models=("qwen3.8-flash", "qwen-flash", "qwen-plus", "qwen-turbo"),
    ),
    ProviderSpec(
        id="custom",
        label="Custom OpenAI-Compatible",
        short_label="Custom",
        protocol="openai_compat",
        default_base_url="",
        requires_api_key=False,
        is_local=False,
        supports_vision=True,
        supports_llm=True,
        env_var="OCR_CUSTOM_API_KEY",
        suggested_vision_models=(),
        suggested_llm_models=(),
    ),
)

PROVIDER_REGISTRY: dict[str, ProviderSpec] = {spec.id: spec for spec in PROVIDER_SPECS}


def get_provider_spec(provider_id: str) -> ProviderSpec:
    """Return ProviderSpec for provider_id, raising ProviderError if unknown."""
    key = (provider_id or "ollama").strip().lower()
    if key not in PROVIDER_REGISTRY:
        raise ProviderError(f"Unsupported provider '{provider_id}'.")
    return PROVIDER_REGISTRY[key]


def default_provider_configs() -> dict[str, dict[str, str]]:
    """Return default {provider_id: {base_url, api_key}} dictionary."""
    return {
        spec.id: {"base_url": spec.default_base_url, "api_key": ""}
        for spec in PROVIDER_SPECS
    }


def list_providers_metadata() -> list[dict]:
    """Return serializable metadata for all registered providers."""
    return [asdict(spec) for spec in PROVIDER_SPECS]


def resolve_api_key(provider_id: str, configured_key: str = "") -> str:
    """Return configured API key or fall back to standard provider environment variable."""
    key = configured_key.strip()
    if key:
        return key
    spec = get_provider_spec(provider_id)
    return os.environ.get(spec.env_var, "").strip()


def resolve_base_url(provider_id: str, configured_url: str = "") -> str:
    """Return normalized base URL, honoring OLLAMA_HOST / spec defaults."""
    url = configured_url.strip().rstrip("/")
    if url:
        return url
    spec = get_provider_spec(provider_id)
    if spec.id == "ollama_local" or (spec.id == "ollama" and not configured_url):
        env_host = os.environ.get("OLLAMA_HOST", "").strip().rstrip("/")
        if env_host:
            return env_host if "://" in env_host else f"http://{env_host}"
    return spec.default_base_url.rstrip("/")


def classify_http_error(status_code: int, body_text: str = "") -> str:
    """Classify an HTTP error status/body into a concise navbar status key."""
    lower = (body_text or "").lower()
    if status_code in (402, 429) or any(
        tok in lower for tok in ("quota", "insufficient_quota", "rate limit", "rate_limit", "billing", "credit")
    ):
        return "out_of_quota"
    if status_code in (401, 403):
        return "invalid_key"
    return "error"


def classify_exception_status(exc: Exception) -> str:
    """Classify a runtime exception into a normalized status key."""
    if isinstance(exc, httpx.HTTPStatusError):
        return classify_http_error(exc.response.status_code, exc.response.text[:300])
    if isinstance(exc, httpx.HTTPError):
        return "offline"
    msg = str(exc).lower()
    if any(tok in msg for tok in ("http 429", "http 402", "quota", "insufficient_quota", "rate limit", "credit")):
        return "out_of_quota"
    if any(tok in msg for tok in ("http 401", "http 403", "api key is required", "invalid api key", "unauthorized")):
        return "invalid_key"
    if "not pulled" in msg:
        return "model_missing"
    if any(tok in msg for tok in ("cannot reach", "unreachable", "connection", "refused", "timed out")):
        return "offline"
    return "error"


def _format_http_error(spec: ProviderSpec, exc: httpx.HTTPStatusError) -> str:
    body = ""
    try:
        payload = exc.response.json()
        err = payload.get("error")
        if isinstance(err, dict):
            body = str(err.get("message") or err)
        elif err:
            body = str(err)
        elif "detail" in payload:
            body = str(payload["detail"])
    except Exception:
        body = exc.response.text[:240].strip()
    suffix = f": {body}" if body else ""
    return f"{spec.label} HTTP {exc.response.status_code}{suffix}"


def _extract_openai_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = [str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(parts).strip()
    return str(content).strip()


def _invoke_with_retries(
    endpoint: ProviderEndpoint,
    request_fn,
    client: httpx.Client | None,
    cancel_event: threading.Event | None,
):
    spec = get_provider_spec(endpoint.provider_id)
    owns_client = client is None
    http = client or httpx.Client(timeout=endpoint.timeout_seconds)
    last_exc: Exception | None = None
    try:
        for attempt in range(endpoint.max_retries + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelledError("Cancelled by user.")
            try:
                return request_fn(http)
            except httpx.HTTPStatusError as exc:
                if cancel_event is not None and cancel_event.is_set():
                    raise JobCancelledError("Cancelled by user.") from exc
                last_exc = ProviderError(_format_http_error(spec, exc))
                if exc.response.status_code in (400, 401, 403, 404):
                    raise last_exc from exc
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                if cancel_event is not None and cancel_event.is_set():
                    raise JobCancelledError("Cancelled by user.") from exc
                last_exc = exc
            if attempt < endpoint.max_retries:
                time.sleep(1.5 * (attempt + 1))
        raise ProviderError(
            f"{spec.label} request failed after {endpoint.max_retries + 1} attempts: {last_exc}"
        ) from last_exc
    finally:
        if owns_client:
            http.close()


def _openai_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _filter_discovered_models(provider_id: str, names: list[str]) -> list[str]:
    """Filter out non-chat/non-vision utility models returned by cloud /models endpoints."""
    pid = (provider_id or "").strip().lower()
    cleaned: set[str] = set()
    openai_skip = (
        "whisper",
        "tts-",
        "dall-e",
        "gpt-image",
        "sora",
        "embedding",
        "moderation",
        "babbage",
        "davinci",
        "realtime",
        "audio",
        "transcribe",
    )
    google_skip = ("embedding", "aqa", "imagen", "veo", "tts", "robotics")
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        low = name.lower()
        if pid == "openai" and any(tok in low for tok in openai_skip):
            continue
        if pid == "xai" and low.startswith("grok-imagine-"):
            continue
        if pid == "google" and any(tok in low for tok in google_skip):
            continue
        cleaned.add(name)
    return sorted(cleaned)


def _send_openai_compat(
    endpoint: ProviderEndpoint,
    messages: list[dict],
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    client: httpx.Client | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    base = resolve_base_url(endpoint.provider_id, endpoint.base_url)
    if not base:
        raise ProviderError("Custom provider base URL is required.")
    url = f"{base}/chat/completions"
    api_key = resolve_api_key(endpoint.provider_id, endpoint.api_key)
    model_lower = endpoint.model.strip().lower()
    payload: dict = {
        "model": endpoint.model,
        "messages": messages,
    }
    if not (endpoint.provider_id == "openai" and model_lower.startswith(("gpt-5", "o1", "o3", "o4"))):
        payload["temperature"] = temperature
    if max_tokens is not None:
        if endpoint.provider_id == "openai":
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if endpoint.provider_id == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    if endpoint.provider_id == "qwen" and model_lower.startswith("qwen3") and "ocr" not in model_lower:
        payload["enable_thinking"] = False

    def call(http: httpx.Client) -> str:
        resp = http.post(url, headers=_openai_headers(api_key), json=payload, timeout=endpoint.timeout_seconds)
        resp.raise_for_status()
        return _extract_openai_text(resp.json())

    return _invoke_with_retries(endpoint, call, client, cancel_event)


def _send_deepseek_ocr(
    image_b64: str,
    endpoint: ProviderEndpoint,
    *,
    client: httpx.Client | None = None,
    cancel_event: threading.Event | None = None,
    metrics_out: dict[str, object] | None = None,
) -> str:
    """Call DeepSeek's dedicated multipart /v1/tools/ocr endpoint for a PNG page."""
    base = resolve_base_url("deepseek", endpoint.base_url)
    v1_base = base if base.endswith("/v1") else f"{base}/v1"
    url = f"{v1_base}/tools/ocr"
    api_key = resolve_api_key("deepseek", endpoint.api_key)
    if not api_key:
        raise ProviderError("DeepSeek API key is required.")
    headers = {"Authorization": f"Bearer {api_key}"}
    purpose = (
        "text_recognition"
        if endpoint.model.strip().lower() == "deepseek-ocr-text"
        else "rich_layout_recognition"
    )
    png_bytes = base64.b64decode(image_b64)

    def call(http: httpx.Client) -> str:
        resp = http.post(
            url,
            headers=headers,
            data={"purpose": purpose},
            files={"file": ("page.png", png_bytes, "image/png")},
            timeout=endpoint.timeout_seconds,
        )
        resp.raise_for_status()
        payload = resp.json()
        pages = sorted(payload.get("pages") or [], key=lambda p: int(p.get("index", 0)))
        if metrics_out is not None and isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
            if "input_tokens" in usage:
                metrics_out["prompt_eval_count"] = int(usage["input_tokens"] or 0)
            if "output_tokens" in usage:
                metrics_out["eval_count"] = int(usage["output_tokens"] or 0)
        texts = [str(p.get("text", "")).strip() for p in pages if str(p.get("text", "")).strip()]
        return "\n\n".join(texts).strip()

    return _invoke_with_retries(endpoint, call, client, cancel_event)


def _send_anthropic(
    endpoint: ProviderEndpoint,
    system_prompt: str,
    user_content: list[dict] | str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    client: httpx.Client | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    del temperature  # Deprecated / rejected on Claude 4.7+ & 5.5+; rely on API defaults.
    base = resolve_base_url("anthropic", endpoint.base_url)
    api_key = resolve_api_key("anthropic", endpoint.api_key)
    if not api_key:
        raise ProviderError("Anthropic API key is required.")
    url = f"{base}/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload: dict = {
        "model": endpoint.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_content}],
    }
    if system_prompt:
        payload["system"] = system_prompt

    def call(http: httpx.Client) -> str:
        resp = http.post(url, headers=headers, json=payload, timeout=endpoint.timeout_seconds)
        resp.raise_for_status()
        blocks = resp.json().get("content") or []
        parts = [str(b.get("text", "")) for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts).strip()

    return _invoke_with_retries(endpoint, call, client, cancel_event)


def _send_google(
    endpoint: ProviderEndpoint,
    system_prompt: str,
    parts: list[dict],
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    client: httpx.Client | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    base = resolve_base_url("google", endpoint.base_url)
    api_key = resolve_api_key("google", endpoint.api_key)
    if not api_key:
        raise ProviderError("Google Gemini API key is required.")
    model_id = endpoint.model.removeprefix("models/")
    url = f"{base}/models/{model_id}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    gen_cfg: dict = {"temperature": temperature}
    if max_tokens is not None:
        gen_cfg["maxOutputTokens"] = max_tokens
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    payload: dict = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": gen_cfg,
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    def call(http: httpx.Client) -> str:
        resp = http.post(url, headers=headers, json=payload, timeout=endpoint.timeout_seconds)
        resp.raise_for_status()
        candidates = resp.json().get("candidates") or []
        if not candidates:
            return ""
        out_parts = candidates[0].get("content", {}).get("parts") or []
        return "\n".join(
            str(p.get("text", "")) for p in out_parts if isinstance(p, dict) and not p.get("thought")
        ).strip()

    return _invoke_with_retries(endpoint, call, client, cancel_event)


def vision_ocr_image_b64(
    image_b64: str,
    endpoint: ProviderEndpoint,
    system_prompt: str,
    user_prompt: str,
    *,
    client: httpx.Client | None = None,
    cancel_event: threading.Event | None = None,
    metrics_out: dict[str, object] | None = None,
) -> str:
    """Run Vision OCR for one base64 PNG image through the configured provider."""
    spec = get_provider_spec(endpoint.provider_id)
    if not endpoint.model.strip():
        raise ProviderError(f"Vision model is required for {spec.label}.")

    if spec.protocol == "ollama":
        cfg = OcrConfig(
            model=endpoint.model,
            provider=spec.id,
            base_url=resolve_base_url(spec.id, endpoint.base_url),
            ollama_url=resolve_base_url(spec.id, endpoint.base_url),
            num_ctx=endpoint.num_ctx,
            timeout_seconds=endpoint.timeout_seconds,
            max_retries=endpoint.max_retries,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return ocr_image_b64(image_b64, cfg, client=client, cancel_event=cancel_event, metrics_out=metrics_out)

    if spec.id == "deepseek":
        model_lower = endpoint.model.strip().lower()
        if model_lower in DEEPSEEK_OCR_TOOL_MODELS:
            return clean_ocr_output(
                _send_deepseek_ocr(
                    image_b64,
                    endpoint,
                    client=client,
                    cancel_event=cancel_event,
                    metrics_out=metrics_out,
                )
            )
        if model_lower in DEEPSEEK_TEXT_ONLY_MODELS:
            raise ProviderError(
                f"DeepSeek model '{endpoint.model}' does not support Vision inputs. Select 'deepseek-ocr' or 'deepseek-flash' instead."
            )

    if spec.protocol == "openai_compat":
        messages = []
        if spec.id == "qwen" and "ocr" in endpoint.model.strip().lower():
            # DashScope Qwen-OCR models use a fixed internal system prompt and reject custom system turns.
            combined = (
                f"{system_prompt.strip()}\n\n{user_prompt.strip()}" if system_prompt.strip() else user_prompt
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": combined},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            )
        else:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            )
        return clean_ocr_output(
            _send_openai_compat(endpoint, messages, temperature=0.0, client=client, cancel_event=cancel_event)
        )

    if spec.protocol == "anthropic":
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
            },
            {"type": "text", "text": user_prompt},
        ]
        return clean_ocr_output(
            _send_anthropic(
                endpoint, system_prompt, content, temperature=0.0, client=client, cancel_event=cancel_event
            )
        )

    if spec.protocol == "google":
        parts = [
            {"text": user_prompt},
            {"inlineData": {"mimeType": "image/png", "data": image_b64}},
        ]
        return clean_ocr_output(
            _send_google(endpoint, system_prompt, parts, temperature=0.0, client=client, cancel_event=cancel_event)
        )

    raise ProviderError(f"Unsupported protocol '{spec.protocol}'.")


def vision_ocr_with_config(
    image_b64: str,
    config: OcrConfig,
    client: httpx.Client | None = None,
    cancel_event: threading.Event | None = None,
    metrics_out: dict[str, object] | None = None,
) -> str:
    """Adapter matching runner's OcrFn signature using OcrConfig provider fields."""
    endpoint = ProviderEndpoint(
        provider_id=config.provider,
        base_url=config.base_url or config.ollama_url,
        api_key=config.api_key,
        model=config.model,
        num_ctx=config.num_ctx,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
    )
    return vision_ocr_image_b64(
        image_b64,
        endpoint,
        config.system_prompt,
        config.user_prompt,
        client=client,
        cancel_event=cancel_event,
        metrics_out=metrics_out,
    )


def chat_text(
    endpoint: ProviderEndpoint,
    system_prompt: str,
    user_prompt: str,
    *,
    json_mode: bool = False,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    client: httpx.Client | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Run text-only completion on the configured Assistant LLM provider."""
    spec = get_provider_spec(endpoint.provider_id)
    if not endpoint.model.strip():
        raise ProviderError(f"LLM model is required for {spec.label}.")
    if spec.id == "deepseek" and endpoint.model.strip().lower() in DEEPSEEK_OCR_TOOL_MODELS:
        raise ProviderError(
            f"'{endpoint.model}' is a dedicated OCR tool and cannot be used as a text LLM. Select 'deepseek-flash' or 'deepseek-v4-pro' instead."
        )

    if spec.protocol == "ollama":
        base = resolve_base_url(spec.id, endpoint.base_url)
        url = f"{base}/api/chat"
        options: dict[str, object] = {"temperature": temperature, "num_ctx": endpoint.num_ctx}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict = {
            "model": endpoint.model,
            "stream": False,
            "options": options,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            payload["format"] = "json"
        if endpoint.model.lower().startswith("qwen3"):
            payload["think"] = False

        def call(http: httpx.Client) -> str:
            resp = http.post(url, json=payload, timeout=endpoint.timeout_seconds)
            resp.raise_for_status()
            return str(resp.json().get("message", {}).get("content", "")).strip()

        return _invoke_with_retries(endpoint, call, client, cancel_event)

    if spec.protocol == "openai_compat":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return _send_openai_compat(
            endpoint,
            messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client,
            cancel_event=cancel_event,
        )

    if spec.protocol == "anthropic":
        return _send_anthropic(
            endpoint,
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client,
            cancel_event=cancel_event,
        )

    if spec.protocol == "google":
        return _send_google(
            endpoint,
            system_prompt,
            [{"text": user_prompt}],
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            client=client,
            cancel_event=cancel_event,
        )

    raise ProviderError(f"Unsupported protocol '{spec.protocol}'.")


def test_provider_connection(
    provider_id: str,
    base_url: str = "",
    api_key: str = "",
    timeout: float = 5.0,
) -> dict:
    """Probe a provider endpoint and list discovered models if supported."""
    spec = get_provider_spec(provider_id)
    resolved_url = resolve_base_url(spec.id, base_url)
    resolved_key = resolve_api_key(spec.id, api_key)

    if spec.id == "custom" and not resolved_url:
        return {
            "provider": spec.id,
            "online": False,
            "installed": False,
            "has_key": bool(resolved_key),
            "status_code": "no_url",
            "status_text": STATUS_LABELS["no_url"],
            "models": [],
            "message": "Custom provider Base URL is required.",
        }

    if spec.requires_api_key and not resolved_key:
        return {
            "provider": spec.id,
            "online": False,
            "installed": False,
            "has_key": False,
            "status_code": "no_key",
            "status_text": STATUS_LABELS["no_key"],
            "models": list(spec.suggested_vision_models or spec.suggested_llm_models),
            "message": f"Missing API key (enter below or set {spec.env_var}).",
        }

    if spec.protocol == "ollama":
        installed = find_ollama_executable() is not None if spec.id == "ollama_local" else True
        try:
            detailed = list_models_detailed(resolved_url, timeout=timeout)
            names = [m["name"] for m in detailed]
            return {
                "provider": spec.id,
                "online": True,
                "installed": True,
                "has_key": True,
                "status_code": "ok",
                "status_text": "",
                "models": names,
                "detailed": detailed,
                "message": f"Connected ({len(names)} installed model(s)).",
            }
        except httpx.HTTPError as exc:
            return {
                "provider": spec.id,
                "online": False,
                "installed": installed,
                "has_key": True,
                "status_code": "offline",
                "status_text": STATUS_LABELS["offline"],
                "models": [],
                "message": f"Ollama unreachable at {resolved_url}: {exc}",
            }

    try:
        if spec.protocol == "google":
            resp = httpx.get(
                f"{resolved_url}/models",
                headers={"x-goog-api-key": resolved_key},
                params={"pageSize": 100},
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("models") or []
            names = _filter_discovered_models(
                "google",
                [
                    str(m.get("name", "")).removeprefix("models/")
                    for m in raw
                    if m.get("name")
                    and (
                        "supportedGenerationMethods" not in m
                        or "generateContent" in (m.get("supportedGenerationMethods") or [])
                    )
                ],
            )
        elif spec.protocol == "anthropic":
            resp = httpx.get(
                f"{resolved_url}/models",
                headers={"x-api-key": resolved_key, "anthropic-version": "2023-06-01"},
                params={"limit": 100},
                timeout=timeout,
            )
            resp.raise_for_status()
            names = sorted({str(m.get("id", "")) for m in (resp.json().get("data") or []) if m.get("id")})
        else:
            headers = {"Authorization": f"Bearer {resolved_key}"} if resolved_key else {}
            resp = httpx.get(f"{resolved_url}/models", headers=headers, timeout=timeout)
            resp.raise_for_status()
            names = _filter_discovered_models(
                spec.id,
                [str(m.get("id", "")) for m in (resp.json().get("data") or []) if m.get("id")],
            )

        return {
            "provider": spec.id,
            "online": True,
            "installed": True,
            "has_key": bool(resolved_key),
            "status_code": "ok",
            "status_text": "",
            "models": names,
            "message": f"Connected ({len(names)} model(s) listed).",
        }
    except httpx.HTTPStatusError as exc:
        code = classify_http_error(exc.response.status_code, exc.response.text[:300])
        return {
            "provider": spec.id,
            "online": False,
            "installed": True,
            "has_key": bool(resolved_key),
            "status_code": code,
            "status_text": STATUS_LABELS.get(code, "Error"),
            "models": [],
            "message": _format_http_error(spec, exc),
        }
    except (httpx.HTTPError, ValueError) as exc:
        return {
            "provider": spec.id,
            "online": False,
            "installed": spec.is_local,
            "has_key": bool(resolved_key),
            "status_code": "offline",
            "status_text": STATUS_LABELS["offline"],
            "models": [],
            "message": f"{spec.label} unreachable: {exc}",
        }


def probe_endpoint_status(
    endpoint: ProviderEndpoint,
    *,
    enabled: bool = True,
    timeout: float = 3.5,
) -> dict:
    """Probe an active role endpoint and return normalized health + optional status badge text."""
    if not enabled:
        return {
            "status": "disabled",
            "status_text": STATUS_LABELS["disabled"],
            "online": False,
            "installed": False,
            "models": [],
            "detailed": [],
            "message": f"{endpoint.credential_name or endpoint.provider_id} is disabled.",
        }

    res = test_provider_connection(
        endpoint.provider_id,
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        timeout=timeout,
    )
    status_code = str(res.get("status_code") or ("ok" if res.get("online") else "error"))
    spec = get_provider_spec(endpoint.provider_id)
    model_name = endpoint.model.strip()
    if status_code == "ok" and spec.protocol == "ollama" and model_name:
        installed_names = set(res.get("models") or [])
        if model_name not in installed_names and f"{model_name}:latest" not in installed_names:
            status_code = "model_missing"
            res["message"] = f"Model '{model_name}' is not pulled on {endpoint.credential_name or spec.label}."

    return {
        "status": status_code,
        "status_text": STATUS_LABELS.get(status_code, ""),
        "online": bool(res.get("online")) and status_code == "ok",
        "installed": bool(res.get("installed")),
        "models": list(res.get("models") or []),
        "detailed": list(res.get("detailed") or []),
        "message": str(res.get("message") or ""),
    }


def check_endpoint_ready(endpoint: ProviderEndpoint, *, require_vision: bool = False) -> None:
    """Validate that a configured Vision or LLM endpoint is ready before running."""
    spec = get_provider_spec(endpoint.provider_id)
    if require_vision and not spec.supports_vision:
        raise ProviderError(
            f"{spec.label} does not support Vision OCR. Pick a vision provider (e.g. Ollama, llama.cpp, OpenAI, Anthropic, Google, Grok, Qwen)."
        )
    if not endpoint.model.strip():
        raise ProviderError(f"Model name is required for {spec.label}.")
    if require_vision and spec.id == "deepseek" and endpoint.model.strip().lower() in DEEPSEEK_TEXT_ONLY_MODELS:
        raise ProviderError(
            f"DeepSeek model '{endpoint.model}' does not support Vision inputs. Select 'deepseek-ocr' or 'deepseek-flash' instead."
        )
    if spec.protocol == "ollama":
        check_ollama_ready(resolve_base_url(spec.id, endpoint.base_url), endpoint.model)
        return
    if spec.id == "custom" and not resolve_base_url("custom", endpoint.base_url):
        raise ProviderError("Custom provider Base URL is required.")
    if spec.requires_api_key and not resolve_api_key(spec.id, endpoint.api_key):
        raise ProviderError(f"{spec.label} API key is required (enter in Models & Providers or set {spec.env_var}).")
    if spec.is_local:
        url = f"{resolve_base_url(spec.id, endpoint.base_url)}/models"
        try:
            resp = httpx.get(url, headers=_openai_headers(resolve_api_key(spec.id, endpoint.api_key)), timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cannot reach {spec.label} at {resolve_base_url(spec.id, endpoint.base_url)}: {exc}") from exc
