"""Local Ollama REST API client for Vision-Language OCR."""

import threading
import time

import httpx

from .config import OcrConfig


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable, missing a model, or fails inference."""


class JobCancelledError(RuntimeError):
    """Raised when an active OCR request is aborted by user cancellation."""


def check_ollama_ready(base_url: str, model: str, timeout: float = 5.0) -> None:
    """Verify Ollama server is reachable and the requested model is available locally."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {base_url}. "
            "Start Ollama (e.g. run `ollama serve` or open the Ollama app) and try again."
        ) from exc

    models = response.json().get("models", [])
    available = {m.get("name", "") for m in models}
    if model in available or f"{model}:latest" in available:
        return

    raise OllamaError(
        f"Model '{model}' is not pulled in Ollama. "
        f"Run `ollama pull {model}` first (available: {', '.join(sorted(available)) or 'none'})."
    )


def pull_model(base_url: str, model: str, timeout: float = 600.0) -> None:
    """Pull a model into local Ollama synchronously."""
    url = f"{base_url.rstrip('/')}/api/pull"
    try:
        response = httpx.post(url, json={"model": model, "stream": False}, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Failed to pull model '{model}': {exc}") from exc


def list_models_detailed(base_url: str, timeout: float = 4.0) -> list[dict]:
    """Return installed Ollama models with size, parameter count, and quantization metadata."""
    url = f"{base_url.rstrip('/')}/api/tags"
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    result: list[dict] = []
    for item in response.json().get("models", []):
        name = item.get("name", "")
        if not name:
            continue
        details = item.get("details") or {}
        size_bytes = item.get("size") or 0
        result.append(
            {
                "name": name,
                "size_gb": round(size_bytes / (1024**3), 2),
                "parameter_size": details.get("parameter_size", ""),
                "quantization_level": details.get("quantization_level", ""),
                "family": details.get("family", ""),
                "modified_at": item.get("modified_at", ""),
            }
        )
    return sorted(result, key=lambda m: m["name"].lower())


def delete_model(base_url: str, model: str, timeout: float = 15.0) -> None:
    """Delete an installed Ollama model."""
    url = f"{base_url.rstrip('/')}/api/delete"
    try:
        response = httpx.request("DELETE", url, json={"model": model}, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Failed to delete model '{model}': {exc}") from exc


def ocr_image_b64(
    image_b64: str,
    config: OcrConfig,
    client: httpx.Client | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Send a base64-encoded image to Ollama /api/chat and return transcribed Markdown."""
    url = f"{config.ollama_url.rstrip('/')}/api/chat"
    payload: dict = {
        "model": config.model,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": config.num_ctx},
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": config.user_prompt, "images": [image_b64]},
        ],
    }
    # Disable reasoning/thinking tokens on Qwen3 VL models to avoid 20s+ overhead per page
    if config.model.lower().startswith("qwen3"):
        payload["think"] = False

    owns_client = client is None
    http = client or httpx.Client(timeout=config.timeout_seconds)
    last_exc: Exception | None = None
    try:
        for attempt in range(config.max_retries + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelledError("Cancelled by user.")
            try:
                resp = http.post(url, json=payload, timeout=config.timeout_seconds)
                resp.raise_for_status()
                content = resp.json().get("message", {}).get("content", "")
                return content.strip()
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                if cancel_event is not None and cancel_event.is_set():
                    raise JobCancelledError("Cancelled by user.") from exc
                last_exc = exc
                if attempt < config.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise OllamaError(f"Ollama OCR failed after {config.max_retries + 1} attempts: {last_exc}") from last_exc
    finally:
        if owns_client:
            http.close()
