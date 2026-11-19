"""Local Ollama REST API client for Vision-Language OCR."""

from collections.abc import Callable
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import threading
import time

import httpx

from .config import OcrConfig


class ProviderError(RuntimeError):
    """Raised when a model provider is unreachable, misconfigured, or fails inference."""


class OllamaError(ProviderError):
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


def pull_model_stream(
    base_url: str,
    model: str,
    on_progress: Callable[[dict], None],
    timeout: float = 1800.0,
) -> None:
    """Pull an Ollama model with streaming layer progress callbacks."""
    url = f"{base_url.rstrip('/')}/api/pull"
    try:
        with httpx.stream("POST", url, json={"model": model, "stream": True}, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if "error" in chunk:
                    raise OllamaError(str(chunk["error"]))
                total = int(chunk.get("total") or 0)
                completed = int(chunk.get("completed") or 0)
                percent = round((completed / total) * 100, 1) if total > 0 else None
                on_progress(
                    {
                        "status": chunk.get("status", "pulling"),
                        "completed": completed,
                        "total": total,
                        "percent": percent,
                    }
                )
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


def find_ollama_executable() -> str | None:
    """Locate the local Ollama binary across Windows, macOS, and Linux."""
    found = shutil.which("ollama")
    if found:
        return found

    system = platform.system()
    candidates: list[Path] = []
    if system == "Windows":
        local_app = os.environ.get("LOCALAPPDATA", "")
        prog_files = os.environ.get("ProgramFiles", "")
        if local_app:
            candidates.append(Path(local_app) / "Programs" / "Ollama" / "ollama.exe")
        if prog_files:
            candidates.append(Path(prog_files) / "Ollama" / "ollama.exe")
    elif system == "Darwin":
        candidates.extend(
            [
                Path("/Applications/Ollama.app/Contents/MacOS/Ollama"),
                Path("/opt/homebrew/bin/ollama"),
                Path("/usr/local/bin/ollama"),
            ]
        )
    else:
        candidates.extend([Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")])

    for path in candidates:
        if path.is_file():
            return str(path)
    return None


_OLLAMA_LAUNCH_LOCK = threading.Lock()


def _is_ollama_process_running() -> bool:
    """Return True if an Ollama server/app process is already active."""
    try:
        import psutil

        for proc in psutil.process_iter(["name"]):
            name = (proc.info.get("name") or "").lower()
            if name in ("ollama", "ollama.exe", "ollama app.exe"):
                return True
    except Exception:
        pass
    return False


def _poll_ollama_ready(tags_url: str, wait_seconds: float) -> bool:
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(tags_url, timeout=1.0)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            time.sleep(0.35)
    return False


def start_ollama_service(base_url: str, wait_seconds: float = 20.0) -> dict:
    """Attempt to start `ollama serve` silently if Ollama is installed but offline."""
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    with _OLLAMA_LAUNCH_LOCK:
        try:
            resp = httpx.get(tags_url, timeout=1.5)
            if resp.status_code == 200:
                return {"online": True, "installed": True, "started": False, "message": "Ollama is already running."}
        except httpx.HTTPError:
            pass

        exe = find_ollama_executable()
        if not exe:
            return {
                "online": False,
                "installed": False,
                "started": False,
                "message": "Ollama executable was not found on this machine.",
            }

        already_running = _is_ollama_process_running()
        started_new = False
        if not already_running:
            kwargs: dict = {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if platform.system() == "Windows":
                # Use CREATE_NO_WINDOW alone (not DETACHED_PROCESS) + SW_HIDE so child probes don't flash consoles
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0
                kwargs["startupinfo"] = startupinfo
            else:
                kwargs["start_new_session"] = True

            try:
                subprocess.Popen([exe, "serve"], **kwargs)
                started_new = True
            except OSError as exc:
                return {
                    "online": False,
                    "installed": True,
                    "started": False,
                    "message": f"Failed to launch Ollama: {exc}",
                }

        if _poll_ollama_ready(tags_url, wait_seconds):
            msg = "Ollama started." if started_new else "Ollama finished starting up."
            return {"online": True, "installed": True, "started": started_new, "message": msg}

        return {
            "online": False,
            "installed": True,
            "started": started_new,
            "message": "Ollama is still warming up. Click Refresh in a moment.",
        }


_TIMING_KEYS = (
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
    "done_reason",
)

_LAYOUT_REF_LABELS: frozenset[str] = frozenset(
    {
        "title",
        "sub_title",
        "subtitle",
        "section_header",
        "text",
        "plain_text",
        "paragraph",
        "equation",
        "equation_number",
        "formula",
        "table",
        "table_caption",
        "table_footnote",
        "image",
        "image_caption",
        "image_footnote",
        "figure",
        "figure_caption",
        "caption",
        "header",
        "footer",
        "page_header",
        "page_footer",
        "page_number",
        "page_footnote",
        "footnote",
        "list",
        "list_item",
        "code",
        "reference",
        "abstract",
        "author",
    }
)

_RE_THINK_BLOCK = re.compile(r"<(?:think|thought)>.*?</(?:think|thought)>", re.DOTALL | re.IGNORECASE)
_RE_DEEPSEEK_REF_TAG = re.compile(
    r"(?P<prefix>^[ \t]*)?<[|｜]ref[|｜]>(?P<label>.*?)<[|｜]/ref[|｜]>"
    r"(?:\s*<[|｜](?:box|det)[|｜]>.*?<[|｜]/(?:box|det)[|｜]>)?[ \t]*",
    re.DOTALL | re.MULTILINE,
)
_RE_DEEPSEEK_BOX_STANDALONE = re.compile(
    r"<[|｜](?:box|det)[|｜]>.*?<[|｜]/(?:box|det)[|｜]>[ \t]*",
    re.DOTALL,
)
_RE_QWEN_OBJ_REF = re.compile(r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>", re.DOTALL)
_RE_QWEN_BOX = re.compile(r"<\|(?:box|quad)_start\|>.*?<\|(?:box|quad)_end\|>[ \t]*", re.DOTALL)
_RE_XML_REF_BOX = re.compile(r"<ref>(.*?)</ref>\s*<box>[\d,\s()\[\].;-]+</box>[ \t]*", re.DOTALL)
_RE_XML_BOX_ONLY = re.compile(r"<box>[\d,\s()\[\].;-]+</box>[ \t]*", re.DOTALL)
_RE_GLM_BOX = re.compile(r"<\|begin_of_box\|>.*?<\|end_of_box\|>[ \t]*", re.DOTALL)
_RE_LOC_TOKEN = re.compile(r"<[|｜]?loc_?\d+[|｜]?>")
_RE_STRAY_VLM_TOKENS = re.compile(
    r"<[|｜]/?(?:"
    r"grounding|ref|det|box|image|vision_start|vision_end|image_pad|video_pad|"
    r"object_ref_start|object_ref_end|box_start|box_end|quad_start|quad_end|"
    r"begin_of_box|end_of_box|endoftext|im_start|im_end|eot_id|end_of_text|"
    r"begin_of_text|user|assistant|observation|end▁of▁sentence|begin▁of▁sentence|eos|bos"
    r")[|｜]>|<end_of_turn>",
    re.IGNORECASE,
)
_RE_OUTER_MD_FENCE = re.compile(r"^```(?:markdown|md)[ \t]*\r?\n(.*?)\r?\n```$", re.DOTALL | re.IGNORECASE)
_RE_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def clean_ocr_output(text: str) -> str:
    """Strip VLM grounding, bounding-box, reference, and control tokens into clean Markdown."""
    if not text:
        return ""

    # 1. Strip hidden reasoning blocks emitted by thinking VLMs
    cleaned = _RE_THINK_BLOCK.sub("", text)

    # 2. Strip/unwrap DeepSeek <｜ref｜>...<｜/ref｜><｜box｜|det｜>...<｜/box｜|det｜> grounding marks
    def _replace_ref(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        at_line_start = prefix is not None
        leading = prefix or ""
        label_raw = match.group("label").strip()
        label_norm = label_raw.lower()
        after = match.string[match.end() :]
        next_ch = after[0] if after else ""

        if label_norm in _LAYOUT_REF_LABELS:
            if at_line_start and next_ch and next_ch not in " \t\r\n#":
                if label_norm == "title":
                    return f"{leading}# "
                if label_norm in ("sub_title", "subtitle", "section_header"):
                    return f"{leading}## "
            return leading

        needs_space = bool(
            match.group(0).endswith((" ", "\t")) and next_ch and next_ch not in " \t\r\n"
        )
        return f"{leading}{label_raw}{' ' if needs_space else ''}"

    cleaned = _RE_DEEPSEEK_REF_TAG.sub(_replace_ref, cleaned)

    # 3. Strip remaining provider-specific box / ref / loc tokens
    cleaned = _RE_DEEPSEEK_BOX_STANDALONE.sub("", cleaned)
    cleaned = _RE_QWEN_OBJ_REF.sub(r"\1", cleaned)
    cleaned = _RE_QWEN_BOX.sub("", cleaned)
    cleaned = _RE_XML_REF_BOX.sub(r"\1", cleaned)
    cleaned = _RE_XML_BOX_ONLY.sub("", cleaned)
    cleaned = _RE_GLM_BOX.sub("", cleaned)
    cleaned = _RE_LOC_TOKEN.sub("", cleaned)

    # 4. Strip stray known VLM control tokens (leaving LaTeX/physics like <|psi|> untouched)
    cleaned = _RE_STRAY_VLM_TOKENS.sub("", cleaned)

    # 5. Unwrap a single outer ```markdown ... ``` fence if the whole response was wrapped
    cleaned = cleaned.strip()
    fence_match = _RE_OUTER_MD_FENCE.match(cleaned)
    if fence_match and "```" not in fence_match.group(1):
        cleaned = fence_match.group(1)

    # 6. Normalize trailing spaces and excessive blank lines
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    cleaned = _RE_EXCESS_NEWLINES.sub("\n\n", cleaned)
    return cleaned.strip()


def ocr_image_b64(
    image_b64: str,
    config: OcrConfig,
    client: httpx.Client | None = None,
    cancel_event: threading.Event | None = None,
    metrics_out: dict[str, object] | None = None,
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
                data = resp.json()
                if metrics_out is not None and isinstance(data, dict):
                    for key in _TIMING_KEYS:
                        if key in data:
                            metrics_out[key] = data[key]
                content = data.get("message", {}).get("content", "") if isinstance(data, dict) else ""
                return clean_ocr_output(str(content))
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
