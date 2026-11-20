"""VRAM budgeting, dynamic context sizing, and structure-aware Markdown chunking for LLMs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import threading
import time

import httpx

from .platform_utils import get_system_info
from .providers import ProviderEndpoint, get_provider_spec, resolve_base_url

CONTEXT_BUCKETS: tuple[int, ...] = (
    2048,
    4096,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
)

_PARAM_SIZE_RE = re.compile(r"(?:^|[:\-_/@])(\d+(?:\.\d+)?)\s*([bm])\b", re.IGNORECASE)
_QUANT_BITS_RE = re.compile(r"\b(f16|fp16|bf16|q8(?:_\d+)?|q6_[k\d]+|q5_[k\d]+|q4_[k\d]+|q3_[k\d]+|q2_[k\d]+)\b", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PAGE_MARKER_RE = re.compile(r"^<!--\s*Page\s+(\d+)\b.*?-->$", re.IGNORECASE)
_OCR_ERROR_MARKER_RE = re.compile(r"^<!--\s*OCR\s+ERROR\b.*?-->$", re.IGNORECASE)

_SYS_CACHE_LOCK = threading.Lock()
_SYS_CACHE: dict[str, object] = {"ts": 0.0, "data": None}
_SHOW_CACHE: dict[tuple[str, str], dict] = {}


@dataclass(frozen=True, slots=True)
class ModelMemoryProfile:
    """Memory parameters for estimating weight and KV-cache VRAM consumption."""

    params_b: float
    weights_gb: float
    kv_bytes_per_token: int
    model_max_ctx: int
    quantization: str = ""
    from_inspect: bool = False


@dataclass(frozen=True, slots=True)
class VramPlan:
    """Calculated VRAM footprint and safe context recommendation for an LLM configuration."""

    provider_id: str
    model: str
    is_local: bool
    hardware_label: str
    budget_gb: float | None
    params_b: float
    weights_gb: float
    kv_kb_per_token: float
    model_max_ctx: int
    recommended_max_ctx: int
    vram_at_4k_gb: float
    vram_at_recommended_gb: float
    fits_in_vram: bool
    split_threshold_tokens: int
    split_threshold_chars: int
    summary_text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContextSelection:
    """Resolved context window and token budgets for a specific LLM task/document."""

    chosen_num_ctx: int
    estimated_vram_gb: float
    estimated_prompt_tokens: int
    max_single_pass_input_tokens: int
    needs_splitting: bool
    vram_plan: VramPlan


@dataclass(frozen=True, slots=True)
class CleanupBudget:
    """Context window and max-output-bounded slice budget for OCR proofreading."""

    chosen_num_ctx: int
    max_output_tokens: int
    max_slice_input_tokens: int
    estimated_vram_gb: float
    body_tokens: int
    needs_splitting: bool
    vram_plan: VramPlan


def get_cached_system_info(ttl_seconds: float = 60.0) -> dict:
    """Return cached hardware info so batch operations avoid repeated nvidia-smi calls."""
    now = time.monotonic()
    with _SYS_CACHE_LOCK:
        cached = _SYS_CACHE.get("data")
        ts = float(_SYS_CACHE.get("ts") or 0.0)
        if isinstance(cached, dict) and (now - ts) < ttl_seconds:
            return cached
    info = get_system_info()
    with _SYS_CACHE_LOCK:
        _SYS_CACHE["data"] = info
        _SYS_CACHE["ts"] = now
    return info


def estimate_tokens(text: str) -> int:
    """Conservative token count estimator for mixed Markdown, LaTeX, and CJK text."""
    if not text:
        return 0
    ascii_chars = 0
    non_ascii_chars = 0
    for ch in text:
        if ord(ch) < 128:
            ascii_chars += 1
        else:
            non_ascii_chars += 1
    return max(1, math.ceil(ascii_chars / 3.5 + non_ascii_chars * 1.2))


def parse_model_params_b(model_name: str, parameter_size: str = "") -> float:
    """Extract parameter count (in billions) from metadata or model tag."""
    for candidate in (parameter_size, model_name):
        if not candidate:
            continue
        match = _PARAM_SIZE_RE.search(f":{candidate.strip()}")
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            return value if unit == "b" else round(value / 1000.0, 3)
    lower = (model_name or "").lower()
    if any(tok in lower for tok in ("mini", "nano", "flash-lite", "haiku", "micro")):
        return 2.0
    return 3.0


def _bytes_per_param(quantization: str, model_name: str = "") -> float:
    q = (quantization or "").strip().lower()
    if not q:
        m = _QUANT_BITS_RE.search(model_name or "")
        q = m.group(1).lower() if m else ""
    if q.startswith(("f16", "fp16", "bf16")):
        return 2.05
    if q.startswith("q8"):
        return 1.08
    if q.startswith("q6"):
        return 0.84
    if q.startswith("q5"):
        return 0.72
    if q.startswith("q3"):
        return 0.50
    if q.startswith("q2"):
        return 0.38
    return 0.62  # Typical Q4_K_M / 4-bit GGUF average including embeddings & scales


def _default_kv_bytes_per_token(params_b: float) -> int:
    """Typical FP16 K+V cache bytes per token for modern GQA small/medium transformers."""
    if params_b <= 1.0:
        return 49_152  # ~48 KB/tok (e.g. 0.5B-0.9B)
    if params_b <= 4.5:
        return 114_688  # ~112 KB/tok (e.g. Qwen3-1.7B, Llama-3.2-3B, Gemma3-4B: 28L x 8kv x 128d)
    if params_b <= 9.5:
        return 131_072  # ~128 KB/tok (e.g. Llama-3-8B / Qwen2.5-7B: 32L x 8kv x 128d)
    if params_b <= 16.0:
        return 196_608  # ~192 KB/tok (e.g. 14B)
    return 262_144  # ~256 KB/tok (e.g. 32B+)


def inspect_ollama_model_specs(
    base_url: str,
    model: str,
    *,
    timeout: float = 1.0,
    client: httpx.Client | None = None,
) -> dict:
    """Optionally query Ollama /api/show for exact layer, KV-head, and max-context specs."""
    resolved_url = resolve_base_url("ollama", base_url)
    key = (resolved_url, model.strip())
    if not key[1]:
        return {}
    with _SYS_CACHE_LOCK:
        if key in _SHOW_CACHE:
            return _SHOW_CACHE[key]

    try:
        url = f"{resolved_url}/api/show"
        if client is not None:
            resp = client.post(url, json={"model": key[1]}, timeout=timeout)
        else:
            resp = httpx.post(url, json={"model": key[1]}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json() if isinstance(resp.json(), dict) else {}
    except (httpx.HTTPError, ValueError):
        return {}

    info = payload.get("model_info") or {}
    details = payload.get("details") or {}
    arch = str(info.get("general.architecture") or "").strip()

    def _find_info(suffix: str) -> int:
        if arch and isinstance(info.get(f"{arch}.{suffix}"), (int, float)):
            return int(info[f"{arch}.{suffix}"])
        for k, v in info.items():
            if str(k).endswith(f".{suffix}") and isinstance(v, (int, float)):
                return int(v)
        return 0

    layers = _find_info("block_count")
    kv_heads = _find_info("attention.head_count_kv")
    heads = _find_info("attention.head_count")
    emb_len = _find_info("embedding_length")
    key_len = _find_info("attention.key_length") or (emb_len // heads if (emb_len and heads) else 0)
    ctx_len = _find_info("context_length")

    kv_bytes = 2 * layers * kv_heads * key_len * 2 if (layers and kv_heads and key_len) else 0
    extracted = {
        "parameter_size": str(details.get("parameter_size") or ""),
        "quantization_level": str(details.get("quantization_level") or ""),
        "kv_bytes_per_token": kv_bytes,
        "context_length": ctx_len,
    }
    with _SYS_CACHE_LOCK:
        _SHOW_CACHE[key] = extracted
    return extracted


def build_model_memory_profile(
    provider_id: str,
    model: str,
    *,
    model_size_gb: float | None = None,
    parameter_size: str = "",
    quantization: str = "",
    ollama_specs: dict | None = None,
) -> ModelMemoryProfile:
    """Construct weight and KV-cache profile from runtime metadata or parameter heuristics."""
    specs = ollama_specs or {}
    param_str = str(specs.get("parameter_size") or parameter_size or "")
    quant_str = str(specs.get("quantization_level") or quantization or "")
    params_b = parse_model_params_b(model, param_str)

    if model_size_gb and model_size_gb > 0:
        weights_gb = float(model_size_gb)
    else:
        weights_gb = round(max(0.25, params_b * _bytes_per_param(quant_str, model)), 2)

    kv_bytes = int(specs.get("kv_bytes_per_token") or 0) or _default_kv_bytes_per_token(params_b)
    raw_max_ctx = int(specs.get("context_length") or 0)
    spec = get_provider_spec(provider_id)
    default_max_ctx = 32768 if spec.is_local else 65536
    model_max_ctx = raw_max_ctx if raw_max_ctx >= 2048 else default_max_ctx

    return ModelMemoryProfile(
        params_b=params_b,
        weights_gb=weights_gb,
        kv_bytes_per_token=kv_bytes,
        model_max_ctx=model_max_ctx,
        quantization=quant_str,
        from_inspect=bool(specs.get("kv_bytes_per_token")),
    )


def estimate_vram_gb_for_ctx(profile: ModelMemoryProfile, num_ctx: int) -> float:
    """Estimate total GPU/unified memory usage (weights + FP16 KV cache + runtime overhead) in GB."""
    kv_gb = (profile.kv_bytes_per_token * max(512, num_ctx)) / (1024**3)
    compute_overhead_gb = 0.35 + (max(512, num_ctx) / 8192.0) * 0.12
    return round(profile.weights_gb + kv_gb + compute_overhead_gb, 2)


def calculate_vram_plan(
    endpoint: ProviderEndpoint,
    *,
    system_info: dict | None = None,
    model_size_gb: float | None = None,
    parameter_size: str = "",
    quantization: str = "",
    probe_ollama: bool = False,
) -> VramPlan:
    """Compute VRAM usage and the highest safe context window for the current hardware."""
    spec = get_provider_spec(endpoint.provider_id)
    sys_info = system_info if system_info is not None else get_cached_system_info()
    ollama_specs: dict = {}
    if spec.protocol == "ollama" and probe_ollama and endpoint.model.strip():
        ollama_specs = inspect_ollama_model_specs(endpoint.base_url, endpoint.model)

    profile = build_model_memory_profile(
        spec.id,
        endpoint.model,
        model_size_gb=model_size_gb,
        parameter_size=parameter_size,
        quantization=quantization,
        ollama_specs=ollama_specs,
    )

    if not spec.is_local:
        rec_ctx = min(profile.model_max_ctx, 65536)
        split_tokens = max(2048, int(rec_ctx * 0.68) - 1024)
        return VramPlan(
            provider_id=spec.id,
            model=endpoint.model,
            is_local=False,
            hardware_label=f"Cloud-managed ({spec.label})",
            budget_gb=None,
            params_b=profile.params_b,
            weights_gb=0.0,
            kv_kb_per_token=round(profile.kv_bytes_per_token / 1024.0, 1),
            model_max_ctx=profile.model_max_ctx,
            recommended_max_ctx=rec_ctx,
            vram_at_4k_gb=0.0,
            vram_at_recommended_gb=0.0,
            fits_in_vram=True,
            split_threshold_tokens=split_tokens,
            split_threshold_chars=split_tokens * 3,
            summary_text=(
                f"Cloud inference ({spec.label}) · uses 0 GB local VRAM · "
                f"auto-splits docs over ~{split_tokens:,} tokens (~{split_tokens * 3 // 1000}k chars)."
            ),
        )

    gpu_vram = sys_info.get("vram_gb")
    gpu_name = str(sys_info.get("gpu_name") or "").strip()
    ram_gb = float(sys_info.get("ram_gb") or 8.0)

    if isinstance(gpu_vram, (int, float)) and gpu_vram > 0:
        raw_vram = float(gpu_vram)
        # Keep >= 0.8 GB or 12% reserve for display compositor, CUDA buffers & OS headroom.
        budget_gb = round(max(1.5, raw_vram - max(0.8, raw_vram * 0.12)), 2)
        hardware_label = f"{gpu_name or 'GPU'} ({raw_vram:.1f} GB VRAM, {budget_gb:.1f} GB usable)"
        ctx_ceiling = min(profile.model_max_ctx, 65536)
    else:
        # CPU / unified RAM: stay under 45% system RAM and cap auto context to avoid multi-minute prompt eval.
        budget_gb = round(max(2.0, min(ram_gb * 0.45, 24.0)), 2)
        hardware_label = f"CPU / Unified RAM ({ram_gb:.1f} GB total, {budget_gb:.1f} GB LLM budget)"
        ctx_ceiling = min(profile.model_max_ctx, 8192 if ram_gb < 16 else 16384)

    fitting_buckets = [
        b
        for b in CONTEXT_BUCKETS
        if b <= ctx_ceiling and estimate_vram_gb_for_ctx(profile, b) <= budget_gb
    ]
    if fitting_buckets:
        rec_ctx = fitting_buckets[-1]
        fits = True
    else:
        rec_ctx = 2048
        fits = estimate_vram_gb_for_ctx(profile, 2048) <= budget_gb

    vram_4k = estimate_vram_gb_for_ctx(profile, 4096)
    vram_rec = estimate_vram_gb_for_ctx(profile, rec_ctx)
    split_tokens = max(1200, int(rec_ctx * 0.70) - 768)
    fit_note = (
        f"fits within {budget_gb:.1f} GB budget"
        if fits
        else f"exceeds {budget_gb:.1f} GB VRAM budget — may spill to RAM"
    )
    summary = (
        f"{hardware_label} · ~{profile.weights_gb:.1f} GB weights + ~{profile.kv_bytes_per_token / 1024.0:.0f} KB/tok KV · "
        f"recommended max {rec_ctx:,} ctx (~{vram_rec:.2f} GB, {fit_note}); auto-splits above ~{split_tokens:,} tokens."
    )

    return VramPlan(
        provider_id=spec.id,
        model=endpoint.model,
        is_local=True,
        hardware_label=hardware_label,
        budget_gb=budget_gb,
        params_b=profile.params_b,
        weights_gb=profile.weights_gb,
        kv_kb_per_token=round(profile.kv_bytes_per_token / 1024.0, 1),
        model_max_ctx=profile.model_max_ctx,
        recommended_max_ctx=rec_ctx,
        vram_at_4k_gb=vram_4k,
        vram_at_recommended_gb=vram_rec,
        fits_in_vram=fits,
        split_threshold_tokens=split_tokens,
        split_threshold_chars=split_tokens * 3,
        summary_text=summary,
    )


def choose_adequate_context(
    endpoint: ProviderEndpoint,
    prompt_text: str,
    *,
    expected_output_tokens: int = 512,
    auto_ctx: bool | None = None,
    system_info: dict | None = None,
    max_cap_ctx: int | None = None,
) -> ContextSelection:
    """Pick the smallest safe context bucket that fits the prompt within hardware VRAM limits."""
    use_auto = endpoint.auto_ctx if auto_ctx is None else auto_ctx
    plan = calculate_vram_plan(endpoint, system_info=system_info, probe_ollama=False)
    profile = build_model_memory_profile(endpoint.provider_id, endpoint.model)

    prompt_tokens = estimate_tokens(prompt_text)
    reserve_tokens = max(256, expected_output_tokens) + 320  # System prompt + formatting margin
    needed_total = prompt_tokens + reserve_tokens

    ceiling = plan.recommended_max_ctx
    if max_cap_ctx is not None:
        ceiling = min(ceiling, max_cap_ctx)

    if not use_auto:
        chosen = max(1024, min(int(endpoint.num_ctx or 4096), plan.model_max_ctx))
        if max_cap_ctx is not None:
            chosen = min(chosen, max_cap_ctx)
    else:
        min_floor = 4096 if plan.recommended_max_ctx >= 4096 else 2048
        chosen = ceiling
        for bucket in CONTEXT_BUCKETS:
            if bucket < min_floor or bucket > ceiling:
                continue
            if bucket >= needed_total:
                chosen = bucket
                break

    single_pass_input = max(800, chosen - reserve_tokens)
    est_vram = 0.0 if not plan.is_local else estimate_vram_gb_for_ctx(profile, chosen)

    return ContextSelection(
        chosen_num_ctx=chosen,
        estimated_vram_gb=est_vram,
        estimated_prompt_tokens=prompt_tokens,
        max_single_pass_input_tokens=single_pass_input,
        needs_splitting=prompt_tokens > single_pass_input,
        vram_plan=plan,
    )


def endpoint_with_ctx(endpoint: ProviderEndpoint, num_ctx: int) -> ProviderEndpoint:
    """Return a shallow copy of ProviderEndpoint tuned to `num_ctx`."""
    return ProviderEndpoint(
        provider_id=endpoint.provider_id,
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        model=endpoint.model,
        num_ctx=num_ctx,
        timeout_seconds=endpoint.timeout_seconds,
        max_retries=endpoint.max_retries,
        auto_ctx=endpoint.auto_ctx,
    )


def _split_atomic_markdown_blocks(body: str) -> list[str]:
    """Split Markdown into atomic blocks without breaking fenced code or $$ math blocks."""
    lines = body.splitlines(keepends=True)
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    in_math = False

    def flush() -> None:
        if current:
            chunk = "".join(current).strip("\r\n")
            if chunk:
                blocks.append(chunk)
            current.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                current.append(line)
                in_fence = False
                flush()
                continue
            flush()
            in_fence = True
            current.append(line)
            continue

        if not in_fence and stripped.startswith("$$"):
            if stripped == "$$" or (stripped.endswith("$$") and len(stripped) > 2):
                if stripped == "$$":
                    if in_math:
                        current.append(line)
                        in_math = False
                        flush()
                        continue
                    flush()
                    in_math = True
                    current.append(line)
                    continue
                flush()
                blocks.append(line.strip("\r\n"))
                continue

        if in_fence or in_math:
            current.append(line)
            continue

        if _HEADING_RE.match(stripped) or _PAGE_MARKER_RE.match(stripped):
            flush()
            blocks.append(stripped)
        elif not stripped:
            flush()
        else:
            current.append(line)

    flush()
    return blocks


def _hard_split_oversized_block(block: str, max_tokens: int) -> list[str]:
    """Split an abnormally long single paragraph/block along newlines or words."""
    if estimate_tokens(block) <= max_tokens:
        return [block]
    sub_parts: list[str] = []
    lines = block.splitlines()
    buf: list[str] = []
    for line in lines:
        candidate = "\n".join([*buf, line]) if buf else line
        if buf and estimate_tokens(candidate) > max_tokens:
            sub_parts.append("\n".join(buf))
            buf = [line]
        elif not buf and estimate_tokens(line) > max_tokens:
            max_chars = max(400, int(max_tokens * 2.8))
            for i in range(0, len(line), max_chars):
                sub_parts.append(line[i : i + max_chars])
        else:
            buf.append(line)
    if buf:
        sub_parts.append("\n".join(buf))
    return sub_parts


def split_markdown_chunks(
    text: str,
    max_chunk_tokens: int,
    *,
    overlap_tokens: int = 120,
) -> list[dict]:
    """Split Markdown into structure-aware chunks respecting headings, pages, code, and paragraphs."""
    clean = (text or "").strip()
    if not clean:
        return []
    safe_limit = max(300, int(max_chunk_tokens))
    raw_blocks = _split_atomic_markdown_blocks(clean)
    blocks: list[str] = []
    for blk in raw_blocks:
        blocks.extend(_hard_split_oversized_block(blk, safe_limit))

    chunks: list[dict] = []
    heading_stack: dict[int, str] = {}
    current_blocks: list[str] = []
    current_heading_path = ""

    def active_heading_label() -> str:
        return " > ".join(heading_stack[lvl] for lvl in sorted(heading_stack))

    def flush_chunk() -> None:
        if not current_blocks:
            return
        chunk_text = "\n\n".join(current_blocks)
        chunks.append(
            {
                "index": len(chunks) + 1,
                "heading_context": current_heading_path,
                "text": chunk_text,
                "tokens": estimate_tokens(chunk_text),
            }
        )
        if overlap_tokens > 0 and len(current_blocks) > 1:
            kept: list[str] = []
            toks = 0
            for blk in reversed(current_blocks):
                bt = estimate_tokens(blk)
                if kept and toks + bt > overlap_tokens:
                    break
                if bt <= overlap_tokens:
                    kept.insert(0, blk)
                    toks += bt
                else:
                    break
            current_blocks[:] = kept
        else:
            current_blocks.clear()

    for blk in blocks:
        m = _HEADING_RE.match(blk)
        if m:
            level = len(m.group(1))
            for deeper in [k for k in heading_stack if k >= level]:
                del heading_stack[deeper]
            heading_stack[level] = blk

        joined = "\n\n".join([*current_blocks, blk]) if current_blocks else blk
        if current_blocks and estimate_tokens(joined) > safe_limit:
            flush_chunk()
            current_heading_path = active_heading_label()
            # Re-check in case overlap + new block exceeds safe_limit
            if current_blocks and estimate_tokens("\n\n".join([*current_blocks, blk])) > safe_limit:
                current_blocks.clear()

        if not current_blocks:
            current_heading_path = active_heading_label()
        current_blocks.append(blk)

    flush_chunk()
    total = len(chunks)
    for item in chunks:
        item["total"] = total
    return chunks


def estimate_model_max_output_tokens(provider_id: str, model: str, chosen_num_ctx: int) -> int:
    """Return conservative per-request output token ceiling for `provider_id`/`model`."""
    spec = get_provider_spec(provider_id)
    lower = (model or "").strip().lower()
    if spec.is_local:
        params_b = parse_model_params_b(model)
        raw_cap = 4096 if params_b <= 4.5 else 8192
        raw_cap = min(raw_cap, max(1024, chosen_num_ctx // 2))
    elif spec.id == "openai":
        raw_cap = 16384 if any(tok in lower for tok in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")) else 4096
    elif spec.id == "anthropic":
        raw_cap = 4096 if any(tok in lower for tok in ("claude-3-opus", "claude-3-sonnet", "claude-3-haiku")) else 8192
    elif spec.id in ("google", "deepseek", "xai", "qwen"):
        raw_cap = 8192
    else:
        raw_cap = 4096
    return max(512, min(raw_cap, max(512, chosen_num_ctx - 512)))


def compute_cleanup_budgets(
    endpoint: ProviderEndpoint,
    body: str,
    *,
    system_prompt_tokens: int = 160,
    system_info: dict | None = None,
) -> CleanupBudget:
    """Compute cleanup slice input budget bounded by both `num_ctx` and model max output tokens."""
    spec = get_provider_spec(endpoint.provider_id)
    body_tokens = estimate_tokens(body)
    cap_ctx = 16384 if spec.is_local else 32768
    est_out = min(8192 if spec.is_local else 16384, max(1024, int(body_tokens * 1.25) + 384))
    selection = choose_adequate_context(
        endpoint,
        body,
        expected_output_tokens=est_out,
        system_info=system_info,
        max_cap_ctx=cap_ctx,
    )
    chosen_ctx = selection.chosen_num_ctx
    max_out = estimate_model_max_output_tokens(endpoint.provider_id, endpoint.model, chosen_ctx)
    ctx_input_limit = int((chosen_ctx - system_prompt_tokens - 240) / 2.25)
    out_input_limit = int((max_out - 256) / 1.25)
    max_slice_input = max(400, min(ctx_input_limit, out_input_limit))
    needs_splitting = body_tokens > max_slice_input

    if endpoint.auto_ctx and needs_splitting:
        min_floor = 4096 if selection.vram_plan.recommended_max_ctx >= 4096 else 2048
        for bucket in CONTEXT_BUCKETS:
            if bucket < min_floor or bucket >= chosen_ctx:
                continue
            b_max_out = estimate_model_max_output_tokens(endpoint.provider_id, endpoint.model, bucket)
            b_ctx_input = int((bucket - system_prompt_tokens - 240) / 2.25)
            if (
                b_max_out >= max_out
                and b_ctx_input >= max_slice_input
                and bucket >= max_slice_input + system_prompt_tokens + max_out + 256
            ):
                chosen_ctx = bucket
                break

    profile = build_model_memory_profile(endpoint.provider_id, endpoint.model)
    est_vram = 0.0 if not selection.vram_plan.is_local else estimate_vram_gb_for_ctx(profile, chosen_ctx)
    return CleanupBudget(
        chosen_num_ctx=chosen_ctx,
        max_output_tokens=max_out,
        max_slice_input_tokens=max_slice_input,
        estimated_vram_gb=est_vram,
        body_tokens=body_tokens,
        needs_splitting=needs_splitting,
        vram_plan=selection.vram_plan,
    )


def _is_ocr_error_only(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return bool(lines) and all(_OCR_ERROR_MARKER_RE.match(line) for line in lines)


def split_markdown_pages_for_cleanup(body: str, max_slice_tokens: int) -> list[dict]:
    """Slice Markdown cleanly along `<!-- Page N -->` boundaries within `max_slice_tokens`.

    Adjacent pages are packed together when they fit within budget so small pages share
    one request, while oversized single pages or non-paginated documents fall back to
    `split_markdown_chunks`.
    """
    clean = (body or "").strip()
    if not clean:
        return []
    safe_limit = max(400, int(max_slice_tokens))

    raw_segments: list[tuple[int | None, str, list[str]]] = []
    cur_page_num: int | None = None
    cur_marker = ""
    cur_lines: list[str] = []
    has_page_markers = False
    uses_hr_separator = False
    in_fence = False

    def trim_blank_edges(lines: list[str]) -> list[str]:
        start = 0
        end = len(lines)
        while start < end and not lines[start].strip():
            start += 1
        while end > start and not lines[end - 1].strip():
            end -= 1
        return lines[start:end]

    for line in clean.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            cur_lines.append(line)
            continue
        if not in_fence:
            marker_match = _PAGE_MARKER_RE.match(stripped)
            if marker_match:
                has_page_markers = True
                if cur_marker or any(item.strip() for item in cur_lines):
                    while cur_lines and not cur_lines[-1].strip():
                        cur_lines.pop()
                    if cur_lines and cur_lines[-1].strip() == "---":
                        cur_lines.pop()
                        uses_hr_separator = True
                        while cur_lines and not cur_lines[-1].strip():
                            cur_lines.pop()
                    trimmed = trim_blank_edges(cur_lines)
                    if cur_marker or trimmed:
                        raw_segments.append((cur_page_num, cur_marker, trimmed))
                cur_page_num = int(marker_match.group(1))
                cur_marker = stripped
                cur_lines = []
                continue
        cur_lines.append(line)

    if not has_page_markers:
        body_tokens = estimate_tokens(clean)
        if body_tokens <= safe_limit:
            return [
                {
                    "index": 1,
                    "total": 1,
                    "page_label": "Page 1",
                    "text": clean,
                    "tokens": body_tokens,
                    "join_with_hr": False,
                    "skip_llm": _is_ocr_error_only(clean),
                    "heading_context": "",
                }
            ]
        chunks = split_markdown_chunks(clean, max_chunk_tokens=safe_limit, overlap_tokens=0)
        return [
            {
                "index": c["index"],
                "total": c["total"],
                "page_label": f"Part {c['index']}/{c['total']}",
                "text": c["text"],
                "tokens": c["tokens"],
                "join_with_hr": False,
                "skip_llm": False,
                "heading_context": c.get("heading_context", ""),
            }
            for c in chunks
        ]

    trimmed_tail = trim_blank_edges(cur_lines)
    if cur_marker or trimmed_tail:
        raw_segments.append((cur_page_num, cur_marker, trimmed_tail))

    page_sep = "\n\n---\n\n" if uses_hr_separator else "\n\n"
    pages: list[dict] = []
    for pnum, marker, lines in raw_segments:
        body_text = "\n".join(lines).strip()
        full_text = f"{marker}\n\n{body_text}" if (marker and body_text) else (marker or body_text)
        pages.append(
            {
                "page_num": pnum,
                "marker": marker,
                "body": body_text,
                "full_text": full_text,
                "skip_llm": bool(marker) and _is_ocr_error_only(body_text),
            }
        )

    slices: list[dict] = []
    cur_group: list[dict] = []

    def _group_label(group: list[dict]) -> str:
        nums = [p["page_num"] for p in group if p["page_num"] is not None]
        if not nums:
            return "Preamble"
        if len(nums) == 1:
            return f"Page {nums[0]}"
        return f"Pages {nums[0]}-{nums[-1]}"

    def _flush_group() -> None:
        if not cur_group:
            return
        joined = page_sep.join(p["full_text"] for p in cur_group)
        slices.append(
            {
                "page_label": _group_label(cur_group),
                "text": joined,
                "tokens": estimate_tokens(joined),
                "join_with_hr": uses_hr_separator and bool(slices),
                "skip_llm": False,
                "heading_context": "",
            }
        )
        cur_group.clear()

    for page in pages:
        if page["skip_llm"]:
            _flush_group()
            label = f"Page {page['page_num']}" if page["page_num"] is not None else "Preamble"
            slices.append(
                {
                    "page_label": label,
                    "text": page["full_text"],
                    "tokens": estimate_tokens(page["full_text"]),
                    "join_with_hr": uses_hr_separator and bool(slices),
                    "skip_llm": True,
                    "heading_context": "",
                }
            )
            continue

        page_tokens = estimate_tokens(page["full_text"])
        if page_tokens > safe_limit:
            _flush_group()
            if page["marker"] and page["body"]:
                marker_tokens = estimate_tokens(page["marker"])
                sub_limit = max(300, safe_limit - marker_tokens - 12)
                sub_chunks = split_markdown_chunks(page["body"], max_chunk_tokens=sub_limit, overlap_tokens=0)
            else:
                sub_chunks = split_markdown_chunks(page["full_text"], max_chunk_tokens=safe_limit, overlap_tokens=0)
            sub_total = len(sub_chunks)
            for j, sub in enumerate(sub_chunks, start=1):
                sub_text = f"{page['marker']}\n\n{sub['text']}" if (j == 1 and page["marker"] and page["body"]) else sub["text"]
                label = (
                    f"Page {page['page_num']} (part {j}/{sub_total})"
                    if page["page_num"] is not None
                    else f"Part {j}/{sub_total}"
                )
                slices.append(
                    {
                        "page_label": label,
                        "text": sub_text,
                        "tokens": estimate_tokens(sub_text),
                        "join_with_hr": j == 1 and uses_hr_separator and bool(slices),
                        "skip_llm": False,
                        "heading_context": sub.get("heading_context", ""),
                    }
                )
            continue

        if cur_group:
            candidate = page_sep.join([*(g["full_text"] for g in cur_group), page["full_text"]])
            if estimate_tokens(candidate) > safe_limit:
                _flush_group()
        cur_group.append(page)

    _flush_group()
    total = len(slices)
    for idx, sl in enumerate(slices, start=1):
        sl["index"] = idx
        sl["total"] = total
    return slices
