"""LLM-powered artifact intelligence (auto-tagging, renaming, organizing, summarizing, cleanup).

Optimizes inference for large documents via hardware VRAM budgeting, dynamic
`num_ctx` selection, hierarchical map-reduce summarization, and lossless chunked
OCR proofreading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re

import httpx

from .artifacts import format_frontmatter, normalize_tags, parse_frontmatter, sanitize_rel_folder, scan_artifacts
from .llm_context import (
    choose_adequate_context,
    endpoint_with_ctx,
    estimate_tokens,
    split_markdown_chunks,
)
from .providers import ProviderEndpoint, chat_text

logger = logging.getLogger("ocr_pipeline")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json|markdown|md)?\s*\r?\n(.*?)\r?\n```\s*$", re.DOTALL | re.IGNORECASE)
_INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _prepare_doc_excerpt(text: str, max_chars: int = 5300) -> str:
    """Build a compact title + headings + head/tail excerpt."""
    meta, body = parse_frontmatter(text)
    clean_body = body.strip()
    title = str(meta.get("title") or "").strip()
    headings = [line.strip() for line in clean_body.splitlines() if line.lstrip().startswith("#")][:15]

    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if headings:
        parts.append("Headings:\n" + "\n".join(headings))

    if len(clean_body) <= max_chars:
        parts.append("Content:\n" + clean_body)
    else:
        head = clean_body[:4300]
        tail = clean_body[-800:]
        parts.append(f"Content (excerpt):\n{head}\n\n[...]\n\n{tail}")
    return "\n\n".join(parts)


def _strip_think_and_fences(raw: str) -> str:
    cleaned = _THINK_RE.sub("", raw or "").strip()
    fenced = _FENCE_RE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    return cleaned


def _parse_json_object(raw: str) -> dict:
    """Parse JSON object defensively from small local or cloud model outputs."""
    cleaned = _strip_think_and_fences(raw)
    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            return loaded
    except ValueError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            loaded = json.loads(cleaned[start : end + 1])
            if isinstance(loaded, dict):
                return loaded
        except ValueError:
            pass
    raise ValueError(f"Expected JSON object from LLM, got: {cleaned[:160]}")


def sanitize_suggested_filename(candidate: str, fallback_name: str = "document.md") -> str:
    """Sanitize an LLM-suggested filename for Windows/macOS/Linux filesystems."""
    raw = str(candidate or "").strip().strip("\"'")
    if raw.lower().endswith(".md"):
        raw = raw[:-3]
    clean = _INVALID_FILENAME_RE.sub("_", raw)
    clean = re.sub(r"[\s_]+", "_", clean).strip(" ._")
    if not clean or clean in (".", ".."):
        return fallback_name if fallback_name.lower().endswith(".md") else f"{fallback_name}.md"
    return f"{clean[:80]}.md"


def _summarize_chunk_map(
    chunk: dict,
    endpoint: ProviderEndpoint,
    *,
    client: httpx.Client | None = None,
) -> str:
    """Summarize one structural Markdown chunk cleanly in the Map step."""
    heading_note = f"Section path: {chunk['heading_context']}\n" if chunk.get("heading_context") else ""
    system = (
        "Summarize this excerpt from a longer archived document in 2 to 4 concise factual sentences. "
        "Preserve key topics, proper nouns, definitions, equations, and outcomes. "
        "Output plain prose only without introductory filler."
    )
    user = f"Excerpt {chunk['index']} of {chunk['total']}:\n{heading_note}\n{chunk['text']}"
    raw = chat_text(
        endpoint,
        system,
        user,
        json_mode=False,
        temperature=0.1,
        max_tokens=320,
        client=client,
    )
    cleaned = " ".join(_strip_think_and_fences(raw).split())
    prefix = f"Section {chunk['index']}/{chunk['total']}"
    if chunk.get("heading_context"):
        prefix += f" ({chunk['heading_context']})"
    return f"{prefix}: {cleaned}" if cleaned else f"{prefix}: {chunk['text'][:400]}"


def _reduce_summary_blocks(
    blocks: list[str],
    reduce_endpoint: ProviderEndpoint,
    target_tokens: int,
    *,
    client: httpx.Client | None = None,
) -> list[str]:
    """Recursively group and condense map summaries if dozens of chunks exceed reduce context."""
    current = list(blocks)
    system = (
        "Condense these sequential section summaries into a shorter cohesive summary of 3 to 6 factual sentences "
        "preserving all major subjects, technical terms, and conclusions. Output plain prose only."
    )
    for _depth in range(4):
        combined = "\n\n".join(current)
        if estimate_tokens(combined) <= target_tokens or len(current) <= 2:
            return current
        next_level: list[str] = []
        batch: list[str] = []
        group_limit = max(600, int(target_tokens * 0.72))
        for item in current:
            candidate = "\n\n".join([*batch, item]) if batch else item
            if batch and estimate_tokens(candidate) > group_limit:
                merged_raw = chat_text(
                    reduce_endpoint,
                    system,
                    "\n\n".join(batch),
                    json_mode=False,
                    temperature=0.1,
                    max_tokens=360,
                    client=client,
                )
                merged = " ".join(_strip_think_and_fences(merged_raw).split())
                next_level.append(f"Part {len(next_level) + 1}: {merged}" if merged else batch[0])
                batch = [item]
            else:
                batch.append(item)
        if batch:
            merged_raw = chat_text(
                reduce_endpoint,
                system,
                "\n\n".join(batch),
                json_mode=False,
                temperature=0.1,
                max_tokens=360,
                client=client,
            )
            merged = " ".join(_strip_think_and_fences(merged_raw).split())
            next_level.append(f"Part {len(next_level) + 1}: {merged}" if merged else batch[0])
        current = next_level
    return current


def _prepare_doc_for_llm(
    text: str,
    endpoint: ProviderEndpoint,
    *,
    expected_output_tokens: int = 512,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> tuple[str, ProviderEndpoint, dict]:
    """Prepare full document or hierarchical map-reduce digest sized for hardware VRAM.

    When the full document fits within the safe context window for the detected VRAM/RAM
    budget, the entire document is passed untouched. Otherwise, the document is split into
    structural chunks and summarized hierarchically before running the downstream task.
    """
    meta, body = parse_frontmatter(text)
    clean_body = body.strip()
    title = str(meta.get("title") or "").strip()
    headings = [line.strip() for line in clean_body.splitlines() if line.lstrip().startswith("#")][:25]

    header_parts: list[str] = []
    if title:
        header_parts.append(f"Title: {title}")
    if headings:
        header_parts.append("Headings:\n" + "\n".join(headings))
    header_block = "\n\n".join(header_parts)
    full_doc = f"{header_block}\n\nContent:\n{clean_body}" if header_block else f"Content:\n{clean_body}"

    selection = choose_adequate_context(
        endpoint,
        full_doc,
        expected_output_tokens=expected_output_tokens,
        system_info=system_info,
    )

    if not selection.needs_splitting:
        tuned = endpoint_with_ctx(endpoint, selection.chosen_num_ctx)
        return (
            full_doc,
            tuned,
            {
                "chunked": False,
                "chunks": 1,
                "chosen_num_ctx": selection.chosen_num_ctx,
                "estimated_vram_gb": selection.estimated_vram_gb,
            },
        )

    # Large document: split along Markdown boundaries with a fast KV-friendly chunk window,
    # summarize each section, and hierarchically reduce if needed.
    max_safe = selection.vram_plan.recommended_max_ctx if endpoint.auto_ctx else selection.chosen_num_ctx
    map_ctx = min(max_safe, 8192) if endpoint.auto_ctx else selection.chosen_num_ctx
    map_endpoint = endpoint_with_ctx(endpoint, map_ctx)
    chunk_budget = max(700, min(int(map_ctx * 0.58), max(700, selection.max_single_pass_input_tokens // 2)))

    chunks = split_markdown_chunks(clean_body, max_chunk_tokens=chunk_budget, overlap_tokens=100)
    logger.info(
        "LLM splitting large document (~%d tokens) into %d chunks (map num_ctx=%d, max safe ctx=%d, vram=%.2f GB)",
        selection.estimated_prompt_tokens,
        len(chunks),
        map_ctx,
        selection.vram_plan.recommended_max_ctx,
        selection.vram_plan.vram_at_recommended_gb,
    )

    chunk_summaries = [_summarize_chunk_map(c, map_endpoint, client=client) for c in chunks]

    # Pick reduce context sized to hold the condensed summaries (+ headings) comfortably in VRAM.
    draft_digest = "\n\n".join(filter(None, [header_block, "Section Summaries:\n" + "\n\n".join(chunk_summaries)]))
    reduce_sel = choose_adequate_context(
        endpoint,
        draft_digest,
        expected_output_tokens=expected_output_tokens,
        system_info=system_info,
    )
    reduce_endpoint = endpoint_with_ctx(endpoint, reduce_sel.chosen_num_ctx)

    header_tokens = estimate_tokens(header_block) + 80
    reduce_body_budget = max(500, reduce_sel.max_single_pass_input_tokens - header_tokens)
    reduced_sections = _reduce_summary_blocks(chunk_summaries, reduce_endpoint, reduce_body_budget, client=client)

    condensed_doc = "\n\n".join(
        filter(None, [header_block, "Condensed Full-Document Sections:\n" + "\n\n".join(reduced_sections)])
    )
    # Re-tune final reduce context to exact condensed digest size
    final_sel = choose_adequate_context(
        endpoint,
        condensed_doc,
        expected_output_tokens=expected_output_tokens,
        system_info=system_info,
    )
    return (
        condensed_doc,
        endpoint_with_ctx(endpoint, final_sel.chosen_num_ctx),
        {
            "chunked": True,
            "chunks": len(chunks),
            "chosen_num_ctx": final_sel.chosen_num_ctx,
            "estimated_vram_gb": final_sel.estimated_vram_gb,
        },
    )


def suggest_tags(
    text: str,
    known_tags: list[str],
    endpoint: ProviderEndpoint,
    *,
    max_tags: int = 5,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> list[str]:
    """Generate 2-5 concise lowercase tags, preferring reuse of existing archive tags."""
    doc_payload, tuned_ep, _ = _prepare_doc_for_llm(
        text, endpoint, expected_output_tokens=256, system_info=system_info, client=client
    )
    known_hint = ", ".join(known_tags[:50]) if known_tags else "(none yet)"
    system = (
        "You categorize archived study/work notes. Output strictly valid JSON matching "
        '{"tags": ["tag1", "tag2"]}. Tags must be concise lowercase slugs using letters, digits, '
        "or hyphens (e.g. calculus, organic-chemistry, lecture-notes). Reuse existing archive tags when relevant."
    )
    user = f"Existing archive tags: {known_hint}\nMax tags: {max_tags}\n\nDocument:\n{doc_payload}"
    raw = chat_text(tuned_ep, system, user, json_mode=True, temperature=0.1, max_tokens=256, client=client)
    data = _parse_json_object(raw)
    raw_tags = data.get("tags") or []
    if isinstance(raw_tags, list):
        raw_tags = [str(t).lower().replace(" ", "-") for t in raw_tags]
    return normalize_tags(raw_tags)[:max_tags]


def suggest_filename(
    text: str,
    current_name: str,
    endpoint: ProviderEndpoint,
    *,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Suggest a clear, concise filename (.md) derived from document content."""
    doc_payload, tuned_ep, _ = _prepare_doc_for_llm(
        text, endpoint, expected_output_tokens=160, system_info=system_info, client=client
    )
    system = (
        "You name archived Markdown documents cleanly. Output strictly valid JSON matching "
        '{"filename": "Descriptive_Title.md"}. Use short descriptive words separated by underscores or hyphens.'
    )
    user = f"Current filename: {current_name}\n\nDocument:\n{doc_payload}"
    raw = chat_text(tuned_ep, system, user, json_mode=True, temperature=0.1, max_tokens=160, client=client)
    data = _parse_json_object(raw)
    return sanitize_suggested_filename(str(data.get("filename") or ""), fallback_name=current_name)


def suggest_folder(
    text: str,
    existing_folders: list[str],
    tags: list[str],
    endpoint: ProviderEndpoint,
    *,
    allow_new_folders: bool = False,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """Suggest an archive folder path and short explanation."""
    doc_payload, tuned_ep, _ = _prepare_doc_for_llm(
        text, endpoint, expected_output_tokens=220, system_info=system_info, client=client
    )
    folder_list = ", ".join(existing_folders) if existing_folders else "(no folders yet — root only)"
    rule = (
        "You may choose one of the existing folders, or propose a short new subfolder path (e.g. 'Subject/Topic'), or '' for archive root."
        if allow_new_folders
        else "Pick the single best matching folder from existing folders, or '' if none fit well. Do NOT invent new folders."
    )
    system = (
        "You organize Markdown documents into folders. Output strictly valid JSON matching "
        '{"folder": "Folder/Path", "reason": "short 1-sentence reason"}. '
        + rule
    )
    user = (
        f"Existing folders: {folder_list}\n"
        f"Tags: {', '.join(tags) if tags else '(none)'}\n\n"
        f"Document:\n{doc_payload}"
    )
    raw = chat_text(tuned_ep, system, user, json_mode=True, temperature=0.1, max_tokens=220, client=client)
    data = _parse_json_object(raw)
    cand = sanitize_rel_folder(str(data.get("folder") or "")).as_posix()
    if cand == ".":
        cand = ""
    if not allow_new_folders and cand and cand not in existing_folders:
        matched = next((f for f in existing_folders if f.lower() == cand.lower()), "")
        cand = matched
    return {
        "folder": cand,
        "reason": str(data.get("reason") or "").strip(),
        "is_new": bool(cand and cand not in existing_folders),
    }


def summarize_document(
    text: str,
    endpoint: ProviderEndpoint,
    *,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Produce a concise archival summary, using hierarchical split-and-summarize on long docs."""
    doc_payload, tuned_ep, stats = _prepare_doc_for_llm(
        text, endpoint, expected_output_tokens=360, system_info=system_info, client=client
    )
    if stats["chunked"]:
        system = (
            "Synthesize these condensed section summaries of a long archived document into one cohesive "
            '1 to 4 sentence summary. Output strictly valid JSON matching {"summary": "..."}. Do not invent facts.'
        )
    else:
        system = (
            "Summarize the archived note accurately in 1 to 3 concise sentences. Output strictly valid JSON matching "
            '{"summary": "..."}. Do not invent facts.'
        )
    raw = chat_text(tuned_ep, system, f"Document:\n{doc_payload}", json_mode=True, temperature=0.1, max_tokens=360, client=client)
    data = _parse_json_object(raw)
    return " ".join(str(data.get("summary") or "").strip().split())


def analyze_artifact_all(
    text: str,
    *,
    current_name: str,
    known_tags: list[str],
    existing_folders: list[str],
    endpoint: ProviderEndpoint,
    allow_new_folders: bool = False,
    max_tags: int = 5,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """Compute tags, filename, folder, and summary via one shared digest (and map-reduce if large)."""
    doc_payload, tuned_ep, _ = _prepare_doc_for_llm(
        text, endpoint, expected_output_tokens=512, system_info=system_info, client=client
    )
    folder_list = ", ".join(existing_folders) if existing_folders else "(no folders yet)"
    known_hint = ", ".join(known_tags[:50]) if known_tags else "(none yet)"
    folder_rule = (
        "Use an existing folder, a short new subfolder path, or '' for root."
        if allow_new_folders
        else "Pick from Existing folders only, or '' for root. Do NOT invent new folders."
    )
    system = (
        "Analyze this archived Markdown document and return strictly valid JSON matching:\n"
        '{"tags": ["slug1", "slug2"], "filename": "Descriptive_Name.md", '
        '"folder": "ExistingOrEmpty", "summary": "1-3 concise sentences.", "reason": "folder choice reason"}.\n'
        f"Tags: 2-{max_tags} concise lowercase slugs reusing Existing tags when relevant. "
        f"Folder rule: {folder_rule}"
    )
    user = (
        f"Current filename: {current_name}\n"
        f"Existing tags: {known_hint}\n"
        f"Existing folders: {folder_list}\n\n"
        f"Document:\n{doc_payload}"
    )
    raw = chat_text(tuned_ep, system, user, json_mode=True, temperature=0.1, max_tokens=512, client=client)
    data = _parse_json_object(raw)

    raw_tags = data.get("tags") or []
    if isinstance(raw_tags, list):
        raw_tags = [str(t).lower().replace(" ", "-") for t in raw_tags]
    tags = normalize_tags(raw_tags)[:max_tags]

    filename = sanitize_suggested_filename(str(data.get("filename") or ""), fallback_name=current_name)

    cand_folder = sanitize_rel_folder(str(data.get("folder") or "")).as_posix()
    if cand_folder == ".":
        cand_folder = ""
    if not allow_new_folders and cand_folder and cand_folder not in existing_folders:
        cand_folder = next((f for f in existing_folders if f.lower() == cand_folder.lower()), "")

    summary = " ".join(str(data.get("summary") or "").strip().split())
    return {
        "tags": tags,
        "filename": filename,
        "folder": cand_folder,
        "is_new_folder": bool(cand_folder and cand_folder not in existing_folders),
        "summary": summary,
        "reason": str(data.get("reason") or "").strip(),
    }


def cleanup_ocr_markdown(
    text: str,
    endpoint: ProviderEndpoint,
    *,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Fix OCR typos, broken line wraps, and Markdown/LaTeX formatting without losing content.

    Because cleanup outputs roughly 1 token per input token, large documents are split
    losslessly along structural boundaries and proofread chunk-by-chunk within VRAM limits.
    """
    meta, body = parse_frontmatter(text)
    clean_body = body.strip()
    if not clean_body:
        return format_frontmatter(meta, "") if meta else ""

    body_tokens = estimate_tokens(clean_body)
    # 1:1 input+output fits best with a moderate ceiling so small local models stay crisp and fast.
    selection = choose_adequate_context(
        endpoint,
        clean_body,
        expected_output_tokens=min(4096, max(1024, body_tokens + 256)),
        system_info=system_info,
        max_cap_ctx=16384,
    )
    tuned_ep = endpoint_with_ctx(endpoint, selection.chosen_num_ctx)
    # Leave room for system prompt + ~1:1 output tokens inside chosen_num_ctx.
    safe_chunk_tokens = max(500, min(int(selection.chosen_num_ctx * 0.36), 2800))

    system = (
        "You are an archival Markdown proofreader. Clean up obvious OCR typos, broken line breaks mid-sentence, "
        "malformed Markdown tables, and broken LaTeX delimiters in the document body. "
        "Preserve EVERY section, fact, number, equation, `[Figure: ...]` placeholder, and `<!-- Page N -->` marker. "
        "Do NOT summarize, omit content, or add commentary. Output ONLY the cleaned Markdown text."
    )

    if body_tokens <= safe_chunk_tokens:
        max_out = max(1024, min(8192, int(body_tokens * 1.4) + 512))
        cleaned_body = chat_text(
            tuned_ep,
            system,
            clean_body,
            json_mode=False,
            temperature=0.0,
            max_tokens=max_out,
            client=client,
        )
        cleaned_body = _strip_think_and_fences(cleaned_body) or clean_body
    else:
        chunks = split_markdown_chunks(clean_body, max_chunk_tokens=safe_chunk_tokens, overlap_tokens=0)
        logger.info(
            "OCR cleanup splitting document (~%d tokens) into %d lossless chunks (num_ctx=%d)",
            body_tokens,
            len(chunks),
            tuned_ep.num_ctx,
        )
        cleaned_parts: list[str] = []
        for chunk in chunks:
            heading_hint = (
                f"(Context section for reference only — do not repeat unless already in the excerpt: {chunk['heading_context']})\n\n"
                if chunk.get("heading_context")
                else ""
            )
            max_out = max(1024, min(6144, int(chunk["tokens"] * 1.45) + 400))
            out = chat_text(
                tuned_ep,
                system,
                f"{heading_hint}{chunk['text']}",
                json_mode=False,
                temperature=0.0,
                max_tokens=max_out,
                client=client,
            )
            cleaned_piece = _strip_think_and_fences(out)
            cleaned_parts.append(cleaned_piece if cleaned_piece else chunk["text"])
        cleaned_body = "\n\n".join(cleaned_parts)

    if meta:
        return format_frontmatter(meta, cleaned_body)
    return cleaned_body + ("\n" if cleaned_body and not cleaned_body.endswith("\n") else "")


def plan_folder_organization(
    archive_dir: Path,
    cwd: str,
    endpoint: ProviderEndpoint,
    *,
    allow_new_folders: bool = False,
    limit: int = 30,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> list[dict]:
    """Generate a dry-run rename/move/tag plan for artifacts inside folder `cwd`."""
    index = scan_artifacts(archive_dir, cwd=cwd, sort="name")
    existing_folders: list[str] = list(index.get("folders") or [])
    known_tags: list[str] = [str(t["name"]) for t in (index.get("tags") or [])]
    root = archive_dir.expanduser().resolve()

    plan: list[dict] = []
    for item in (index.get("artifacts") or [])[:limit]:
        rel_path = str(item["path"])
        target = root / rel_path
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        result = analyze_artifact_all(
            text,
            current_name=item["name"],
            known_tags=known_tags,
            existing_folders=existing_folders,
            endpoint=endpoint,
            allow_new_folders=allow_new_folders,
            system_info=system_info,
            client=client,
        )
        # Feed newly proposed tags forward within this batch for consistency
        for t in result["tags"]:
            if t not in known_tags:
                known_tags.append(t)
        plan.append(
            {
                "path": rel_path,
                "current_folder": item["folder"],
                "current_name": item["name"],
                "suggested_folder": result["folder"],
                "suggested_name": result["filename"],
                "suggested_tags": result["tags"],
                "summary": result["summary"],
                "reason": result["reason"],
            }
        )
    return plan
