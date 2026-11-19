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

from .artifacts import (
    format_frontmatter,
    normalize_tags,
    parse_frontmatter,
    sanitize_rel_folder,
    scan_artifacts,
    update_artifact_summary,
    write_folder_meta,
)
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
_UUID_OR_HEX_RE = re.compile(r"^(?:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|[0-9a-f]{12,})$", re.IGNORECASE)


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


def _is_unhelpful_filename(candidate: str, current_name: str) -> bool:
    stem = Path(candidate).stem.strip().lower()
    if not stem or stem in ("descriptive_title", "descriptive_name", "document", "untitled", "title"):
        return True
    if candidate.strip().lower() == current_name.strip().lower():
        return True
    return bool(_UUID_OR_HEX_RE.match(stem))


def suggest_tags(
    text: str,
    known_tags: list[str],
    endpoint: ProviderEndpoint,
    *,
    max_tags: int = 5,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> list[str]:
    """Generate 2-5 concise lowercase tags grounded strictly in this document's content."""
    doc_payload, tuned_ep, _ = _prepare_doc_for_llm(
        text, endpoint, expected_output_tokens=256, system_info=system_info, client=client
    )
    system = (
        "Extract concise lowercase hyphenated topic tags describing what THIS specific document is about. "
        f'Output strictly valid JSON matching {{"tags": ["topic-one", "topic-two"]}} (2 to {max_tags} tags). '
        "Base tags strictly on THIS document's subject matter. "
        "Only reuse an optional archive tag if this document genuinely covers that exact subject."
    )
    user = f"Document:\n{doc_payload}"
    if known_tags:
        user += f"\n\nOptional existing archive tags (use ONLY if accurately relevant to this document): {', '.join(known_tags[:30])}"
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
        "Generate a short, descriptive Markdown filename (.md) capturing the main topic of the document below. "
        'Output strictly valid JSON matching {"filename": "Descriptive_Topic_Title.md"}. '
        "Use 2 to 6 Title_Case words separated by underscores. Never output UUIDs or hex hashes."
    )
    user = f"Document:\n{doc_payload}"
    raw = chat_text(tuned_ep, system, user, json_mode=True, temperature=0.1, max_tokens=160, client=client)
    data = _parse_json_object(raw)
    cand_raw = str(
        data.get("filename")
        or data.get("new_filename")
        or data.get("suggested_name")
        or data.get("name")
        or data.get("title")
        or ""
    ).strip()
    candidate = sanitize_suggested_filename(cand_raw, fallback_name="") if cand_raw else ""

    if _is_unhelpful_filename(candidate, current_name):
        retry_sys = (
            "Write a short 2 to 5 word topic title describing the text below. "
            'Output strictly valid JSON matching {"title": "Short Descriptive Title"}.'
        )
        retry_raw = chat_text(
            tuned_ep, retry_sys, f"Document:\n{doc_payload}", json_mode=True, temperature=0.3, max_tokens=120, client=client
        )
        try:
            retry_data = _parse_json_object(retry_raw)
            cand2 = str(retry_data.get("title") or retry_data.get("filename") or "").strip()
        except ValueError:
            cand2 = _strip_think_and_fences(retry_raw).splitlines()[0].strip()
        cleaned2 = sanitize_suggested_filename(cand2, fallback_name=current_name) if cand2 else ""
        if cleaned2 and not _is_unhelpful_filename(cleaned2, current_name):
            return cleaned2

    return candidate or (current_name if current_name.lower().endswith(".md") else f"{current_name}.md")


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
    existing_folders: list[str] | None = None,
    endpoint: ProviderEndpoint,
    allow_new_folders: bool = False,
    max_tags: int = 5,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """Compute tags, descriptive filename, and summary via one shared digest (and map-reduce if large)."""
    doc_payload, tuned_ep, _ = _prepare_doc_for_llm(
        text, endpoint, expected_output_tokens=512, system_info=system_info, client=client
    )
    system = (
        "Analyze this archived Markdown document and return strictly valid JSON matching:\n"
        '{"tags": ["slug1", "slug2"], "filename": "Descriptive_Topic_Title.md", '
        '"summary": "1-3 concise sentences."}.\n'
        f"Tags: 2-{max_tags} concise lowercase hyphenated slugs matching THIS document's topic "
        "(reuse an optional archive tag ONLY if truly relevant). "
        "Filename: 2-6 Title_Case words with underscores ending in .md (never a UUID or random string)."
    )
    user = f"Document:\n{doc_payload}"
    if known_tags:
        user += f"\n\nOptional existing archive tags (reuse ONLY if accurately relevant): {', '.join(known_tags[:30])}"
    raw = chat_text(tuned_ep, system, user, json_mode=True, temperature=0.1, max_tokens=512, client=client)
    data = _parse_json_object(raw)

    raw_tags = data.get("tags") or []
    if isinstance(raw_tags, list):
        raw_tags = [str(t).lower().replace(" ", "-") for t in raw_tags]
    tags = normalize_tags(raw_tags)[:max_tags]

    raw_fn = str(data.get("filename") or data.get("title") or "").strip()
    filename = sanitize_suggested_filename(raw_fn, fallback_name=current_name) if raw_fn else current_name
    summary = " ".join(str(data.get("summary") or "").strip().split())
    return {
        "tags": tags,
        "filename": filename,
        "summary": summary,
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


def summarize_folder_contents(
    archive_dir: Path,
    cwd: str,
    endpoint: ProviderEndpoint,
    *,
    limit: int = 50,
    system_info: dict | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """Synthesize a Markdown overview of all documents inside folder `cwd` and save `.folder.meta`.

    Reuses saved `<file.md>.meta` summaries where available and generates missing ones cleanly.
    """
    clean_rel = sanitize_rel_folder(cwd or "")
    clean_cwd = clean_rel.as_posix() if clean_rel.parts else ""
    index = scan_artifacts(archive_dir, folder=clean_cwd, sort="name")
    artifacts = list(index.get("artifacts") or [])
    if not artifacts:
        return {"folder": clean_cwd, "doc_count": 0, "summary": "", "updated_at": ""}

    root = archive_dir.expanduser().resolve()
    entries: list[str] = []
    for item in artifacts[:limit]:
        rel_path = str(item["path"])
        target = root / rel_path
        if not target.is_file():
            continue
        doc_summary = str(item.get("summary") or "").strip()
        if not doc_summary:
            raw_text = target.read_text(encoding="utf-8", errors="replace")
            doc_summary = summarize_document(raw_text, endpoint, system_info=system_info, client=client)
            if doc_summary:
                update_artifact_summary(target, doc_summary)
        tag_str = f" [tags: {', '.join(item['tags'][:5])}]" if item.get("tags") else ""
        snippet = doc_summary or str(item.get("preview") or "").strip()[:180]
        entries.append(f"- **{rel_path}**{tag_str}: {snippet}")

    combined = "\n".join(entries)
    selection = choose_adequate_context(
        endpoint,
        combined,
        expected_output_tokens=750,
        system_info=system_info,
    )
    tuned_ep = endpoint_with_ctx(endpoint, selection.chosen_num_ctx)
    if selection.needs_splitting:
        condensed = _reduce_summary_blocks(
            entries, tuned_ep, max(600, selection.max_single_pass_input_tokens), client=client
        )
        combined = "\n\n".join(condensed)

    folder_label = clean_cwd or "Archive Root"
    system = (
        "Synthesize the document summaries inside this archive folder into a clean, useful Markdown overview. "
        "Include concise sections:\n"
        "### Overview\n(2-4 sentences explaining what this folder contains as a whole)\n"
        "### Key Topics\n(bullet points of main subjects covered)\n"
        "### Documents Included\n(brief bullets citing filenames and core takeaways)\n"
        "Stay strictly factual and grounded in the summaries provided."
    )
    user = f"Folder: {folder_label} ({len(entries)} document(s))\n\nDocument summaries:\n{combined}"
    raw = chat_text(tuned_ep, system, user, json_mode=False, temperature=0.15, max_tokens=800, client=client)
    summary_md = _strip_think_and_fences(raw).strip()
    saved = write_folder_meta(archive_dir, clean_cwd, summary_md, doc_count=len(artifacts))
    return saved
