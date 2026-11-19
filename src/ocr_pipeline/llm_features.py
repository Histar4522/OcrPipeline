"""LLM-powered artifact intelligence (auto-tagging, renaming, organizing, summarizing, cleanup)."""

from __future__ import annotations

import json
from pathlib import Path
import re

import httpx

from .artifacts import normalize_tags, parse_frontmatter, sanitize_rel_folder, scan_artifacts
from .providers import ProviderEndpoint, chat_text

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json|markdown|md)?\s*\r?\n(.*?)\r?\n```\s*$", re.DOTALL | re.IGNORECASE)
_INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _prepare_doc_excerpt(text: str, max_chars: int = 5300) -> str:
    """Build a compact title + headings + head/tail excerpt suited for small local LLMs."""
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


def suggest_tags(
    text: str,
    known_tags: list[str],
    endpoint: ProviderEndpoint,
    *,
    max_tags: int = 5,
    client: httpx.Client | None = None,
) -> list[str]:
    """Generate 2-5 concise lowercase tags, preferring reuse of existing archive tags."""
    excerpt = _prepare_doc_excerpt(text)
    known_hint = ", ".join(known_tags[:50]) if known_tags else "(none yet)"
    system = (
        "You categorize archived study/work notes. Output strictly valid JSON matching "
        '{"tags": ["tag1", "tag2"]}. Tags must be concise lowercase slugs using letters, digits, '
        "or hyphens (e.g. calculus, organic-chemistry, lecture-notes). Reuse existing archive tags when relevant."
    )
    user = f"Existing archive tags: {known_hint}\nMax tags: {max_tags}\n\nDocument:\n{excerpt}"
    raw = chat_text(endpoint, system, user, json_mode=True, temperature=0.1, max_tokens=256, client=client)
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
    client: httpx.Client | None = None,
) -> str:
    """Suggest a clear, concise filename (.md) derived from document content."""
    excerpt = _prepare_doc_excerpt(text)
    system = (
        "You name archived Markdown documents cleanly. Output strictly valid JSON matching "
        '{"filename": "Descriptive_Title.md"}. Use short descriptive words separated by underscores or hyphens.'
    )
    user = f"Current filename: {current_name}\n\nDocument:\n{excerpt}"
    raw = chat_text(endpoint, system, user, json_mode=True, temperature=0.1, max_tokens=160, client=client)
    data = _parse_json_object(raw)
    return sanitize_suggested_filename(str(data.get("filename") or ""), fallback_name=current_name)


def suggest_folder(
    text: str,
    existing_folders: list[str],
    tags: list[str],
    endpoint: ProviderEndpoint,
    *,
    allow_new_folders: bool = False,
    client: httpx.Client | None = None,
) -> dict:
    """Suggest an archive folder path and short explanation."""
    excerpt = _prepare_doc_excerpt(text)
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
        f"Document:\n{excerpt}"
    )
    raw = chat_text(endpoint, system, user, json_mode=True, temperature=0.1, max_tokens=220, client=client)
    data = _parse_json_object(raw)
    cand = sanitize_rel_folder(str(data.get("folder") or "")).as_posix()
    if cand == ".":
        cand = ""
    if not allow_new_folders and cand and cand not in existing_folders:
        # Case-insensitive match fallback to an existing folder
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
    client: httpx.Client | None = None,
) -> str:
    """Produce a concise 1-3 sentence archival summary of the document."""
    excerpt = _prepare_doc_excerpt(text)
    system = (
        "Summarize the archived note accurately in 1 to 3 concise sentences. Output strictly valid JSON matching "
        '{"summary": "..."}. Do not invent facts.'
    )
    raw = chat_text(endpoint, system, f"Document:\n{excerpt}", json_mode=True, temperature=0.1, max_tokens=300, client=client)
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
    client: httpx.Client | None = None,
) -> dict:
    """Compute tags, filename, folder, and summary in a single LLM request."""
    excerpt = _prepare_doc_excerpt(text)
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
        '"folder": "ExistingOrEmpty", "summary": "1-2 concise sentences.", "reason": "folder choice reason"}.\n'
        f"Tags: 2-{max_tags} concise lowercase slugs reusing Existing tags when relevant. "
        f"Folder rule: {folder_rule}"
    )
    user = (
        f"Current filename: {current_name}\n"
        f"Existing tags: {known_hint}\n"
        f"Existing folders: {folder_list}\n\n"
        f"Document:\n{excerpt}"
    )
    raw = chat_text(endpoint, system, user, json_mode=True, temperature=0.1, max_tokens=512, client=client)
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
    client: httpx.Client | None = None,
) -> str:
    """Fix OCR typos, broken line wraps, and Markdown/LaTeX formatting without omitting content."""
    meta, body = parse_frontmatter(text)
    system = (
        "You are an archival Markdown proofreader. Clean up obvious OCR typos, broken line breaks mid-sentence, "
        "malformed Markdown tables, and broken LaTeX delimiters in the document body. "
        "Preserve EVERY section, fact, number, equation, `[Figure: ...]` placeholder, and `<!-- Page N -->` marker. "
        "Do NOT summarize, omit content, or add commentary. Output ONLY the cleaned Markdown body."
    )
    max_tokens = max(2048, min(8192, len(body) // 2 + 1024))
    cleaned_body = chat_text(
        endpoint,
        system,
        body.strip(),
        json_mode=False,
        temperature=0.0,
        max_tokens=max_tokens,
        client=client,
    )
    cleaned_body = _strip_think_and_fences(cleaned_body)
    if meta:
        from .artifacts import format_frontmatter

        return format_frontmatter(meta, cleaned_body)
    return cleaned_body + ("\n" if cleaned_body and not cleaned_body.endswith("\n") else "")


def plan_folder_organization(
    archive_dir: Path,
    cwd: str,
    endpoint: ProviderEndpoint,
    *,
    allow_new_folders: bool = False,
    limit: int = 30,
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
