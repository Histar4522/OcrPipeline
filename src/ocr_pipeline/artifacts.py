"""Artifact indexing, YAML frontmatter tagging, folder management, and Obsidian vault helpers."""

from datetime import datetime
import json
from pathlib import Path
import re
import shutil

_INLINE_TAG_RE = re.compile(r"(?<!\S)#([A-Za-z][A-Za-z0-9_/-]{0,48})")
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse minimal YAML frontmatter from Markdown text and return (metadata, body)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    raw_yaml, body = match.group(1), match.group(2)
    meta: dict[str, object] = {}
    current_list_key: str | None = None

    for line in raw_yaml.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list_key is not None:
            val = stripped[2:].strip().strip("\"'")
            existing = meta.setdefault(current_list_key, [])
            if isinstance(existing, list) and val:
                existing.append(val)
            continue

        current_list_key = None
        if ":" not in line:
            continue
        key, raw_val = line.split(":", 1)
        key = key.strip()
        val = raw_val.strip()
        if val == "":
            meta[key] = []
            current_list_key = key
        elif val.startswith("[") and val.endswith("]"):
            items = [item.strip().strip("\"'") for item in val[1:-1].split(",")]
            meta[key] = [item for item in items if item]
        elif val.lower() in ("true", "false"):
            meta[key] = val.lower() == "true"
        else:
            meta[key] = val.strip("\"'")

    return meta, body


def format_frontmatter(meta: dict[str, object], body: str) -> str:
    """Serialize metadata as YAML frontmatter prepended to Markdown body."""
    if not meta:
        return body
    lines = ["---"]
    for key, val in meta.items():
        if isinstance(val, list):
            if not val:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}:")
                for item in val:
                    lines.append(f"  - {item}")
        elif isinstance(val, bool):
            lines.append(f"{key}: {'true' if val else 'false'}")
        else:
            escaped = str(val).replace('"', '\\"')
            lines.append(f'{key}: "{escaped}"')
    lines.append("---")
    clean_body = body.lstrip("\r\n")
    return "\n".join(lines) + ("\n\n" + clean_body if clean_body else "\n")


def normalize_tags(raw_tags: str | list[str]) -> list[str]:
    """Normalize comma/space-separated tags into unique lowercase slugs without leading #."""
    items = raw_tags.split(",") if isinstance(raw_tags, str) else raw_tags
    seen: list[str] = []
    for item in items:
        clean = item.strip().lstrip("#").strip()
        if clean and clean not in seen:
            seen.append(clean)
    return seen


def extract_tags(text: str) -> list[str]:
    """Return unique tags from YAML frontmatter, or inline #tags when frontmatter has no tags key."""
    meta, body = parse_frontmatter(text)
    if "tags" in meta:
        fm_tags = meta.get("tags")
        found: list[str] = []
        if isinstance(fm_tags, list):
            for t in fm_tags:
                val = str(t).strip().lstrip("#")
                if val and val not in found:
                    found.append(val)
        elif isinstance(fm_tags, str) and fm_tags.strip():
            for t in normalize_tags(fm_tags):
                if t not in found:
                    found.append(t)
        return found

    found = []
    for match in _INLINE_TAG_RE.finditer(body):
        tag = match.group(1)
        if tag not in found:
            found.append(tag)
    return found


def update_artifact_tags(file_path: Path, tags: list[str]) -> list[str]:
    """Update the YAML frontmatter tags of a Markdown artifact on disk."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    meta, body = parse_frontmatter(text)
    cleaned = normalize_tags(tags)
    if cleaned:
        meta["tags"] = cleaned
    elif _INLINE_TAG_RE.search(body):
        # Keep explicit empty frontmatter list so inline #tags in body don't reappear after clearing
        meta["tags"] = []
    else:
        meta.pop("tags", None)
    file_path.write_text(format_frontmatter(meta, body), encoding="utf-8")
    return cleaned


def ensure_obsidian_vault(archive_dir: Path) -> Path:
    """Initialize minimal Obsidian vault configuration inside archive_dir."""
    root = archive_dir.expanduser().resolve()
    obsidian_dir = root / ".obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    app_json = obsidian_dir / "app.json"
    if not app_json.exists():
        app_json.write_text(json.dumps({"useMarkdownLinks": False}, indent=2), encoding="utf-8")
    return obsidian_dir


def sanitize_rel_folder(folder: str) -> Path:
    """Sanitize a user-supplied relative folder path inside the archive."""
    parts = [p.strip() for p in folder.replace("\\", "/").split("/") if p.strip() not in ("", ".", "..")]
    return Path(*parts) if parts else Path()


def _resolve_folder_path(archive_dir: Path, folder: str) -> tuple[Path, Path]:
    root = archive_dir.expanduser().resolve()
    rel = sanitize_rel_folder(folder)
    if not rel.parts:
        raise ValueError("Folder path is required.")
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Invalid folder path.")
    return root, target


def rename_folder(archive_dir: Path, folder: str, new_name: str) -> str:
    """Rename an archive subfolder within its parent directory."""
    clean_name = new_name.strip()
    if not clean_name or "/" in clean_name or "\\" in clean_name or clean_name in (".", "..") or clean_name.startswith("."):
        raise ValueError("Invalid folder name.")
    root, src = _resolve_folder_path(archive_dir, folder)
    if not src.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    dst = (src.parent / clean_name).resolve()
    if dst == src:
        return src.relative_to(root).as_posix()
    if dst.exists():
        raise FileExistsError(f"Folder already exists: {clean_name}")
    src.rename(dst)
    return dst.relative_to(root).as_posix()


def delete_folder(archive_dir: Path, folder: str, recursive: bool = False) -> None:
    """Delete an empty subfolder, or recursively delete it when recursive=True."""
    _root, target = _resolve_folder_path(archive_dir, folder)
    if not target.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if recursive:
        shutil.rmtree(target)
    else:
        target.rmdir()


def _build_breadcrumbs(current_folder: str) -> list[dict[str, str]]:
    crumbs = [{"name": "Archive", "path": ""}]
    if not current_folder:
        return crumbs
    acc: list[str] = []
    for part in current_folder.split("/"):
        acc.append(part)
        crumbs.append({"name": part, "path": "/".join(acc)})
    return crumbs


def scan_artifacts(
    archive_dir: Path,
    query: str = "",
    folder: str = "",
    tag: str = "",
    sort: str = "newest",
    cwd: str | None = None,
) -> dict:
    """Index Markdown artifacts in archive_dir with Explorer-style folder browsing, tags, and search."""
    root = archive_dir.expanduser().resolve()
    items: list[dict] = []
    folder_set: set[str] = set()
    folder_doc_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    q_lower = query.strip().lower()
    folder_filter = folder.strip().strip("/")
    tag_filter = tag.strip().lstrip("#").lower()
    current_rel = sanitize_rel_folder(cwd or "")
    current_folder = current_rel.as_posix() if current_rel.parts else ""
    browse_exact_folder = cwd is not None and not q_lower and not tag_filter

    if root.exists():
        for d in root.rglob("*"):
            if d.is_dir() and not any(part.startswith(".") for part in d.relative_to(root).parts):
                folder_set.add(d.relative_to(root).as_posix())

        for md_file in root.rglob("*.md"):
            rel_parts = md_file.relative_to(root).parts
            if any(part.startswith(".") for part in rel_parts):
                continue

            rel_path = md_file.relative_to(root).as_posix()
            rel_folder = md_file.parent.relative_to(root).as_posix()
            if rel_folder == ".":
                rel_folder = ""
            else:
                folder_set.add(rel_folder)
                acc = []
                for seg in rel_folder.split("/"):
                    acc.append(seg)
                    key = "/".join(acc)
                    folder_doc_counts[key] = folder_doc_counts.get(key, 0) + 1

            raw_text = md_file.read_text(encoding="utf-8", errors="replace")
            meta, body = parse_frontmatter(raw_text)
            doc_tags = extract_tags(raw_text)
            for t in doc_tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

            if browse_exact_folder:
                if rel_folder != current_folder:
                    continue
            elif folder_filter and rel_folder != folder_filter and not rel_folder.startswith(f"{folder_filter}/"):
                continue
            if tag_filter and tag_filter not in [t.lower() for t in doc_tags]:
                continue
            if q_lower:
                haystack = f"{rel_path}\n{' '.join(doc_tags)}\n{raw_text}".lower()
                if q_lower not in haystack:
                    continue

            stat = md_file.stat()
            items.append(
                {
                    "path": rel_path,
                    "folder": rel_folder,
                    "name": md_file.name,
                    "title": str(meta.get("title") or md_file.stem),
                    "tags": doc_tags,
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "preview": body.strip()[:180],
                }
            )

    if sort == "oldest":
        items.sort(key=lambda x: x["mtime"])
    elif sort == "name":
        items.sort(key=lambda x: x["path"].lower())
    elif sort == "size":
        items.sort(key=lambda x: x["size_bytes"], reverse=True)
    else:
        items.sort(key=lambda x: x["mtime"], reverse=True)

    child_folders: list[dict] = []
    prefix = f"{current_folder}/" if current_folder else ""
    for fpath in sorted(folder_set, key=str.lower):
        if not fpath.startswith(prefix):
            continue
        rest = fpath[len(prefix) :]
        if rest and "/" not in rest:
            child_folders.append(
                {"name": rest, "path": fpath, "doc_count": folder_doc_counts.get(fpath, 0)}
            )

    return {
        "archive_dir": str(root),
        "vault_name": root.name,
        "current_folder": current_folder,
        "breadcrumbs": _build_breadcrumbs(current_folder),
        "child_folders": child_folders,
        "folders": sorted(folder_set, key=str.lower),
        "tags": [{"name": k, "count": tag_counts[k]} for k in sorted(tag_counts, key=str.lower)],
        "artifacts": items,
    }
