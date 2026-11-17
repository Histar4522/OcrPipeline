import {
  artifactDownloadUrl,
  createArtifactFolder,
  deleteArtifact,
  fetchArtifactContent,
  fetchArtifacts,
  moveOrRenameArtifact,
  openInObsidian,
  revealArtifact,
  updateArtifactTags,
} from "./api.js";
import { renderMarkdownContent } from "./markdown.js";

const el = (id) => document.getElementById(id);

export function initArtifactsView({ showToast }) {
  const artState = { items: [], selected: null, rawMode: false, rawText: "", obsidianMode: false };

  function renderPreview() {
    renderMarkdownContent(el("artifactPreview"), artState.rawText, artState.rawMode, "Select an artifact to preview.");
  }

  function renderSelectedTags(tags) {
    const wrap = el("artifactSelectedTags");
    wrap.innerHTML = "";
    for (const tag of tags || []) {
      const span = document.createElement("span");
      span.className = "badge text-bg-secondary";
      span.textContent = `#${tag}`;
      wrap.appendChild(span);
    }
  }

  function setActionsEnabled(enabled) {
    el("artifactTagsBtn").disabled = !enabled;
    el("artifactRenameBtn").disabled = !enabled;
    el("artifactRevealBtn").disabled = !enabled;
    el("artifactObsidianBtn").disabled = !enabled;
    el("artifactDeleteBtn").disabled = !enabled;
    el("artifactDownloadBtn").classList.toggle("disabled", !enabled);
  }

  async function selectArtifact(item) {
    artState.selected = item;
    el("artifactSelectedPath").textContent = item.path;
    renderSelectedTags(item.tags);
    el("artifactDownloadBtn").href = artifactDownloadUrl(item.path);
    setActionsEnabled(true);
    artState.rawText = await fetchArtifactContent(item.path);
    renderPreview();
    renderList();
  }

  function populateFilterDropdowns(folders, tags) {
    const folderSel = el("artifactFolderSelect");
    const prevFolder = folderSel.value;
    folderSel.innerHTML = '<option value="">All Folders</option>';
    for (const f of folders || []) {
      const opt = document.createElement("option");
      opt.value = f;
      opt.textContent = f;
      folderSel.appendChild(opt);
    }
    if (prevFolder && (folders || []).includes(prevFolder)) folderSel.value = prevFolder;

    const tagSel = el("artifactTagSelect");
    const prevTag = tagSel.value;
    tagSel.innerHTML = '<option value="">All Tags</option>';
    for (const t of tags || []) {
      const opt = document.createElement("option");
      opt.value = t.name;
      opt.textContent = `#${t.name} (${t.count})`;
      tagSel.appendChild(opt);
    }
    if (prevTag && (tags || []).some((t) => t.name === prevTag)) tagSel.value = prevTag;
  }

  function renderList() {
    const list = el("artifactList");
    if (!artState.items.length) {
      list.innerHTML = '<div class="list-group-item text-muted small">No matching .md artifacts.</div>';
      return;
    }
    list.innerHTML = "";
    for (const item of artState.items) {
      const btn = document.createElement("button");
      btn.type = "button";
      const active = artState.selected && item.path === artState.selected.path ? "active" : "";
      btn.className = `list-group-item list-group-item-action ${active}`;
      btn.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
          <span class="fw-semibold text-truncate me-2"></span>
          <small class="opacity-75 text-nowrap">${Math.max(1, Math.round(item.size_bytes / 1024))} KB</small>
        </div>
        <div class="small opacity-75 text-truncate meta-line"></div>
        <div class="d-flex flex-wrap gap-1 mt-1 tag-row"></div>`;
      btn.querySelector("span").textContent = item.name;
      btn.querySelector(".meta-line").textContent = `${item.folder ? `${item.folder} · ` : ""}${item.modified_at}`;
      const tagRow = btn.querySelector(".tag-row");
      for (const t of (item.tags || []).slice(0, 4)) {
        const badge = document.createElement("span");
        badge.className = "badge text-bg-light text-dark border";
        badge.textContent = `#${t}`;
        tagRow.appendChild(badge);
      }
      btn.addEventListener("click", () => selectArtifact(item));
      list.appendChild(btn);
    }
  }

  async function refresh() {
    const data = await fetchArtifacts({
      q: el("artifactSearchInput").value.trim(),
      folder: el("artifactFolderSelect").value,
      tag: el("artifactTagSelect").value,
      sort: el("artifactSortSelect").value,
    });
    artState.items = data.artifacts || [];
    artState.obsidianMode = Boolean(data.obsidian_mode);
    el("artifactDirLabel").textContent = data.archive_dir || "";
    el("obsidianBadge").classList.toggle("d-none", !artState.obsidianMode);
    el("artifactObsidianBtn").classList.toggle("d-none", !artState.obsidianMode);
    populateFilterDropdowns(data.folders, data.tags);
    renderList();
  }

  el("artifactSearchInput").addEventListener("input", refresh);
  el("artifactFolderSelect").addEventListener("change", refresh);
  el("artifactTagSelect").addEventListener("change", refresh);
  el("artifactSortSelect").addEventListener("change", refresh);
  el("refreshArtifactsBtn").addEventListener("click", refresh);
  el("openArchiveFolderBtn").addEventListener("click", () => revealArtifact(""));

  el("newFolderBtn").addEventListener("click", async () => {
    const name = window.prompt("New archive folder path (e.g. Notes/Physics):");
    if (!name || !name.trim()) return;
    try {
      const res = await createArtifactFolder(name.trim());
      showToast("Folder Created", res.folder, "success");
      await refresh();
    } catch (err) {
      showToast("Folder Error", err.message, "danger");
    }
  });

  el("artifactRevealBtn").addEventListener("click", () => {
    if (artState.selected) revealArtifact(artState.selected.path);
  });
  el("artifactObsidianBtn").addEventListener("click", async () => {
    if (!artState.selected) return;
    await openInObsidian(artState.selected.path);
  });
  el("artifactRawBtn").addEventListener("click", () => {
    artState.rawMode = !artState.rawMode;
    el("artifactRawBtn").textContent = artState.rawMode ? "Rendered" : "Raw";
    renderPreview();
  });
  el("artifactCopyBtn").addEventListener("click", async () => {
    if (artState.rawText) {
      await navigator.clipboard.writeText(artState.rawText);
      showToast("Copied", "Markdown copied to clipboard.");
    }
  });

  el("artifactTagsBtn").addEventListener("click", async () => {
    if (!artState.selected) return;
    const current = (artState.selected.tags || []).join(", ");
    const next = window.prompt("Comma-separated tags:", current);
    if (next === null) return;
    try {
      const res = await updateArtifactTags(artState.selected.path, next.split(","));
      artState.selected.tags = res.tags;
      await selectArtifact(artState.selected);
      await refresh();
      showToast("Tags Updated", res.tags.length ? res.tags.map((t) => `#${t}`).join(" ") : "Cleared tags.");
    } catch (err) {
      showToast("Tag Error", err.message, "danger");
    }
  });

  el("artifactRenameBtn").addEventListener("click", async () => {
    if (!artState.selected) return;
    const nextName = window.prompt("Filename (.md):", artState.selected.name);
    if (nextName === null) return;
    const nextFolder = window.prompt("Target folder (blank = archive root):", artState.selected.folder || "");
    if (nextFolder === null) return;
    try {
      const res = await moveOrRenameArtifact(artState.selected.path, {
        newName: nextName.trim() || artState.selected.name,
        targetFolder: nextFolder.trim(),
      });
      showToast("Moved / Renamed", res.path, "success");
      await refresh();
      const updated = artState.items.find((i) => i.path === res.path);
      if (updated) await selectArtifact(updated);
    } catch (err) {
      showToast("Move / Rename Failed", err.message, "danger");
    }
  });

  el("artifactDeleteBtn").addEventListener("click", async () => {
    if (!artState.selected) return;
    if (!window.confirm(`Delete "${artState.selected.path}" permanently?`)) return;
    try {
      await deleteArtifact(artState.selected.path);
      artState.selected = null;
      artState.rawText = "";
      el("artifactSelectedPath").textContent = "Select an artifact";
      renderSelectedTags([]);
      setActionsEnabled(false);
      renderPreview();
      await refresh();
      showToast("Deleted", "Artifact removed.");
    } catch (err) {
      showToast("Delete Failed", err.message, "danger");
    }
  });

  renderPreview();
  return { refresh };
}
