import {
  aiCleanupArtifact,
  aiSuggestRename,
  aiSuggestTags,
  aiSummarizeArtifact,
  aiSummarizeFolder,
  artifactDownloadUrl,
  createArtifactFolder,
  deleteArtifact,
  deleteArtifactFolder,
  fetchArtifactContent,
  fetchArtifacts,
  moveOrRenameArtifact,
  openInObsidian,
  renameArtifactFolder,
  revealArtifact,
  saveArtifactContent,
  updateArtifactTags,
} from "./api.js";
import { bsConfirm, bsEditTags, bsPrompt, triggerDownload } from "./dialogs.js";
import { renderMarkdownContent } from "./markdown.js";

const el = (id) => document.getElementById(id);

export function initArtifactsView({ showToast, onArchiveChanged }) {
  const artState = {
    cwd: "",
    breadcrumbs: [],
    childFolders: [],
    items: [],
    folders: [],
    folderSummary: null,
    knownTags: [],
    selected: null,
    rawMode: false,
    rawText: "",
    obsidianMode: false,
    llmAvailable: false,
  };

  function notifyChanged() {
    if (onArchiveChanged) onArchiveChanged();
  }

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
    el("artifactAiBtn").disabled = !enabled || !artState.llmAvailable;
    el("artifactRevealBtn").disabled = !enabled;
    el("artifactObsidianBtn").disabled = !enabled;
    el("artifactDeleteBtn").disabled = !enabled;
    el("artifactDownloadBtn").classList.toggle("disabled", !enabled);
  }

  function setLlmAvailable(available) {
    artState.llmAvailable = Boolean(available);
    el("aiFolderSummaryBtn").disabled = !artState.llmAvailable;
    el("artifactAiBtn").disabled = !artState.selected || !artState.llmAvailable;
  }

  function clearSelection() {
    artState.selected = null;
    artState.rawText = "";
    el("artifactSelectedPath").textContent = "Select an artifact";
    el("artifactSelectedSummary").textContent = "";
    renderSelectedTags([]);
    setActionsEnabled(false);
    renderPreview();
  }

  async function selectArtifact(item) {
    artState.selected = item;
    el("artifactSelectedPath").textContent = item.path;
    el("artifactSelectedSummary").textContent = item.summary || "";
    renderSelectedTags(item.tags);
    el("artifactDownloadBtn").href = artifactDownloadUrl(item.path);
    setActionsEnabled(true);
    artState.rawText = await fetchArtifactContent(item.path);
    renderPreview();
    renderBrowser();
  }

  function populateTagDropdown(tags) {
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

  function renderBreadcrumbs(filtering) {
    const bc = el("artifactBreadcrumb");
    bc.innerHTML = "";
    if (filtering) {
      const li = document.createElement("li");
      li.className = "breadcrumb-item active";
      li.textContent = "Search / Tag Results across Archive";
      bc.appendChild(li);
      return;
    }
    for (let i = 0; i < artState.breadcrumbs.length; i++) {
      const crumb = artState.breadcrumbs[i];
      const isLast = i === artState.breadcrumbs.length - 1;
      const li = document.createElement("li");
      li.className = `breadcrumb-item ${isLast ? "active fw-semibold" : ""}`;
      if (isLast) {
        li.textContent = crumb.name;
      } else {
        const a = document.createElement("a");
        a.href = "#";
        a.textContent = crumb.name;
        a.addEventListener("click", (e) => {
          e.preventDefault();
          artState.cwd = crumb.path;
          refresh();
        });
        li.appendChild(a);
      }
      bc.appendChild(li);
    }
  }

  function attachDropTarget(row, targetFolder) {
    row.addEventListener("dragover", (e) => {
      if (e.dataTransfer?.types?.includes("text/plain")) {
        e.preventDefault();
        row.classList.add("folder-drop-over");
      }
    });
    row.addEventListener("dragleave", () => row.classList.remove("folder-drop-over"));
    row.addEventListener("drop", async (e) => {
      e.preventDefault();
      row.classList.remove("folder-drop-over");
      const srcPath = e.dataTransfer?.getData("text/plain");
      if (!srcPath) return;
      try {
        const res = await moveOrRenameArtifact(srcPath, { targetFolder });
        showToast("Moved", res.path, "success");
        if (artState.selected?.path === srcPath) clearSelection();
        await refresh();
        notifyChanged();
      } catch (err) {
        showToast("Move Failed", err.message, "danger");
      }
    });
  }

  function makeFolderRow(folder) {
    const row = document.createElement("div");
    row.className = "list-group-item list-group-item-action d-flex justify-content-between align-items-center folder-row";
    row.style.cursor = "pointer";
    row.innerHTML = `
      <div class="d-flex align-items-center gap-2 text-truncate me-2">
        <i class="bi bi-folder-fill text-warning"></i>
        <span class="fw-medium text-truncate folder-name"></span>
        <span class="badge bg-body-secondary text-body border">${folder.doc_count}</span>
      </div>
      <div class="btn-group btn-group-sm folder-actions flex-shrink-0">
        <button type="button" class="btn btn-xs btn-outline-secondary rename-dir-btn" title="Rename folder">
          <i class="bi bi-pencil"></i>
        </button>
        <button type="button" class="btn btn-xs btn-outline-danger del-dir-btn" title="Delete folder">
          <i class="bi bi-trash"></i>
        </button>
      </div>`;
    row.querySelector(".folder-name").textContent = folder.name;

    row.addEventListener("click", (e) => {
      if (e.target.closest("button")) return;
      artState.cwd = folder.path;
      refresh();
    });

    row.querySelector(".rename-dir-btn").addEventListener("click", async () => {
      const nextName = await bsPrompt({
        title: "Rename Folder",
        label: "New folder name",
        value: folder.name,
        confirmText: "Rename",
      });
      if (!nextName || !nextName.trim() || nextName.trim() === folder.name) return;
      try {
        await renameArtifactFolder(folder.path, nextName.trim());
        showToast("Folder Renamed", nextName.trim(), "success");
        await refresh();
        notifyChanged();
      } catch (err) {
        showToast("Rename Failed", err.message, "danger");
      }
    });

    row.querySelector(".del-dir-btn").addEventListener("click", async () => {
      const ok = await bsConfirm({
        title: "Delete Folder",
        message: `Delete folder "${folder.path}"?`,
        confirmText: "Delete",
        variant: "danger",
      });
      if (!ok) return;
      try {
        await deleteArtifactFolder(folder.path, false);
        showToast("Folder Deleted", folder.path);
        await refresh();
        notifyChanged();
      } catch (err) {
        if (err.message.toLowerCase().includes("not empty")) {
          const force = await bsConfirm({
            title: "Folder Not Empty",
            message: `"${folder.path}" contains files or subfolders. Delete it and ALL contents permanently?`,
            confirmText: "Delete All",
            variant: "danger",
          });
          if (!force) return;
          try {
            await deleteArtifactFolder(folder.path, true);
            if (artState.selected?.path.startsWith(`${folder.path}/`)) clearSelection();
            showToast("Folder Deleted", folder.path);
            await refresh();
            notifyChanged();
          } catch (err2) {
            showToast("Delete Failed", err2.message, "danger");
          }
        } else {
          showToast("Delete Failed", err.message, "danger");
        }
      }
    });

    attachDropTarget(row, folder.path);
    return row;
  }

  function renderBrowser() {
    const q = el("artifactSearchInput").value.trim();
    const tag = el("artifactTagSelect").value;
    const filtering = Boolean(q || tag);
    el("clearFiltersBtn").classList.toggle("d-none", !filtering);
    renderBreadcrumbs(filtering);

    const list = el("artifactList");
    list.innerHTML = "";

    if (!filtering && artState.cwd) {
      const parent = artState.cwd.includes("/") ? artState.cwd.slice(0, artState.cwd.lastIndexOf("/")) : "";
      const upRow = document.createElement("button");
      upRow.type = "button";
      upRow.className = "list-group-item list-group-item-action text-muted small d-flex align-items-center gap-2";
      upRow.innerHTML = '<i class="bi bi-arrow-90deg-up"></i><span>.. (Up a folder)</span>';
      upRow.addEventListener("click", () => {
        artState.cwd = parent;
        refresh();
      });
      attachDropTarget(upRow, parent);
      list.appendChild(upRow);
    }

    if (!filtering) {
      for (const folder of artState.childFolders) {
        list.appendChild(makeFolderRow(folder));
      }
    }

    for (const item of artState.items) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.draggable = true;
      const active = artState.selected && item.path === artState.selected.path ? "active" : "";
      btn.className = `list-group-item list-group-item-action ${active}`;
      btn.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
          <span class="fw-semibold text-truncate me-2"><i class="bi bi-file-earmark-text me-1"></i><span class="file-title"></span></span>
          <small class="opacity-75 text-nowrap">${Math.max(1, Math.round(item.size_bytes / 1024))} KB</small>
        </div>
        <div class="small opacity-75 text-truncate meta-line"></div>
        <div class="d-flex flex-wrap gap-1 mt-1 tag-row"></div>`;
      btn.querySelector(".file-title").textContent = item.name;
      const folderHint = filtering && item.folder ? `${item.folder} · ` : "";
      const summaryHint = item.summary ? ` · ${item.summary}` : "";
      btn.querySelector(".meta-line").textContent = `${folderHint}${item.modified_at}${summaryHint}`;
      const tagRow = btn.querySelector(".tag-row");
      for (const t of (item.tags || []).slice(0, 4)) {
        const badge = document.createElement("span");
        badge.className = "badge bg-body-secondary text-body border";
        badge.textContent = `#${t}`;
        tagRow.appendChild(badge);
      }
      btn.addEventListener("dragstart", (e) => {
        e.dataTransfer?.setData("text/plain", item.path);
      });
      btn.addEventListener("click", () => selectArtifact(item));
      list.appendChild(btn);
    }

    if (!list.children.length) {
      list.innerHTML = '<div class="list-group-item text-muted small">This folder is empty.</div>';
    }
  }

  async function refresh() {
    const data = await fetchArtifacts({
      cwd: artState.cwd,
      q: el("artifactSearchInput").value.trim(),
      tag: el("artifactTagSelect").value,
      sort: el("artifactSortSelect").value,
    });
    artState.cwd = data.current_folder || "";
    artState.breadcrumbs = data.breadcrumbs || [{ name: "Archive", path: "" }];
    artState.childFolders = data.child_folders || [];
    artState.items = data.artifacts || [];
    artState.folders = data.folders || [];
    artState.folderSummary = data.folder_summary || null;
    artState.knownTags = data.recent_tags || (data.tags || []).map((t) => t.name);
    artState.obsidianMode = Boolean(data.obsidian_mode);
    el("artifactDirLabel").textContent = data.archive_dir || "";
    el("obsidianBadge").classList.toggle("d-none", !artState.obsidianMode);
    el("artifactObsidianBtn").classList.toggle("d-none", !artState.obsidianMode);

    const fBox = el("artifactFolderSummaryBox");
    if (artState.folderSummary?.summary) {
      fBox.classList.remove("d-none");
      const plain = artState.folderSummary.summary.replace(/^#+\s*/gm, "").replace(/\s+/g, " ").trim();
      el("artifactFolderSummaryText").textContent = plain.slice(0, 180) + (plain.length > 180 ? "…" : "");
    } else {
      fBox.classList.add("d-none");
      el("artifactFolderSummaryText").textContent = "";
    }

    populateTagDropdown(data.tags);
    renderBrowser();
  }

  el("artifactSearchInput").addEventListener("input", refresh);
  el("artifactTagSelect").addEventListener("change", refresh);
  el("artifactSortSelect").addEventListener("change", refresh);
  el("refreshArtifactsBtn").addEventListener("click", refresh);
  el("openArchiveFolderBtn").addEventListener("click", () => revealArtifact(""));
  el("clearFiltersBtn").addEventListener("click", () => {
    el("artifactSearchInput").value = "";
    el("artifactTagSelect").value = "";
    refresh();
  });

  el("newFolderBtn").addEventListener("click", async () => {
    const name = await bsPrompt({
      title: "New Folder",
      label: artState.cwd ? `Create subfolder inside "${artState.cwd}"` : "Folder name inside archive",
      placeholder: "e.g. Chemistry",
      confirmText: "Create",
    });
    if (!name || !name.trim()) return;
    const target = artState.cwd ? `${artState.cwd}/${name.trim()}` : name.trim();
    try {
      const res = await createArtifactFolder(target);
      showToast("Folder Created", res.folder, "success");
      await refresh();
      notifyChanged();
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
  el("artifactDownloadBtn").addEventListener("click", (e) => {
    if (!artState.selected) return;
    e.preventDefault();
    triggerDownload(artifactDownloadUrl(artState.selected.path), artState.selected.name, showToast);
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
    const nextTags = await bsEditTags({
      path: artState.selected.path,
      initialTags: artState.selected.tags || [],
      suggestions: artState.knownTags,
    });
    if (nextTags === null) return;
    try {
      const res = await updateArtifactTags(artState.selected.path, nextTags);
      artState.selected.tags = res.tags;
      await selectArtifact(artState.selected);
      await refresh();
      notifyChanged();
      showToast("Tags Updated", res.tags.length ? res.tags.map((t) => `#${t}`).join(" ") : "Cleared tags.");
    } catch (err) {
      showToast("Tag Error", err.message, "danger");
    }
  });

  el("artifactRenameBtn").addEventListener("click", async () => {
    if (!artState.selected) return;
    const nextName = await bsPrompt({
      title: "Rename Artifact",
      label: "New filename (.md)",
      value: artState.selected.name,
      confirmText: "Rename",
    });
    if (!nextName || !nextName.trim() || nextName.trim() === artState.selected.name) return;
    try {
      const res = await moveOrRenameArtifact(artState.selected.path, { newName: nextName.trim() });
      showToast("Renamed", res.path, "success");
      await refresh();
      const updated = artState.items.find((i) => i.path === res.path);
      if (updated) await selectArtifact(updated);
    } catch (err) {
      showToast("Rename Failed", err.message, "danger");
    }
  });

  el("artifactDeleteBtn").addEventListener("click", async () => {
    if (!artState.selected) return;
    const confirmed = await bsConfirm({
      title: "Delete Artifact",
      message: `Delete "${artState.selected.path}" permanently? This cannot be undone.`,
      confirmText: "Delete",
      variant: "danger",
    });
    if (!confirmed) return;
    try {
      await deleteArtifact(artState.selected.path);
      clearSelection();
      await refresh();
      notifyChanged();
      showToast("Deleted", "Artifact removed.");
    } catch (err) {
      showToast("Delete Failed", err.message, "danger");
    }
  });

  el("aiTagsBtn").addEventListener("click", async () => {
    if (!artState.selected) return;
    showToast("AI Tags", "Analyzing document tags...");
    try {
      const res = await aiSuggestTags(artState.selected.path, true);
      artState.selected.tags = res.tags;
      await selectArtifact(artState.selected);
      await refresh();
      notifyChanged();
      showToast("AI Tags Applied", res.tags.map((t) => `#${t}`).join(" ") || "No tags suggested.", "success");
    } catch (err) {
      showToast("AI Tag Error", err.message, "danger");
    }
  });

  el("aiRenameBtn").addEventListener("click", async () => {
    if (!artState.selected) return;
    showToast("AI Rename", "Generating suggested filename...");
    try {
      const suggestion = await aiSuggestRename(artState.selected.path);
      const nextName = await bsPrompt({
        title: "AI Suggested Rename",
        label: "Confirm or edit suggested filename (.md)",
        value: suggestion.suggested_name,
        confirmText: "Rename",
      });
      if (!nextName || !nextName.trim() || nextName.trim() === artState.selected.name) return;
      const res = await moveOrRenameArtifact(artState.selected.path, { newName: nextName.trim() });
      showToast("Renamed", res.path, "success");
      await refresh();
      const updated = artState.items.find((i) => i.path === res.path);
      if (updated) await selectArtifact(updated);
    } catch (err) {
      showToast("AI Rename Error", err.message, "danger");
    }
  });

  el("aiSummarizeBtn").addEventListener("click", async () => {
    if (!artState.selected) return;
    showToast("AI Summary", "Summarizing note...");
    try {
      const res = await aiSummarizeArtifact(artState.selected.path);
      artState.selected.summary = res.summary;
      await selectArtifact(artState.selected);
      await refresh();
      showToast("Summary Saved", res.summary || "Updated.", "success");
    } catch (err) {
      showToast("AI Summary Error", err.message, "danger");
    }
  });

  let pendingCleanup = null;
  el("aiCleanupBtn").addEventListener("click", async () => {
    if (!artState.selected) return;
    showToast("AI Proofread", "Cleaning OCR typos and formatting...");
    try {
      const res = await aiCleanupArtifact(artState.selected.path);
      pendingCleanup = { path: artState.selected.path, content: res.cleaned_markdown };
      el("aiCleanupFileLabel").textContent = artState.selected.path;
      renderMarkdownContent(el("aiCleanupPreview"), res.cleaned_markdown, false);
      window.bootstrap.Modal.getOrCreateInstance(el("aiCleanupModal")).show();
    } catch (err) {
      showToast("AI Cleanup Error", err.message, "danger");
    }
  });

  el("aiCleanupApplyBtn").addEventListener("click", async () => {
    if (!pendingCleanup) return;
    try {
      await saveArtifactContent(pendingCleanup.path, pendingCleanup.content);
      window.bootstrap.Modal.getOrCreateInstance(el("aiCleanupModal")).hide();
      if (artState.selected?.path === pendingCleanup.path) {
        await selectArtifact(artState.selected);
      }
      await refresh();
      showToast("Cleanup Applied", "Saved cleaned Markdown.", "success");
    } catch (err) {
      showToast("Save Failed", err.message, "danger");
    }
  });

  function showFolderSummaryModal(summaryObj) {
    const folderLabel = summaryObj.folder || "Archive Root";
    const updatedPart = summaryObj.updated_at ? ` · updated ${summaryObj.updated_at}` : "";
    el("aiFolderSummaryMeta").textContent = `${folderLabel} · ${summaryObj.doc_count || 0} document(s)${updatedPart}`;
    renderMarkdownContent(
      el("aiFolderSummaryPreview"),
      summaryObj.summary || "_No Markdown documents found in this folder._",
      false
    );
    window.bootstrap.Modal.getOrCreateInstance(el("aiFolderSummaryModal")).show();
  }

  async function runFolderSummarize() {
    showToast("AI Folder Summary", "Summarizing folder contents...");
    try {
      const res = await aiSummarizeFolder(artState.cwd);
      if (!res.doc_count) {
        showToast("Empty Folder", "No Markdown documents found in this folder.", "info");
        return;
      }
      await refresh();
      showFolderSummaryModal(res);
      showToast("Folder Summarized", `Synthesized ${res.doc_count} document(s).`, "success");
    } catch (err) {
      showToast("AI Folder Summary Error", err.message, "danger");
    }
  }

  el("aiFolderSummaryBtn").addEventListener("click", runFolderSummarize);
  el("aiFolderSummaryRefreshBtn").addEventListener("click", runFolderSummarize);
  el("viewFolderSummaryBtn").addEventListener("click", () => {
    if (artState.folderSummary) showFolderSummaryModal(artState.folderSummary);
  });
  el("aiFolderSummaryCopyBtn").addEventListener("click", async () => {
    const text = artState.folderSummary?.summary || "";
    if (!text) return;
    await navigator.clipboard.writeText(text);
    showToast("Copied", "Folder summary copied to clipboard.");
  });

  renderPreview();
  return { refresh, setLlmAvailable };
}
