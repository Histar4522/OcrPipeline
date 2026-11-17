import {
  artifactDownloadUrl,
  deleteArtifact,
  fetchArtifactContent,
  fetchArtifacts,
  renameArtifact,
  revealArtifact,
} from "./api.js";

const el = (id) => document.getElementById(id);

export function initArtifactsView({ showToast }) {
  const artState = { items: [], selectedPath: null, rawMode: false, rawText: "" };

  function renderPreview() {
    const box = el("artifactPreview");
    if (!artState.rawText) {
      box.innerHTML = '<span class="text-muted small">Select an artifact to preview.</span>';
      return;
    }
    if (artState.rawMode) {
      box.innerHTML = '<pre class="mb-0 small"></pre>';
      box.querySelector("pre").textContent = artState.rawText;
    } else {
      box.innerHTML = window.marked.parse(artState.rawText);
    }
  }

  function setActionsEnabled(enabled) {
    el("artifactRenameBtn").disabled = !enabled;
    el("artifactRevealBtn").disabled = !enabled;
    el("artifactDeleteBtn").disabled = !enabled;
    el("artifactDownloadBtn").classList.toggle("disabled", !enabled);
  }

  async function selectArtifact(path) {
    artState.selectedPath = path;
    el("artifactSelectedPath").textContent = path;
    el("artifactDownloadBtn").href = artifactDownloadUrl(path);
    setActionsEnabled(true);
    artState.rawText = await fetchArtifactContent(path);
    renderPreview();
    renderList();
  }

  function renderList() {
    const query = el("artifactSearchInput").value.trim().toLowerCase();
    const filtered = artState.items.filter(
      (a) => !query || a.path.toLowerCase().includes(query) || a.preview.toLowerCase().includes(query)
    );
    const list = el("artifactList");
    if (!filtered.length) {
      list.innerHTML = '<div class="list-group-item text-muted small">No archived .md artifacts found.</div>';
      return;
    }
    list.innerHTML = "";
    for (const item of filtered) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `list-group-item list-group-item-action ${item.path === artState.selectedPath ? "active" : ""}`;
      btn.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
          <span class="fw-semibold text-truncate me-2"></span>
          <small class="opacity-75 text-nowrap">${Math.max(1, Math.round(item.size_bytes / 1024))} KB</small>
        </div>
        <div class="small opacity-75 text-truncate">${item.modified_at}</div>`;
      btn.querySelector("span").textContent = item.path;
      btn.addEventListener("click", () => selectArtifact(item.path));
      list.appendChild(btn);
    }
  }

  async function refresh() {
    const data = await fetchArtifacts();
    artState.items = data.artifacts || [];
    el("artifactDirLabel").textContent = data.archive_dir || "";
    renderList();
  }

  el("artifactSearchInput").addEventListener("input", renderList);
  el("refreshArtifactsBtn").addEventListener("click", refresh);
  el("openArchiveFolderBtn").addEventListener("click", () => revealArtifact(""));
  el("artifactRevealBtn").addEventListener("click", () => {
    if (artState.selectedPath) revealArtifact(artState.selectedPath);
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
  el("artifactRenameBtn").addEventListener("click", async () => {
    if (!artState.selectedPath) return;
    const currentName = artState.selectedPath.split("/").pop();
    const nextName = window.prompt("Rename artifact (.md):", currentName);
    if (!nextName || nextName === currentName) return;
    try {
      const res = await renameArtifact(artState.selectedPath, nextName);
      showToast("Renamed", `Saved as ${res.path}`);
      await refresh();
      await selectArtifact(res.path);
    } catch (err) {
      showToast("Rename Failed", err.message, "danger");
    }
  });
  el("artifactDeleteBtn").addEventListener("click", async () => {
    if (!artState.selectedPath) return;
    if (!window.confirm(`Delete "${artState.selectedPath}" permanently?`)) return;
    try {
      await deleteArtifact(artState.selectedPath);
      artState.selectedPath = null;
      artState.rawText = "";
      el("artifactSelectedPath").textContent = "Select an artifact";
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
