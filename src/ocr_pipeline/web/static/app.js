import {
  cancelJob,
  createArtifactFolder,
  fetchArtifacts,
  fetchFolders,
  fetchHealth,
  fetchJob,
  fetchJobFile,
  fetchJobLog,
  fetchJobs,
  fetchModels,
  fetchSettings,
  fileDownloadUrl,
  pullModel,
  saveSettings,
  startJob,
  startOllama,
} from "./api.js";
import { initArtifactsView } from "./artifacts.js";
import { bsPrompt, triggerDownload } from "./dialogs.js";
import { renderMarkdownContent } from "./markdown.js";
import { initModelsView } from "./models.js";
import { initSettingsAndWizard } from "./settings_wizard.js";
import { createTagInput } from "./tag_input.js";

const state = {
  selectedFiles: [],
  pulledModels: [],
  sourceMode: "upload",
  activeJobId: null,
  activeFile: null,
  rawMode: false,
  rawMarkdown: "",
  pollTimer: null,
  lastJobStatuses: new Map(),
};

const el = (id) => document.getElementById(id);
const mediaDark = window.matchMedia("(prefers-color-scheme: dark)");

function applyTheme(mode = "auto") {
  state.theme = mode;
  const effective = mode === "auto" ? (mediaDark.matches ? "dark" : "light") : mode;
  document.documentElement.setAttribute("data-bs-theme", effective);

  const labels = { auto: "Auto", light: "Light", dark: "Dark" };
  const icons = { auto: "bi-circle-half", light: "bi-sun", dark: "bi-moon-stars" };
  el("themeLabel").textContent = labels[mode] || "Auto";
  el("themeIcon").className = `bi ${icons[mode] || "bi-circle-half"} me-1`;
  document.querySelectorAll("#themeMenu [data-theme]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.theme === mode);
  });
}

mediaDark.addEventListener("change", () => {
  if ((state.theme || "auto") === "auto") applyTheme("auto");
});

function showToast(title, body, variant = "primary") {
  const container = el("toastContainer");
  const node = document.createElement("div");
  node.className = `toast align-items-center text-bg-${variant} border-0`;
  node.innerHTML = `
    <div class="d-flex">
      <div class="toast-body"><strong class="me-1"></strong><span></span></div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
    </div>`;
  node.querySelector("strong").textContent = title;
  node.querySelector("span").textContent = body;
  container.appendChild(node);
  const toast = new window.bootstrap.Toast(node, { delay: 4500 });
  node.addEventListener("hidden.bs.toast", () => node.remove());
  toast.show();
}

function updatePullButton() {
  const selected = el("modelSelect").value;
  const pulled = state.pulledModels.includes(selected) || state.pulledModels.includes(`${selected}:latest`);
  el("pullModelBtn").classList.toggle("d-none", !selected || pulled);
}

function setNewJobAvailability({ status, installed = false }) {
  const fieldset = el("newJobFieldset");
  const banner = el("ollamaLockBanner");
  const msg = el("ollamaLockMsg");
  const quickStart = el("quickStartOllamaBtn");

  if (status === "online") {
    fieldset.disabled = false;
    fieldset.classList.remove("job-panel-disabled");
    banner.classList.add("d-none");
    return;
  }

  fieldset.disabled = true;
  fieldset.classList.add("job-panel-disabled");
  banner.classList.remove("d-none");
  if (status === "connecting") {
    banner.className = "alert alert-secondary py-2 px-3 small d-flex justify-content-between align-items-center gap-2 mb-3";
    msg.textContent = "Connecting to Ollama...";
    quickStart.classList.add("d-none");
  } else {
    banner.className = "alert alert-warning py-2 px-3 small d-flex justify-content-between align-items-center gap-2 mb-3";
    msg.textContent = installed
      ? "Ollama is installed but offline. Start Ollama to run OCR jobs."
      : "Ollama is unavailable. Open Settings -> Setup Wizard to configure Ollama.";
    quickStart.classList.toggle("d-none", !installed);
  }
}

async function syncSettingsToJobForm() {
  const badge = el("ollamaBadge");
  badge.className = "badge text-bg-secondary";
  badge.textContent = "Connecting Ollama...";
  setNewJobAvailability({ status: "connecting" });

  let modelsData;
  let settings;
  try {
    [modelsData, settings] = await Promise.all([fetchModels(), fetchSettings()]);
  } catch {
    badge.className = "badge text-bg-danger";
    badge.textContent = "Ollama Offline";
    setNewJobAvailability({ status: "offline", installed: false });
    return;
  }

  const models = modelsData.models || [];
  state.pulledModels = models;

  if (modelsData.online) {
    badge.className = "badge text-bg-success";
    badge.textContent = `Ollama Online (${models.length})`;
    setNewJobAvailability({ status: "online" });
  } else {
    badge.className = "badge text-bg-danger";
    badge.textContent = "Ollama Offline";
    setNewJobAvailability({ status: "offline", installed: Boolean(modelsData.installed) });
  }

  const defaults = ["glm-ocr", "deepseek-ocr", "qwen3-vl:4b", "qwen3-vl:8b"];
  const options = [...new Set([settings.default_model, ...defaults, ...models].filter(Boolean))];
  const select = el("modelSelect");
  select.innerHTML = options
    .map((m) => {
      const ok = models.includes(m) || models.includes(`${m}:latest`);
      return `<option value="${m}">${m}${ok ? "" : " (not pulled)"}</option>`;
    })
    .join("");
  select.value = settings.default_model;
  el("dpiInput").value = settings.dpi;
  el("maxDimInput").value = settings.max_image_dim;
  applyTheme(settings.theme || "auto");
  updatePullButton();
}

function updateSelectedFiles(files) {
  state.selectedFiles = Array.from(files);
  const info = el("selectedFilesInfo");
  if (!state.selectedFiles.length) {
    info.textContent = "No files selected.";
    return;
  }
  const preview = state.selectedFiles.slice(0, 3).map((f) => f.webkitRelativePath || f.name).join(", ");
  const extra = state.selectedFiles.length > 3 ? ` (+${state.selectedFiles.length - 3} more)` : "";
  info.textContent = `${state.selectedFiles.length} file(s) ready: ${preview}${extra}`;
}

function renderJobHistory(jobs) {
  el("jobCountBadge").textContent = String(jobs.length);
  const list = el("jobHistoryList");
  if (!jobs.length) {
    list.innerHTML = '<div class="list-group-item text-muted small">No jobs yet.</div>';
    return;
  }
  list.innerHTML = "";
  for (const j of jobs) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `list-group-item list-group-item-action ${j.job_id === state.activeJobId ? "active" : ""}`;
    btn.innerHTML = `
      <div class="d-flex justify-content-between align-items-center">
        <span class="fw-semibold text-truncate me-2"></span>
        <span class="badge text-bg-secondary">${j.status}</span>
      </div>
      <div class="small opacity-75">${j.model} · ${j.completed_docs}/${j.total_docs} docs</div>`;
    btn.querySelector("span").textContent = j.label;
    btn.addEventListener("click", () => selectJob(j.job_id));
    list.appendChild(btn);
  }
}

function renderMarkdown() {
  renderMarkdownContent(
    el("markdownPreview"),
    state.rawMarkdown,
    state.rawMode,
    "Select an output file to preview."
  );
}

async function openOutputFile(jobId, filePath) {
  state.activeFile = filePath;
  el("previewFileName").textContent = filePath;
  el("downloadFileBtn").href = fileDownloadUrl(jobId, filePath);
  el("downloadFileBtn").classList.remove("disabled");
  state.rawMarkdown = await fetchJobFile(jobId, filePath);
  renderMarkdown();
}

function formatDuration(sec) {
  if (sec === null || sec === undefined || Number.isNaN(sec)) return "--";
  const total = Math.max(0, Math.round(sec));
  const hrs = Math.floor(total / 3600);
  const mins = Math.floor((total % 3600) / 60);
  const rem = total % 60;
  if (hrs > 0) return `${hrs}h ${mins}m ${rem}s`;
  if (mins > 0) return `${mins}m ${rem}s`;
  return `${rem}s`;
}

function renderJobDetail(job) {
  el("emptyState").classList.add("d-none");
  el("jobDetailPanel").classList.remove("d-none");
  el("detailTitle").textContent = `${job.label} (${job.job_id})`;

  const badge = el("detailStatusBadge");
  const badgeMap = { completed: "text-bg-success", failed: "text-bg-danger", cancelled: "text-bg-warning", cancelling: "text-bg-warning", running: "text-bg-primary" };
  badge.className = `badge ms-2 ${badgeMap[job.status] || "text-bg-secondary"}`;
  badge.textContent = job.status;
  el("detailMeta").textContent = `Model: ${job.model} | DPI: ${job.dpi} | Output: ${job.output_dir}`;
  const etaText =
    job.status === "completed"
      ? "Done"
      : job.eta_seconds !== null && job.eta_seconds !== undefined
        ? `~${formatDuration(job.eta_seconds)}`
        : "Calculating...";
  el("detailTiming").textContent = `Elapsed: ${formatDuration(job.elapsed_seconds)} · ETA: ${etaText}`;

  const isActive = job.status === "queued" || job.status === "running";
  el("cancelJobBtn").classList.toggle("d-none", !isActive);
  el("cancelJobBtn").disabled = job.status === "cancelling";

  const docPct = job.total_docs ? Math.round((job.completed_docs / job.total_docs) * 100) : 0;
  el("docProgressText").textContent = `${job.completed_docs} / ${job.total_docs} (skipped: ${job.skipped_docs})`;
  el("docProgressBar").style.width = `${docPct}%`;

  const pagePct = job.total_pages ? Math.round((job.current_page / job.total_pages) * 100) : 0;
  el("pageProgressLabel").textContent = job.current_doc ? `Current: ${job.current_doc}` : "Current Page";
  el("pageProgressText").textContent = `${job.current_page} / ${job.total_pages}`;
  el("pageProgressBar").style.width = `${pagePct}%`;

  const errBox = el("detailError");
  errBox.classList.toggle("d-none", !job.error);
  errBox.textContent = job.error || "";

  const dlAll = el("downloadAllBtn");
  dlAll.classList.toggle("d-none", !job.files.length);
  if (job.files.length) dlAll.href = `/api/jobs/${encodeURIComponent(job.job_id)}/download-all`;

  const fileList = el("outputFileList");
  fileList.innerHTML = "";
  if (!job.files.length) {
    fileList.innerHTML = '<div class="list-group-item text-muted">No .md files yet.</div>';
  } else {
    for (const f of job.files) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `list-group-item list-group-item-action text-truncate ${f === state.activeFile ? "active" : ""}`;
      btn.textContent = f;
      btn.addEventListener("click", () => openOutputFile(job.job_id, f));
      fileList.appendChild(btn);
    }
    if (!state.activeFile) openOutputFile(job.job_id, job.files[0]);
  }
}

async function refreshActiveJob() {
  if (!state.activeJobId) return;
  const [job, logText, allJobs] = await Promise.all([
    fetchJob(state.activeJobId),
    fetchJobLog(state.activeJobId),
    fetchJobs(),
  ]);
  const prev = state.lastJobStatuses.get(job.job_id);
  state.lastJobStatuses.set(job.job_id, job.status);
  if (prev && prev !== job.status && ["completed", "failed", "cancelled"].includes(job.status)) {
    const variant = job.status === "completed" ? "success" : job.status === "cancelled" ? "warning" : "danger";
    showToast(`Job ${job.status}`, `${job.label} (${job.completed_docs}/${job.total_docs} docs)`, variant);
  }
  renderJobHistory(allJobs);
  renderJobDetail(job);
  el("logViewer").textContent = logText;
  if (["completed", "failed", "cancelled"].includes(job.status)) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    fetchArtifacts().then((d) => jobTagEditor.setSuggestions((d.tags || []).map((t) => t.name))).catch(() => {});
  }
}

function selectJob(jobId) {
  state.activeJobId = jobId;
  state.activeFile = null;
  state.rawMarkdown = "";
  renderMarkdown();
  clearInterval(state.pollTimer);
  refreshActiveJob();
  state.pollTimer = setInterval(refreshActiveJob, 1500);
}

async function syncJobArchiveOptions(selectFolder = null) {
  try {
    const [folderData, artData] = await Promise.all([fetchFolders(), fetchArtifacts()]);
    const sel = el("jobFolderSelect");
    const prev = selectFolder !== null ? selectFolder : sel.value;
    sel.innerHTML = '<option value="">(Archive Root)</option>';
    for (const f of folderData.folders || []) {
      const opt = document.createElement("option");
      opt.value = f;
      opt.textContent = f;
      sel.appendChild(opt);
    }
    if (prev && (folderData.folders || []).includes(prev)) sel.value = prev;
    jobTagEditor.setSuggestions((artData.tags || []).map((t) => t.name));
  } catch {
    // Ignore transient fetch failures
  }
}

const artifactsView = initArtifactsView({ showToast, onArchiveChanged: () => syncJobArchiveOptions() });
const jobTagEditor = createTagInput(el("jobTagsContainer"), { placeholder: "Add tag and press Enter..." });
const modelsView = initModelsView({ showToast, onModelsChanged: syncSettingsToJobForm });
const settingsWizard = initSettingsAndWizard({
  showToast,
  onSettingsSaved: syncSettingsToJobForm,
  onWizardFinished: () => switchView("jobsView"),
});

function switchView(target) {
  document.querySelectorAll("#mainNav [data-view]").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === target);
  });
  document.querySelectorAll(".app-view").forEach((v) => v.classList.add("d-none"));
  el(target).classList.remove("d-none");
  if (target === "jobsView") syncJobArchiveOptions();
  if (target === "artifactsView") artifactsView.refresh();
  if (target === "modelsView") modelsView.refresh();
  if (target === "settingsView") settingsWizard.loadSettingsForm();
}

document.querySelectorAll("#mainNav [data-view]").forEach((btn) => {
  btn.addEventListener("click", () => switchView(btn.dataset.view));
});

el("navRefreshBtn").addEventListener("click", syncSettingsToJobForm);
el("quickSettingsBtn").addEventListener("click", () => switchView("settingsView"));
el("quickStartOllamaBtn").addEventListener("click", async () => {
  const btn = el("quickStartOllamaBtn");
  btn.disabled = true;
  btn.textContent = "Starting...";
  el("ollamaLockMsg").textContent = "Starting Ollama service (this may take a few seconds)...";
  try {
    const res = await startOllama();
    showToast("Ollama", res.message, res.online ? "success" : "warning");
    await syncSettingsToJobForm();
  } catch (err) {
    showToast("Ollama", err.message, "danger");
  } finally {
    btn.disabled = false;
    btn.textContent = "Start Ollama";
  }
});

el("modelSelect").addEventListener("change", updatePullButton);
el("pullModelBtn").addEventListener("click", async () => {
  const btn = el("pullModelBtn");
  btn.disabled = true;
  btn.textContent = "Pulling...";
  try {
    await pullModel(el("modelSelect").value);
    await syncSettingsToJobForm();
    showToast("Model Ready", `${el("modelSelect").value} installed.`, "success");
  } catch (err) {
    showToast("Pull Failed", err.message, "danger");
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-download me-1"></i>Pull';
    updatePullButton();
  }
});

el("cancelJobBtn").addEventListener("click", async () => {
  if (!state.activeJobId) return;
  el("cancelJobBtn").disabled = true;
  await cancelJob(state.activeJobId);
  await refreshActiveJob();
});

el("tabUploadBtn").addEventListener("click", () => (state.sourceMode = "upload"));
el("tabLocalBtn").addEventListener("click", () => (state.sourceMode = "local"));
el("fileInput").addEventListener("change", (e) => updateSelectedFiles(e.target.files));
el("folderInput").addEventListener("change", (e) => updateSelectedFiles(e.target.files));

const dz = el("dropZone");
dz.addEventListener("click", (e) => { if (!e.target.closest("label, input, button")) el("fileInput").click(); });
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag-over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("drag-over");
  if (e.dataTransfer?.files?.length) updateSelectedFiles(e.dataTransfer.files);
});

el("toggleRawBtn").addEventListener("click", () => {
  state.rawMode = !state.rawMode;
  el("toggleRawBtn").textContent = state.rawMode ? "Rendered" : "Raw";
  renderMarkdown();
});
el("copyMdBtn").addEventListener("click", async () => {
  if (state.rawMarkdown) {
    await navigator.clipboard.writeText(state.rawMarkdown);
    showToast("Copied", "Markdown copied to clipboard.");
  }
});
el("downloadFileBtn").addEventListener("click", (e) => {
  if (!state.activeJobId || !state.activeFile) return;
  e.preventDefault();
  const fname = state.activeFile.split("/").pop() || "output.md";
  triggerDownload(fileDownloadUrl(state.activeJobId, state.activeFile), fname, showToast);
});
el("downloadAllBtn").addEventListener("click", (e) => {
  if (!state.activeJobId) return;
  e.preventDefault();
  triggerDownload(
    `/api/jobs/${encodeURIComponent(state.activeJobId)}/download-all`,
    `ocr_${state.activeJobId}.zip`,
    showToast
  );
});

el("jobNewFolderBtn").addEventListener("click", async () => {
  const name = await bsPrompt({
    title: "New Archive Folder",
    label: "Folder path inside archive",
    placeholder: "e.g. Notes/Math",
    confirmText: "Create",
  });
  if (!name || !name.trim()) return;
  try {
    const res = await createArtifactFolder(name.trim());
    await syncJobArchiveOptions(res.folder);
    showToast("Folder Created", res.folder, "success");
  } catch (err) {
    showToast("Folder Error", err.message, "danger");
  }
});

el("jobForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errEl = el("formError");
  errEl.classList.add("d-none");
  const fd = new FormData();
  fd.append("source_mode", state.sourceMode);
  fd.append("local_input_path", el("localInputPath").value);
  fd.append("local_output_path", el("localOutputPath").value);
  fd.append("job_name", el("jobNameInput").value);
  fd.append("folder", el("jobFolderSelect").value);
  fd.append("tags", jobTagEditor.getTags().join(","));
  fd.append("model", el("modelSelect").value);
  fd.append("dpi", el("dpiInput").value);
  fd.append("max_dim", el("maxDimInput").value);
  fd.append("series_mode", String(el("seriesSwitch").checked));
  fd.append("force", String(el("forceSwitch").checked));
  if (state.sourceMode === "upload") {
    for (const f of state.selectedFiles) fd.append("files", f, f.webkitRelativePath || f.name);
  }
  el("startBtn").disabled = true;
  try {
    const job = await startJob(fd);
    state.lastJobStatuses.set(job.job_id, job.status);
    renderJobHistory(await fetchJobs());
    selectJob(job.job_id);
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove("d-none");
  } finally {
    el("startBtn").disabled = false;
  }
});

document.querySelectorAll("#themeMenu [data-theme]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const mode = btn.dataset.theme;
    applyTheme(mode);
    await saveSettings({ theme: mode });
  });
});

(async () => {
  const health = await fetchHealth();
  el("aboutVersion").textContent = `v${health.version}`;
  el("aboutLicense").textContent = health.license;
  el("aboutCopyright").textContent = health.copyright;
  el("footerVersion").textContent = `OcrPipeline v${health.version}`;
  el("footerLicense").textContent = `${health.license} · ${health.copyright}`;

  await syncSettingsToJobForm();
  await syncJobArchiveOptions();
  renderJobHistory(await fetchJobs());
  if (!health.setup_completed) await settingsWizard.openWizard();
})();
