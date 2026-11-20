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
  fetchProviders,
  fetchProviderStatus,
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
import { CUSTOM_MODEL_VALUE, populateRoleModelSelect, readRoleModelChoice } from "./model_dropdown.js";
import { initProvidersAndModelsView } from "./providers_ui.js";
import { initSettingsAndWizard } from "./settings_wizard.js";
import { createTagInput } from "./tag_input.js";
import { initUpdatesView } from "./updates.js";

const state = {
  selectedFiles: [],
  pulledModels: [],
  providerRegistry: [],
  credentials: [],
  discoveredByCred: {},
  localOllamaStatus: null,
  ollamaOnline: false,
  ollamaInstalled: false,
  lastSettings: null,
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

function currentJobChoice() {
  return readRoleModelChoice({
    selectEl: el("jobVisualModelSelect"),
    customCredEl: el("jobCustomCredSelect"),
    customInputEl: el("jobCustomModelInput"),
    credentials: state.credentials,
  });
}

function applyNavPill(pillEl, rolePrefix, roleState, okClass) {
  if (!pillEl) return;
  if (rolePrefix === "LLM" && roleState && !roleState.enabled) {
    pillEl.className = "nav-model-pill nav-pill-disabled";
    pillEl.textContent = "LLM: Disabled";
    pillEl.title = "Assistant LLM disabled — click to configure";
    return;
  }
  const label = roleState?.label || "(Unconfigured)";
  const status = roleState?.status || "ok";
  const statusText = roleState?.status_text || "";
  const fullText = statusText ? `${rolePrefix}: ${label} [${statusText}]` : `${rolePrefix}: ${label}`;
  pillEl.textContent = fullText;
  pillEl.title = roleState?.message ? `${fullText} — ${roleState.message}` : fullText;

  let cls = okClass;
  if (status === "connecting") cls = "nav-pill-connecting";
  else if (["no_key", "no_url", "model_missing", "disabled"].includes(status)) cls = "nav-pill-warn";
  else if (["offline", "invalid_key", "out_of_quota", "error"].includes(status)) cls = "nav-pill-error";
  pillEl.className = `nav-model-pill ${cls}`;
}

function updatePullButton() {
  const choice = currentJobChoice();
  const spec = state.providerRegistry.find((r) => r.id === choice.provider);
  const isOllama = spec?.protocol === "ollama";
  const known = state.discoveredByCred[choice.credentialId] || [];
  const pulled = known.includes(choice.model) || known.includes(`${choice.model}:latest`);
  el("pullModelBtn").classList.toggle("d-none", !isOllama || !choice.model || pulled);
}

function setNewJobAvailability({ status, installed = false, message = "" }) {
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
    msg.textContent = message || "Checking provider...";
    quickStart.classList.add("d-none");
  } else {
    banner.className = "alert alert-warning py-2 px-3 small d-flex justify-content-between align-items-center gap-2 mb-3";
    msg.textContent =
      message ||
      (installed
        ? "Ollama Local is installed but offline. Start Ollama or switch Visual Model."
        : "Visual provider needs configuration in Providers & Models.");
    quickStart.classList.toggle("d-none", !installed);
  }
}

function refreshJobProviderState() {
  el("jobCustomModelPanel").classList.toggle("d-none", el("jobVisualModelSelect").value !== CUSTOM_MODEL_VALUE);
  const choice = currentJobChoice();
  const cred = state.credentials.find((c) => c.id === choice.credentialId);
  const spec = state.providerRegistry.find((r) => r.id === choice.provider);

  if (!cred || cred.enabled === false) {
    setNewJobAvailability({
      status: "offline",
      installed: false,
      message: cred ? `Credential "${cred.name}" is disabled. Enable it in Providers & Models.` : "Select a Visual credential.",
    });
  } else if (cred.id === "cred_ollama_local" || choice.provider === "ollama_local") {
    if (state.ollamaOnline) {
      setNewJobAvailability({ status: "online" });
    } else {
      setNewJobAvailability({ status: "offline", installed: state.ollamaInstalled });
    }
  } else if (spec?.requires_api_key && !cred.has_key) {
    setNewJobAvailability({
      status: "offline",
      installed: false,
      message: `${cred.name} API key missing. Configure in Providers & Models or set ${spec.env_var}.`,
    });
  } else if (choice.provider === "custom" && !cred.base_url) {
    setNewJobAvailability({
      status: "offline",
      installed: false,
      message: `${cred.name} Base URL missing. Configure in Providers & Models.`,
    });
  } else {
    setNewJobAvailability({ status: "online" });
  }
  updatePullButton();
}

async function syncSettingsToJobForm(forceRefresh = false) {
  const visPill = el("navVisualBadge");
  const llmPill = el("navLlmBadge");
  const prevVisLabel = state.lastSettings?.vision_model
    ? `${state.lastSettings.vision_model} (${state.credentials.find((c) => c.id === state.lastSettings.vision_credential_id)?.name || "Provider"})`
    : "…";
  applyNavPill(visPill, "Visual", { label: prevVisLabel, status: "connecting", status_text: "Connecting" }, "nav-pill-visual-ok");
  if (state.lastSettings?.llm_enabled && state.lastSettings?.llm_model) {
    const lCred = state.credentials.find((c) => c.id === state.lastSettings.llm_credential_id)?.name || "Provider";
    applyNavPill(llmPill, "LLM", { enabled: true, label: `${state.lastSettings.llm_model} (${lCred})`, status: "connecting", status_text: "Connecting" }, "nav-pill-llm-ok");
  }
  setNewJobAvailability({ status: "connecting" });

  let provData;
  let modelsData;
  let settings;
  let statusData;
  try {
    [provData, modelsData, settings, statusData] = await Promise.all([
      fetchProviders(),
      fetchModels(),
      fetchSettings(),
      fetchProviderStatus(forceRefresh),
    ]);
  } catch {
    applyNavPill(visPill, "Visual", { label: prevVisLabel, status: "offline", status_text: "Offline", message: "Backend unreachable" }, "nav-pill-visual-ok");
    setNewJobAvailability({ status: "offline", installed: false });
    return;
  }

  state.providerRegistry = provData.registry || [];
  state.credentials = provData.credentials || [];
  state.localOllamaStatus = statusData?.ollama_local || null;
  state.pulledModels = statusData?.ollama_local?.models || modelsData.models || [];
  state.ollamaOnline = Boolean(statusData?.ollama_local?.online ?? modelsData.online);
  state.ollamaInstalled = Boolean(statusData?.ollama_local?.installed ?? modelsData.installed);
  state.lastSettings = settings;
  if (state.pulledModels.length) {
    state.discoveredByCred["cred_ollama_local"] = state.pulledModels;
  }

  populateRoleModelSelect({
    selectEl: el("jobVisualModelSelect"),
    customPanelEl: el("jobCustomModelPanel"),
    customCredEl: el("jobCustomCredSelect"),
    customInputEl: el("jobCustomModelInput"),
    credentials: state.credentials,
    registry: state.providerRegistry,
    discoveredByCred: state.discoveredByCred,
    role: "vision",
    current: {
      credentialId: settings.vision_credential_id || "cred_ollama_local",
      model: settings.vision_model || settings.default_model || "glm-ocr",
      custom: Boolean(settings.vision_custom_model),
    },
  });

  el("dpiInput").value = settings.dpi;
  el("maxDimInput").value = settings.max_image_dim;
  applyTheme(settings.theme || "auto");

  applyNavPill(visPill, "Visual", statusData.vision, "nav-pill-visual-ok");
  applyNavPill(llmPill, "LLM", statusData.llm, "nav-pill-llm-ok");

  const llmOn = Boolean(settings.llm_available);
  el("jobAiStatusHint").textContent = llmOn ? statusData.llm?.label || settings.llm_model : "Optional LLM disabled";
  for (const [id, val] of [
    ["jobAutoTag", settings.auto_tag_on_ocr],
    ["jobAutoSummarize", settings.auto_summarize_on_ocr],
    ["jobAutoRename", settings.auto_rename_on_ocr],
  ]) {
    el(id).disabled = !llmOn;
    el(id).checked = llmOn && Boolean(val);
  }
  artifactsView.setLlmAvailable(llmOn);
  refreshJobProviderState();
}

function formatBytes(bytes) {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function openSelectedFile(item) {
  if (item.path && window.pywebview?.api?.open_file) {
    try {
      const res = await window.pywebview.api.open_file(item.path);
      if (!res?.ok) showToast("Open Failed", res?.error || "Could not open file.", "danger");
    } catch (err) {
      showToast("Open Failed", err?.message || String(err), "danger");
    }
    return;
  }
  if (item.file) {
    const url = URL.createObjectURL(item.file);
    window.open(url, "_blank", "noopener");
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }
}

async function revealSelectedFile(item) {
  if (!item.path || !window.pywebview?.api?.reveal_file) return;
  try {
    const res = await window.pywebview.api.reveal_file(item.path);
    if (!res?.ok) showToast("Open Folder Failed", res?.error || "Could not reveal file.", "danger");
  } catch (err) {
    showToast("Open Folder Failed", err?.message || String(err), "danger");
  }
}

function renderSelectedFiles() {
  const info = el("selectedFilesInfo");
  const clearBtn = el("clearSelectedFilesBtn");
  const list = el("selectedFilesList");
  const count = state.selectedFiles.length;

  info.textContent = count ? `${count} file(s) ready:` : "No files selected.";
  clearBtn.classList.toggle("d-none", count === 0);
  list.classList.toggle("d-none", count === 0);
  list.innerHTML = "";

  for (const item of state.selectedFiles) {
    const row = document.createElement("div");
    row.className = "list-group-item d-flex align-items-center justify-content-between gap-2 py-1 px-2 small";

    const isPdf = item.name.toLowerCase().endsWith(".pdf");
    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "btn btn-link btn-sm p-0 text-start text-decoration-none text-body text-truncate flex-grow-1 d-flex align-items-center gap-1 open-file-btn";
    openBtn.title = item.path ? `${item.path} — Click to open` : `${item.rel} — Click to preview`;
    const icon = document.createElement("i");
    icon.className = isPdf ? "bi bi-file-earmark-pdf text-danger flex-shrink-0" : "bi bi-file-earmark-image text-primary flex-shrink-0";
    const nameSpan = document.createElement("span");
    nameSpan.className = "text-truncate";
    nameSpan.textContent = item.rel;
    openBtn.append(icon, nameSpan);
    openBtn.addEventListener("click", () => openSelectedFile(item));

    const actions = document.createElement("div");
    actions.className = "d-flex align-items-center gap-1 flex-shrink-0";

    const sizeEl = document.createElement("small");
    sizeEl.className = "text-muted text-nowrap me-1";
    sizeEl.textContent = formatBytes(item.size);
    actions.appendChild(sizeEl);

    if (item.path && window.pywebview?.api?.reveal_file) {
      const revealBtn = document.createElement("button");
      revealBtn.type = "button";
      revealBtn.className = "btn btn-xs btn-outline-secondary py-0 px-1 reveal-file-btn";
      revealBtn.title = "Show in folder";
      revealBtn.innerHTML = '<i class="bi bi-folder2-open"></i>';
      revealBtn.addEventListener("click", () => revealSelectedFile(item));
      actions.appendChild(revealBtn);
    }

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn btn-xs btn-outline-danger py-0 px-1 remove-file-btn";
    removeBtn.title = "Remove file";
    removeBtn.innerHTML = '<i class="bi bi-x-lg"></i>';
    removeBtn.addEventListener("click", () => {
      state.selectedFiles = state.selectedFiles.filter((f) => f.key !== item.key);
      renderSelectedFiles();
    });
    actions.appendChild(removeBtn);

    row.append(openBtn, actions);
    list.appendChild(row);
  }
}

function addSelectedFiles(entries) {
  const existing = new Set(state.selectedFiles.map((f) => f.key));
  for (const entry of entries) {
    if (!existing.has(entry.key)) {
      state.selectedFiles.push(entry);
      existing.add(entry.key);
    }
  }
  renderSelectedFiles();
}

function addBrowserFiles(fileList) {
  const entries = Array.from(fileList || []).map((f) => {
    const rel = f.webkitRelativePath || f.name;
    return {
      key: `file:${rel}:${f.size}:${f.lastModified}`,
      name: f.name,
      rel,
      path: null,
      size: f.size,
      file: f,
    };
  });
  addSelectedFiles(entries);
}

function addNativeFiles(items) {
  const entries = (items || []).map((it) => ({
    key: `path:${it.path}`,
    name: it.name,
    rel: it.rel || it.name,
    path: it.path,
    size: it.size || 0,
    file: null,
  }));
  addSelectedFiles(entries);
}

window.__addNativeSelectedFiles = addNativeFiles;

function clearSelectedFiles() {
  state.selectedFiles = [];
  el("fileInput").value = "";
  el("folderInput").value = "";
  renderSelectedFiles();
}

const STATUS_BADGE_MAP = {
  waiting: "text-bg-info",
  queued: "text-bg-info",
  running: "text-bg-primary",
  completed: "text-bg-success",
  cancelled: "text-bg-warning",
  cancelling: "text-bg-warning",
  failed: "text-bg-danger",
};

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
    const isSelected = j.job_id === state.activeJobId;
    btn.className = `list-group-item list-group-item-action ${isSelected ? "active" : ""}`;
    const badgeColor = STATUS_BADGE_MAP[j.status] || "text-bg-secondary";
    const activeBorder = isSelected && j.status === "running" ? "border border-light border-opacity-50" : "";
    btn.innerHTML = `
      <div class="d-flex justify-content-between align-items-center">
        <span class="fw-semibold text-truncate me-2"></span>
        <span class="badge ${badgeColor} ${activeBorder}"></span>
      </div>
      <div class="small opacity-75">${j.model} · ${j.completed_docs}/${j.total_docs} docs</div>`;
    btn.querySelector("span").textContent = j.label;
    btn.querySelector(".badge").textContent = j.status;
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
  badge.className = `badge ms-2 ${STATUS_BADGE_MAP[job.status] || "text-bg-secondary"}`;
  badge.textContent = job.status;
  el("detailMeta").textContent = `Model: ${job.model} | DPI: ${job.dpi} | Output: ${job.output_dir}`;
  const etaText =
    job.status === "completed"
      ? "Done"
      : job.status === "waiting"
        ? "Waiting in queue..."
        : job.eta_seconds !== null && job.eta_seconds !== undefined
          ? `~${formatDuration(job.eta_seconds)}`
          : "Calculating...";
  el("detailTiming").textContent = `Elapsed: ${formatDuration(job.elapsed_seconds)} · ETA: ${etaText}`;

  const isActive = job.status === "waiting" || job.status === "queued" || job.status === "running";
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
  const anyRunning = allJobs.some((j) => !["completed", "failed", "cancelled"].includes(j.status));
  if (!anyRunning) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    fetchArtifacts()
      .then((d) => jobTagEditor.setSuggestions(d.recent_tags || (d.tags || []).map((t) => t.name)))
      .catch(() => {});
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
    jobTagEditor.setSuggestions(artData.recent_tags || (artData.tags || []).map((t) => t.name));
  } catch {
    // Ignore transient fetch failures
  }
}

const artifactsView = initArtifactsView({ showToast, onArchiveChanged: () => syncJobArchiveOptions() });
const jobTagEditor = createTagInput(el("jobTagsContainer"), { placeholder: "Add tag and press Enter..." });
const modelsView = initProvidersAndModelsView({ showToast, onProvidersChanged: () => syncSettingsToJobForm(true) });
const settingsWizard = initSettingsAndWizard({
  showToast,
  onSettingsSaved: () => syncSettingsToJobForm(true),
  onWizardFinished: () => switchView("jobsView"),
  onOpenModels: (paneId = "visualConfigPane") => {
    switchView("modelsView");
    modelsView.setPane(paneId);
  },
});
const updatesView = initUpdatesView({
  showToast,
  onOpenUpdateSection: () => {
    switchView("settingsView");
    settingsWizard.setSettingsPane("settingsUpdateCard");
  },
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
  if (target === "settingsView") {
    settingsWizard.loadSettingsForm().then((s) => updatesView.syncSettingsForm(s)).catch(() => {});
    if (!updatesView.getData()) updatesView.check({ silent: true });
  }
}

document.querySelectorAll("#mainNav [data-view]").forEach((btn) => {
  btn.addEventListener("click", () => switchView(btn.dataset.view));
});

el("navVisualBadge").addEventListener("click", () => {
  switchView("modelsView");
  modelsView.setPane("visualConfigPane");
});
el("navLlmBadge").addEventListener("click", () => {
  switchView("modelsView");
  modelsView.setPane("llmConfigPane");
});
el("navRefreshBtn").addEventListener("click", () => syncSettingsToJobForm(true));
el("quickSettingsBtn").addEventListener("click", () => {
  switchView("modelsView");
  modelsView.setPane("visualConfigPane");
});
el("quickStartOllamaBtn").addEventListener("click", async () => {
  const btn = el("quickStartOllamaBtn");
  btn.disabled = true;
  btn.textContent = "Starting...";
  el("ollamaLockMsg").textContent = "Starting Ollama Local service (this may take a few seconds)...";
  try {
    const res = await startOllama();
    showToast("Ollama Local", res.message, res.online ? "success" : "warning");
    await syncSettingsToJobForm(true);
  } catch (err) {
    showToast("Ollama Local", err.message, "danger");
  } finally {
    btn.disabled = false;
    btn.textContent = "Start Ollama";
  }
});

el("jobVisualModelSelect").addEventListener("change", refreshJobProviderState);
el("jobCustomCredSelect").addEventListener("change", refreshJobProviderState);
el("jobCustomModelInput").addEventListener("input", updatePullButton);
el("pullModelBtn").addEventListener("click", async () => {
  const choice = currentJobChoice();
  if (!choice.model) return;
  const btn = el("pullModelBtn");
  btn.disabled = true;
  btn.textContent = "Pulling...";
  try {
    await pullModel(choice.model, choice.credentialId);
    await syncSettingsToJobForm(true);
    showToast("Model Ready", `${choice.model} (${choice.credentialName}) installed.`, "success");
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
el("selectFilesBtn").addEventListener("click", async () => {
  if (window.pywebview?.api?.pick_files) {
    try {
      const items = await window.pywebview.api.pick_files();
      if (items?.length) addNativeFiles(items);
    } catch (err) {
      showToast("Select Files Failed", err?.message || String(err), "danger");
    }
    return;
  }
  el("fileInput").click();
});
el("selectFolderBtn").addEventListener("click", async () => {
  if (window.pywebview?.api?.pick_folder) {
    try {
      const items = await window.pywebview.api.pick_folder();
      if (items?.length) addNativeFiles(items);
    } catch (err) {
      showToast("Select Folder Failed", err?.message || String(err), "danger");
    }
    return;
  }
  el("folderInput").click();
});
el("fileInput").addEventListener("change", (e) => {
  if (e.target.files?.length) addBrowserFiles(e.target.files);
  e.target.value = "";
});
el("folderInput").addEventListener("change", (e) => {
  if (e.target.files?.length) addBrowserFiles(e.target.files);
  e.target.value = "";
});
el("clearSelectedFilesBtn").addEventListener("click", clearSelectedFiles);

const dz = el("dropZone");
dz.addEventListener("click", (e) => { if (!e.target.closest("label, input, button")) el("selectFilesBtn").click(); });
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag-over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("drag-over");
  if (!window.pywebview?.api?.pick_files && e.dataTransfer?.files?.length) {
    addBrowserFiles(e.dataTransfer.files);
  }
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
  const choice = currentJobChoice();
  if (!choice.model) {
    errEl.textContent = "Please select or enter a Visual OCR model.";
    errEl.classList.remove("d-none");
    return;
  }
  fd.append("credential_id", choice.credentialId);
  fd.append("provider", choice.provider);
  fd.append("model", choice.model);
  fd.append("dpi", el("dpiInput").value);
  fd.append("max_dim", el("maxDimInput").value);
  fd.append("series_mode", String(el("seriesSwitch").checked));
  fd.append("force", String(el("forceSwitch").checked));
  if (!el("jobAutoTag").disabled) {
    fd.append("auto_tag", String(el("jobAutoTag").checked));
    fd.append("auto_summarize", String(el("jobAutoSummarize").checked));
    fd.append("auto_rename", String(el("jobAutoRename").checked));
  }
  if (state.sourceMode === "upload") {
    const nativeEntries = [];
    for (const item of state.selectedFiles) {
      if (item.path) nativeEntries.push({ path: item.path, rel: item.rel });
      else if (item.file) fd.append("files", item.file, item.rel);
    }
    if (nativeEntries.length) fd.append("selected_paths", JSON.stringify(nativeEntries));
  }
  el("startBtn").disabled = true;
  try {
    const job = await startJob(fd);
    if (state.sourceMode === "upload") clearSelectedFiles();
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

function syncWindowFocus(focused) {
  window.pywebview?.api?.set_focus?.(focused);
}

window.addEventListener("focus", () => syncWindowFocus(true));
window.addEventListener("blur", () => syncWindowFocus(false));
window.addEventListener("pywebviewready", () => syncWindowFocus(document.hasFocus()));

(async () => {
  syncWindowFocus(document.hasFocus());
  const health = await fetchHealth();
  el("aboutVersion").textContent = `v${health.version}`;
  el("updateCurrentVersion").textContent = `v${health.version}`;
  el("aboutLicense").textContent = health.license;
  el("aboutCopyright").textContent = health.copyright;
  el("footerVersion").textContent = `OcrPipeline v${health.version}`;
  el("footerLicense").textContent = `${health.license} · ${health.copyright}`;

  await syncSettingsToJobForm();
  await syncJobArchiveOptions();
  renderJobHistory(await fetchJobs());
  if (!health.setup_completed) {
    await settingsWizard.openWizard();
  } else {
    fetchSettings()
      .then((s) => {
        updatesView.syncSettingsForm(s);
        if (s.auto_check_updates !== false) {
          updatesView.check({ silent: true });
        }
      })
      .catch(() => {});
  }
})();
