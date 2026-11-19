import {
  createCredential,
  fetchModelPullStatus,
  fetchModels,
  fetchProviders,
  fetchSettings,
  fetchSystemInfo,
  openExternalUrl,
  revealArtifact,
  saveSettings,
  startModelPull,
  startOllama,
  testProvider,
  updateCredential,
} from "./api.js";
import {
  CUSTOM_MODEL_VALUE,
  encodeModelChoice,
  populateRoleModelSelect,
  readRoleModelChoice,
} from "./model_dropdown.js";

const el = (id) => document.getElementById(id);
const LOCAL_CRED_ID = "cred_ollama_local";

const STEP_TITLES = [
  "Step 1 of 5 · Welcome & System Check",
  "Step 2 of 5 · Providers & Credentials",
  "Step 3 of 5 · Visual Model & Optional LLM",
  "Step 4 of 5 · Archive, Presets & Desktop Behavior",
  "Step 5 of 5 · Review & Finish",
];

export function initSettingsAndWizard({ showToast, onSettingsSaved, onWizardFinished, onOpenModels }) {
  const wizState = {
    step: 1,
    dpi: 220,
    maxDim: 1536,
    modelsData: null,
    providersData: null,
    credentials: [],
    discoveredByCred: {},
    pullTimer: null,
  };

  function toggleWizardShell(open) {
    el("mainNavbar").classList.toggle("d-none", open);
    el("mainContent").classList.toggle("d-none", open);
    el("mainFooter").classList.toggle("d-none", open);
    el("wizardView").classList.toggle("d-none", !open);
  }

  function stopPullPolling() {
    if (wizState.pullTimer) {
      clearInterval(wizState.pullTimer);
      wizState.pullTimer = null;
    }
  }

  function specFor(pid) {
    return (wizState.providersData?.registry || []).find((r) => r.id === pid) || null;
  }

  function suggestUniqueCredName(pid) {
    const spec = specFor(pid);
    const base = spec?.short_label || spec?.label || pid;
    const taken = new Set(wizState.credentials.map((c) => c.name.toLowerCase()));
    if (!taken.has(base.toLowerCase())) return base;
    let i = 2;
    while (taken.has(`${base} ${i}`.toLowerCase())) i += 1;
    return `${base} ${i}`;
  }

  function syncCloudProviderFields() {
    const pid = el("wizVisionProvider").value || "deepseek";
    const spec = specFor(pid);
    el("wizCloudBaseUrl").value = spec?.default_base_url || "";
    el("wizCredName").value = suggestUniqueCredName(pid);
    el("wizApiKeyRow").classList.toggle("d-none", !spec?.requires_api_key && pid !== "custom");
  }

  function syncWizardLlmFields() {
    const on = el("wizLlmEnabled").checked;
    el("wizLlmFields").classList.toggle("opacity-50", !on);
    el("wizLlmModelSelect").disabled = !on;
    el("wizLlmCustomCred").disabled = !on;
    el("wizLlmCustomModel").disabled = !on;
  }

  function renderWizardRoleDropdowns() {
    const registry = wizState.providersData?.registry || [];
    const vis = wizState.providersData?.vision || {};
    const llm = wizState.providersData?.llm || {};

    populateRoleModelSelect({
      selectEl: el("wizVisualModelSelect"),
      customPanelEl: el("wizVisualCustomPanel"),
      customCredEl: el("wizVisualCustomCred"),
      customInputEl: el("wizVisualCustomModel"),
      credentials: wizState.credentials,
      registry,
      discoveredByCred: wizState.discoveredByCred,
      role: "vision",
      current: {
        credentialId: vis.credential_id || LOCAL_CRED_ID,
        model: vis.model || "glm-ocr",
        custom: Boolean(vis.custom),
      },
    });

    populateRoleModelSelect({
      selectEl: el("wizLlmModelSelect"),
      customPanelEl: el("wizLlmCustomPanel"),
      customCredEl: el("wizLlmCustomCred"),
      customInputEl: el("wizLlmCustomModel"),
      credentials: wizState.credentials,
      registry,
      discoveredByCred: wizState.discoveredByCred,
      role: "llm",
      current: {
        credentialId: llm.credential_id || LOCAL_CRED_ID,
        model: llm.model || "qwen3:1.7b",
        custom: Boolean(llm.custom),
      },
    });
  }

  async function loadSettingsForm() {
    const [s, provData] = await Promise.all([fetchSettings(), fetchProviders()]);
    wizState.providersData = provData;
    wizState.credentials = provData.credentials || [];

    const visLabel = provData.vision?.display_label || `${s.vision_model}`;
    const llmLabel = provData.llm?.display_label || "Disabled";
    el("settingsModelSummaryBox").innerHTML =
      `<strong>Visual:</strong> ${visLabel} &nbsp;·&nbsp; <strong>LLM:</strong> ${llmLabel}`;

    el("setArchiveDir").value = s.archive_dir;
    el("setDpi").value = s.dpi;
    el("setMaxDim").value = s.max_image_dim;
    el("setNumCtx").value = s.num_ctx;
    el("setTheme").value = s.theme || "auto";
    el("setSystemPrompt").value = s.system_prompt;
    el("setMinimizeTray").checked = Boolean(s.minimize_to_tray);
    el("setDesktopNotify").value = s.desktop_notifications || "always";
    el("setObsidianMode").checked = Boolean(s.obsidian_mode);

    wizState.dpi = s.dpi || 220;
    wizState.maxDim = s.max_image_dim || 1536;

    const localCred = wizState.credentials.find((c) => c.id === LOCAL_CRED_ID);
    el("wizUseLocalOllama").checked = localCred ? localCred.enabled !== false : true;
    el("wizOllamaUrl").value = localCred?.base_url || s.ollama_url || "http://localhost:11434";
    el("wizLocalOllamaFields").classList.toggle("d-none", !el("wizUseLocalOllama").checked);

    const pSel = el("wizVisionProvider");
    pSel.innerHTML = "";
    for (const spec of provData.registry || []) {
      if (spec.singleton || spec.id === "ollama_local") continue;
      pSel.add(new Option(spec.label, spec.id));
    }
    pSel.value = "deepseek";
    syncCloudProviderFields();

    el("wizLlmEnabled").checked = Boolean(s.llm_enabled);
    renderWizardRoleDropdowns();
    syncWizardLlmFields();

    el("wizArchiveDir").value = s.archive_dir;
    el("wizTray").checked = Boolean(s.minimize_to_tray);
    el("wizNotify").value = s.desktop_notifications || "always";
    el("wizObsidian").checked = Boolean(s.obsidian_mode);
    syncPresetCards();
    return s;
  }

  function syncPresetCards() {
    document.querySelectorAll("#wizPresetCards [data-preset]").forEach((card) => {
      const match = Number(card.dataset.dpi) === wizState.dpi && Number(card.dataset.dim) === wizState.maxDim;
      card.classList.toggle("selected", match);
    });
  }

  async function loadSystemStep() {
    try {
      const info = await fetchSystemInfo();
      const gpuLabel = info.gpu_name ? `${info.gpu_name} (${info.vram_gb} GB VRAM)` : "CPU / Integrated";
      el("wizSystemGrid").innerHTML = `
        <div class="col-md-4"><strong>OS:</strong> ${info.os}</div>
        <div class="col-md-4"><strong>CPU / RAM:</strong> ${info.cpu_count} cores · ${info.ram_gb} GB</div>
        <div class="col-md-4"><strong>GPU:</strong> ${gpuLabel}</div>`;
      el("wizVramNote").textContent = info.vram_note;
    } catch {
      el("wizSystemGrid").innerHTML = '<div class="col-12 text-muted">System check unavailable.</div>';
    }
  }

  async function refreshProviderStep(autoLaunch = false) {
    const enabled = el("wizUseLocalOllama").checked;
    const alertBox = el("wizOllamaAlert");
    const startBtn = el("wizStartOllamaBtn");
    const installGuide = el("wizInstallGuide");

    if (!enabled) {
      alertBox.className = "alert alert-secondary py-2 px-3 small mb-0";
      alertBox.textContent = "Ollama Local disabled. Add a cloud or remote credential below to continue.";
      startBtn.classList.add("d-none");
      installGuide.classList.add("d-none");
      renderModelCards();
      return;
    }

    let data = await fetchModels(LOCAL_CRED_ID);
    if (!data.online && data.installed && autoLaunch) {
      alertBox.className = "alert alert-info py-2 px-3 small mb-0";
      alertBox.textContent = "Local Ollama binary detected. Launching background service...";
      startBtn.classList.add("d-none");
      try {
        const res = await startOllama();
        if (res.online) await onSettingsSaved();
      } catch {
        // Re-check below
      }
      data = await fetchModels(LOCAL_CRED_ID);
    }

    wizState.modelsData = data;
    if (data.models?.length) {
      wizState.discoveredByCred[LOCAL_CRED_ID] = data.models;
    }

    if (data.online) {
      alertBox.className = "alert alert-success py-2 px-3 small mb-0";
      alertBox.textContent = `Connected to ${data.ollama_url} — ${data.models.length} local model(s) installed.`;
      startBtn.classList.add("d-none");
      installGuide.classList.add("d-none");
    } else if (data.installed) {
      alertBox.className = "alert alert-warning py-2 px-3 small mb-0";
      alertBox.textContent = `Ollama is installed, but offline at ${data.ollama_url}. Click Start Ollama or add a cloud provider below.`;
      startBtn.classList.remove("d-none");
      installGuide.classList.add("d-none");
    } else {
      alertBox.className = "alert alert-warning py-2 px-3 small mb-0";
      alertBox.textContent = `Ollama Local was not found at ${data.ollama_url}. Install Ollama below or add a cloud credential.`;
      startBtn.classList.add("d-none");
      installGuide.classList.remove("d-none");
    }
    renderWizardRoleDropdowns();
    renderModelCards();
  }

  function renderModelCards() {
    const container = el("wizModelCards");
    container.innerHTML = "";
    const localCred = wizState.credentials.find((c) => c.id === LOCAL_CRED_ID);
    if (!localCred || localCred.enabled === false) return;

    const data = wizState.modelsData;
    if (!data) return;
    const installed = new Set(data.models || []);
    const activeChoice = readRoleModelChoice({
      selectEl: el("wizVisualModelSelect"),
      customCredEl: el("wizVisualCustomCred"),
      customInputEl: el("wizVisualCustomModel"),
      credentials: wizState.credentials,
    });

    for (const rec of data.recommended || []) {
      const isPulled = installed.has(rec.name) || installed.has(`${rec.name}:latest`);
      const isSelected =
        activeChoice.credentialId === LOCAL_CRED_ID &&
        (activeChoice.model === rec.name || activeChoice.model === `${rec.name}:latest`);
      const col = document.createElement("div");
      col.className = "col-md-6";
      col.innerHTML = `
        <div class="border rounded p-3 h-100 wizard-choice-card ${isSelected ? "selected" : ""}">
          <div class="d-flex justify-content-between align-items-start gap-2 mb-1">
            <div>
              <span class="fw-semibold">${rec.name} (Ollama Local)</span>
              <span class="badge bg-body-secondary text-body border ms-1">${rec.size}</span>
              <span class="badge text-bg-info ms-1">${rec.vram}</span>
            </div>
            <span class="badge ${isPulled ? "text-bg-success" : "text-bg-secondary"}">${isPulled ? "Installed" : "Not Pulled"}</span>
          </div>
          <p class="small text-muted mb-2">${rec.description}</p>
          <div class="d-flex gap-2">
            <button type="button" class="btn btn-xs ${isSelected ? "btn-primary" : "btn-outline-primary"} select-btn">
              ${isSelected ? "Selected" : "Use for Visual"}
            </button>
            ${!isPulled ? `<button type="button" class="btn btn-xs btn-outline-success pull-btn" ${data.online ? "" : "disabled"}>Download</button>` : ""}
          </div>
        </div>`;

      col.querySelector(".select-btn").addEventListener("click", () => {
        const key = encodeModelChoice(LOCAL_CRED_ID, rec.name);
        const sel = el("wizVisualModelSelect");
        if ([...sel.options].some((o) => o.value === key)) {
          sel.value = key;
          el("wizVisualCustomPanel").classList.add("d-none");
        } else {
          sel.value = CUSTOM_MODEL_VALUE;
          el("wizVisualCustomPanel").classList.remove("d-none");
          el("wizVisualCustomCred").value = LOCAL_CRED_ID;
          el("wizVisualCustomModel").value = rec.name;
        }
        renderModelCards();
      });
      const pullBtn = col.querySelector(".pull-btn");
      if (pullBtn) pullBtn.addEventListener("click", () => beginModelPull(rec.name));
      container.appendChild(col);
    }
  }

  async function beginModelPull(modelName) {
    const panel = el("wizPullPanel");
    const statusText = el("wizPullStatusText");
    const pctText = el("wizPullPercent");
    const bar = el("wizPullBar");
    panel.classList.remove("d-none");
    bar.className = "progress-bar progress-bar-striped progress-bar-animated";
    statusText.textContent = `Starting download for ${modelName}...`;
    pctText.textContent = "0%";
    bar.style.width = "0%";

    try {
      await startModelPull(modelName, LOCAL_CRED_ID);
    } catch (err) {
      showToast("Pull Error", err.message, "danger");
      return;
    }

    stopPullPolling();
    wizState.pullTimer = setInterval(async () => {
      try {
        const st = await fetchModelPullStatus();
        const pct = Math.min(100, Math.round(st.percent || 0));
        const sizeNote = st.total_mb > 0 ? ` (${st.completed_mb} / ${st.total_mb} MB)` : "";
        statusText.textContent = `${st.model}: ${st.status}${sizeNote}`;
        pctText.textContent = `${pct}%`;
        bar.style.width = `${pct}%`;

        if (st.done) {
          stopPullPolling();
          if (st.error) {
            bar.className = "progress-bar bg-danger";
            statusText.textContent = st.error;
            showToast("Download Failed", st.error, "danger");
          } else {
            bar.className = "progress-bar bg-success";
            await refreshProviderStep(false);
            await onSettingsSaved();
            showToast("Model Ready", `${modelName} installed.`, "success");
          }
        }
      } catch {
        stopPullPolling();
      }
    }, 600);
  }

  function renderSummaryStep() {
    const notifyLabels = {
      always: "Always",
      not_in_focus: "Not in focus",
      minimized: "Minimized to tray",
      never: "Never",
    };
    const vis = readRoleModelChoice({
      selectEl: el("wizVisualModelSelect"),
      customCredEl: el("wizVisualCustomCred"),
      customInputEl: el("wizVisualCustomModel"),
      credentials: wizState.credentials,
    });
    const llm = readRoleModelChoice({
      selectEl: el("wizLlmModelSelect"),
      customCredEl: el("wizLlmCustomCred"),
      customInputEl: el("wizLlmCustomModel"),
      credentials: wizState.credentials,
    });
    const llmSummary = el("wizLlmEnabled").checked
      ? `${llm.model || "(unset)"} (${llm.credentialName || "Provider"})`
      : "Disabled";
    const items = [
      ["Visual Model", `${vis.model || "glm-ocr"} (${vis.credentialName || "Ollama Local"})`],
      ["LLM Model", llmSummary],
      ["Configured Credentials", wizState.credentials.filter((c) => c.enabled !== false).map((c) => c.name).join(", ") || "None"],
      ["Quality Preset", `${wizState.dpi} DPI · ${wizState.maxDim} px max dimension`],
      ["Archive Directory", el("wizArchiveDir").value.trim()],
      ["Obsidian Vault Mode", el("wizObsidian").checked ? "Enabled (.obsidian + frontmatter)" : "Disabled"],
      ["Desktop Integration", `${el("wizTray").checked ? "Minimize to tray" : "Exit on close"} · Toasts: ${notifyLabels[el("wizNotify").value] || "Always"}`],
    ];
    el("wizSummaryList").innerHTML = items
      .map(([k, v]) => `<li class="list-group-item d-flex justify-content-between"><span class="fw-semibold">${k}</span><span class="text-muted text-end">${v}</span></li>`)
      .join("");
  }

  function goToStep(stepNum) {
    wizState.step = Math.max(1, Math.min(5, stepNum));
    el("wizStepSubtitle").textContent = STEP_TITLES[wizState.step - 1];
    el("wizProgressBar").style.width = `${wizState.step * 20}%`;

    document.querySelectorAll("#wizStepNav [data-wiz-step]").forEach((btn) => {
      btn.classList.toggle("active", Number(btn.dataset.wizStep) === wizState.step);
    });
    document.querySelectorAll("#wizardView .wiz-step").forEach((pane) => {
      pane.classList.toggle("d-none", Number(pane.dataset.step) !== wizState.step);
    });

    el("wizBackBtn").disabled = wizState.step === 1;
    el("wizNextBtn").classList.toggle("d-none", wizState.step === 5);
    el("wizFinishBtn").classList.toggle("d-none", wizState.step !== 5);

    if (wizState.step === 2) refreshProviderStep(el("wizUseLocalOllama").checked);
    if (wizState.step === 3) {
      renderWizardRoleDropdowns();
      renderModelCards();
    }
    if (wizState.step === 5) renderSummaryStep();
  }

  async function openWizard() {
    toggleWizardShell(true);
    goToStep(1);
    await Promise.all([loadSettingsForm(), loadSystemStep()]);
    await refreshProviderStep(el("wizUseLocalOllama").checked);
  }

  function closeWizard() {
    stopPullPolling();
    toggleWizardShell(false);
  }

  el("openWizardBtn").addEventListener("click", openWizard);
  el("settingsOpenModelsBtn").addEventListener("click", () => {
    if (onOpenModels) onOpenModels("visualConfigPane");
  });
  el("wizExitBtn").addEventListener("click", closeWizard);
  el("wizBackBtn").addEventListener("click", () => goToStep(wizState.step - 1));
  el("wizNextBtn").addEventListener("click", () => goToStep(wizState.step + 1));
  document.querySelectorAll("#wizStepNav [data-wiz-step]").forEach((btn) => {
    btn.addEventListener("click", () => goToStep(Number(btn.dataset.wizStep)));
  });

  el("wizUseLocalOllama").addEventListener("change", async (e) => {
    el("wizLocalOllamaFields").classList.toggle("d-none", !e.target.checked);
    await updateCredential(LOCAL_CRED_ID, {
      provider: "ollama_local",
      enabled: e.target.checked,
      base_url: el("wizOllamaUrl").value.trim(),
    });
    await loadSettingsForm();
    await refreshProviderStep(false);
  });

  el("wizAddCloudCred").addEventListener("change", (e) => {
    el("wizCloudCredFields").classList.toggle("d-none", !e.target.checked);
  });
  el("wizVisionProvider").addEventListener("change", syncCloudProviderFields);

  el("wizSaveCloudCredBtn").addEventListener("click", async () => {
    const pid = el("wizVisionProvider").value;
    const name = el("wizCredName").value.trim();
    const base_url = el("wizCloudBaseUrl").value.trim();
    const api_key = el("wizApiKey").value.trim();
    const msg = el("wizCloudCredMsg");
    msg.className = "text-muted";
    msg.textContent = "Testing & saving...";
    try {
      const probe = await testProvider({ provider: pid, base_url, api_key });
      const created = await createCredential({ name, provider: pid, base_url, api_key });
      if (probe.online && probe.models?.length) {
        wizState.discoveredByCred[created.id] = probe.models;
      }
      await loadSettingsForm();
      msg.className = probe.online ? "text-success" : "text-warning";
      msg.textContent = `${created.name} added (${probe.message})`;
      el("wizAddCloudCred").checked = false;
      el("wizCloudCredFields").classList.add("d-none");
    } catch (err) {
      msg.className = "text-danger";
      msg.textContent = err.message;
    }
  });

  el("wizVisualModelSelect").addEventListener("change", () => {
    el("wizVisualCustomPanel").classList.toggle("d-none", el("wizVisualModelSelect").value !== CUSTOM_MODEL_VALUE);
    renderModelCards();
  });
  el("wizLlmEnabled").addEventListener("change", syncWizardLlmFields);
  el("wizLlmModelSelect").addEventListener("change", () => {
    el("wizLlmCustomPanel").classList.toggle("d-none", el("wizLlmModelSelect").value !== CUSTOM_MODEL_VALUE);
  });

  el("wizRecheckBtn").addEventListener("click", async () => {
    await updateCredential(LOCAL_CRED_ID, {
      provider: "ollama_local",
      enabled: el("wizUseLocalOllama").checked,
      base_url: el("wizOllamaUrl").value.trim(),
    });
    await refreshProviderStep(false);
    await onSettingsSaved();
  });

  el("wizStartOllamaBtn").addEventListener("click", async () => {
    const btn = el("wizStartOllamaBtn");
    btn.disabled = true;
    btn.textContent = "Starting...";
    try {
      const res = await startOllama();
      if (!res.online) showToast("Ollama Local", res.message, "warning");
      await onSettingsSaved();
    } finally {
      btn.disabled = false;
      btn.textContent = "Start Ollama";
      await refreshProviderStep(false);
    }
  });

  el("wizDownloadOllamaBtn").addEventListener("click", () => openExternalUrl("https://ollama.com/download"));
  el("wizOpenArchiveBtn").addEventListener("click", () => revealArtifact(""));

  document.querySelectorAll("#wizPresetCards [data-preset]").forEach((card) => {
    card.addEventListener("click", () => {
      wizState.dpi = Number(card.dataset.dpi);
      wizState.maxDim = Number(card.dataset.dim);
      syncPresetCards();
    });
  });

  el("wizFinishBtn").addEventListener("click", async () => {
    await updateCredential(LOCAL_CRED_ID, {
      provider: "ollama_local",
      enabled: el("wizUseLocalOllama").checked,
      base_url: el("wizOllamaUrl").value.trim(),
    });
    const vis = readRoleModelChoice({
      selectEl: el("wizVisualModelSelect"),
      customCredEl: el("wizVisualCustomCred"),
      customInputEl: el("wizVisualCustomModel"),
      credentials: wizState.credentials,
    });
    const llm = readRoleModelChoice({
      selectEl: el("wizLlmModelSelect"),
      customCredEl: el("wizLlmCustomCred"),
      customInputEl: el("wizLlmCustomModel"),
      credentials: wizState.credentials,
    });
    await saveSettings({
      setup_completed: true,
      vision_credential_id: vis.credentialId || LOCAL_CRED_ID,
      vision_provider: vis.provider || "ollama_local",
      vision_model: vis.model || "glm-ocr",
      vision_custom_model: vis.custom,
      llm_enabled: el("wizLlmEnabled").checked,
      llm_credential_id: llm.credentialId || LOCAL_CRED_ID,
      llm_provider: llm.provider || "ollama_local",
      llm_model: llm.model,
      llm_custom_model: llm.custom,
      dpi: wizState.dpi,
      max_image_dim: wizState.maxDim,
      archive_dir: el("wizArchiveDir").value.trim(),
      minimize_to_tray: el("wizTray").checked,
      desktop_notifications: el("wizNotify").value,
      obsidian_mode: el("wizObsidian").checked,
    });
    closeWizard();
    await loadSettingsForm();
    await onSettingsSaved();
    if (onWizardFinished) onWizardFinished();
    showToast("Setup Complete", "OcrPipeline is configured and ready.", "success");
  });

  el("resetPromptBtn").addEventListener("click", async () => {
    const updated = await saveSettings({ reset_prompt: true });
    el("setSystemPrompt").value = updated.system_prompt;
    showToast("Prompt Reset", "Restored default archival OCR prompt.");
  });

  el("settingsForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const saved = await saveSettings({
      archive_dir: el("setArchiveDir").value.trim(),
      dpi: Number(el("setDpi").value),
      max_image_dim: Number(el("setMaxDim").value),
      num_ctx: Number(el("setNumCtx").value),
      theme: el("setTheme").value,
      system_prompt: el("setSystemPrompt").value,
      minimize_to_tray: el("setMinimizeTray").checked,
      desktop_notifications: el("setDesktopNotify").value,
      obsidian_mode: el("setObsidianMode").checked,
    });
    el("settingsSaveMsg").textContent = "Saved!";
    setTimeout(() => (el("settingsSaveMsg").textContent = ""), 2500);
    await onSettingsSaved(saved);
    showToast("Settings Saved", "Configuration updated.", "success");
  });

  return { loadSettingsForm, openWizard };
}
