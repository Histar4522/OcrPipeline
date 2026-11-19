import {
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
} from "./api.js";

const el = (id) => document.getElementById(id);

const STEP_TITLES = [
  "Step 1 of 5 · Welcome & System Check",
  "Step 2 of 5 · Vision OCR Provider",
  "Step 3 of 5 · Vision Model & Optional LLM",
  "Step 4 of 5 · Archive, Presets & Desktop Behavior",
  "Step 5 of 5 · Review & Finish",
];

export function initSettingsAndWizard({ showToast, onSettingsSaved, onWizardFinished }) {
  const wizState = {
    step: 1,
    visionProvider: "ollama",
    selectedModel: "glm-ocr",
    llmEnabled: false,
    llmProvider: "ollama",
    llmModel: "qwen3:1.7b",
    dpi: 220,
    maxDim: 1536,
    providerOnline: false,
    modelsData: null,
    providersData: null,
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

  function syncWizardLlmFields() {
    const on = el("wizLlmEnabled").checked;
    wizState.llmEnabled = on;
    el("wizLlmFields").classList.toggle("opacity-50", !on);
    el("wizLlmProvider").disabled = !on;
    el("wizLlmModel").disabled = !on;
  }

  function syncWizardProviderUi() {
    const pid = el("wizVisionProvider").value || "ollama";
    wizState.visionProvider = pid;
    const spec = specFor(pid);
    const saved = wizState.providersData?.providers?.[pid] || {};
    el("wizOllamaUrl").value = saved.base_url || spec?.default_base_url || "";
    el("wizApiKey").value = saved.api_key || "";
    el("wizApiKeyRow").classList.toggle("d-none", !spec?.requires_api_key && pid !== "custom");
    if (pid !== "ollama") {
      el("wizStartOllamaBtn").classList.add("d-none");
      el("wizInstallGuide").classList.add("d-none");
    }
    const dl = el("wizVisionModelDatalist");
    dl.innerHTML = "";
    for (const m of spec?.suggested_vision_models || []) {
      const opt = document.createElement("option");
      opt.value = m;
      dl.appendChild(opt);
    }
  }

  async function loadSettingsForm() {
    const [s, provData] = await Promise.all([fetchSettings(), fetchProviders()]);
    wizState.providersData = provData;

    el("setOllamaUrl").value = s.ollama_url;
    el("setDefaultModel").value = s.vision_model || s.default_model;
    el("setArchiveDir").value = s.archive_dir;
    el("setDpi").value = s.dpi;
    el("setMaxDim").value = s.max_image_dim;
    el("setNumCtx").value = s.num_ctx;
    el("setTheme").value = s.theme || "auto";
    el("setSystemPrompt").value = s.system_prompt;
    el("setMinimizeTray").checked = Boolean(s.minimize_to_tray);
    el("setDesktopNotify").value = s.desktop_notifications || "always";
    el("setObsidianMode").checked = Boolean(s.obsidian_mode);

    wizState.visionProvider = s.vision_provider || "ollama";
    wizState.selectedModel = s.vision_model || s.default_model || "glm-ocr";
    wizState.llmEnabled = Boolean(s.llm_enabled);
    wizState.llmProvider = s.llm_provider || "ollama";
    wizState.llmModel = s.llm_model || "qwen3:1.7b";
    wizState.dpi = s.dpi || 220;
    wizState.maxDim = s.max_image_dim || 1536;

    const vSel = el("wizVisionProvider");
    const lSel = el("wizLlmProvider");
    vSel.innerHTML = "";
    lSel.innerHTML = "";
    for (const spec of provData.registry || []) {
      if (spec.supports_vision) vSel.add(new Option(spec.label, spec.id));
      if (spec.supports_llm) lSel.add(new Option(spec.label, spec.id));
    }
    vSel.value = wizState.visionProvider;
    lSel.value = wizState.llmProvider;
    el("wizVisionModel").value = wizState.selectedModel;
    el("wizLlmEnabled").checked = wizState.llmEnabled;
    el("wizLlmModel").value = wizState.llmModel;
    syncWizardProviderUi();
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
    const pid = el("wizVisionProvider").value || "ollama";
    const alertBox = el("wizOllamaAlert");
    const startBtn = el("wizStartOllamaBtn");
    const installGuide = el("wizInstallGuide");

    if (pid !== "ollama") {
      startBtn.classList.add("d-none");
      installGuide.classList.add("d-none");
      const res = await testProvider({
        provider: pid,
        base_url: el("wizOllamaUrl").value.trim(),
        api_key: el("wizApiKey").value.trim(),
      });
      wizState.providerOnline = Boolean(res.online);
      alertBox.className = `alert small mb-3 ${res.online ? "alert-success" : "alert-warning"}`;
      alertBox.textContent = res.message;
      const dl = el("wizVisionModelDatalist");
      const spec = specFor(pid);
      dl.innerHTML = "";
      for (const m of [...new Set([...(res.models || []), ...(spec?.suggested_vision_models || [])])]) {
        const opt = document.createElement("option");
        opt.value = m;
        dl.appendChild(opt);
      }
      renderModelCards();
      return;
    }

    let data = await fetchModels();
    if (!data.online && data.installed && autoLaunch) {
      alertBox.className = "alert alert-info small mb-3";
      alertBox.textContent = "Local Ollama binary detected. Launching background Ollama service...";
      startBtn.classList.add("d-none");
      try {
        const res = await startOllama();
        if (res.online) await onSettingsSaved();
      } catch {
        // Re-check below
      }
      data = await fetchModels();
    }

    wizState.modelsData = data;
    wizState.providerOnline = Boolean(data.online);

    if (data.online) {
      alertBox.className = "alert alert-success small mb-3";
      alertBox.textContent = `Connected to ${data.ollama_url} — ${data.models.length} model(s) installed.`;
      startBtn.classList.add("d-none");
      installGuide.classList.add("d-none");
    } else if (data.installed) {
      alertBox.className = "alert alert-warning small mb-3";
      alertBox.textContent = `Ollama is installed, but not responding at ${data.ollama_url}. Click Start Ollama or switch provider above.`;
      startBtn.classList.remove("d-none");
      installGuide.classList.add("d-none");
    } else {
      alertBox.className = "alert alert-warning small mb-3";
      alertBox.textContent = `Ollama was not found at ${data.ollama_url}. Install Ollama below or choose another Vision provider (llama.cpp / OpenAI / Anthropic / Google).`;
      startBtn.classList.add("d-none");
      installGuide.classList.remove("d-none");
    }
    renderModelCards();
  }

  function renderModelCards() {
    const pid = wizState.visionProvider;
    const container = el("wizModelCards");
    container.innerHTML = "";
    el("wizVisionModel").value = wizState.selectedModel;

    if (pid !== "ollama") {
      const spec = specFor(pid);
      for (const name of spec?.suggested_vision_models || []) {
        const isSelected = wizState.selectedModel === name;
        const col = document.createElement("div");
        col.className = "col-md-6";
        col.innerHTML = `
          <div class="border rounded p-2 wizard-choice-card ${isSelected ? "selected" : ""} d-flex justify-content-between align-items-center">
            <span class="fw-semibold small"></span>
            <button type="button" class="btn btn-xs ${isSelected ? "btn-primary" : "btn-outline-primary"}">${isSelected ? "Selected" : "Select"}</button>
          </div>`;
        col.querySelector("span").textContent = name;
        col.querySelector("button").addEventListener("click", () => {
          wizState.selectedModel = name;
          el("wizVisionModel").value = name;
          renderModelCards();
        });
        container.appendChild(col);
      }
      return;
    }

    const data = wizState.modelsData;
    if (!data) return;
    const installed = new Set(data.models || []);
    for (const rec of data.recommended || []) {
      const isPulled = installed.has(rec.name) || installed.has(`${rec.name}:latest`);
      const isSelected = wizState.selectedModel === rec.name || wizState.selectedModel === `${rec.name}:latest`;
      const col = document.createElement("div");
      col.className = "col-md-6";
      col.innerHTML = `
        <div class="border rounded p-3 h-100 wizard-choice-card ${isSelected ? "selected" : ""}">
          <div class="d-flex justify-content-between align-items-start gap-2 mb-1">
            <div>
              <span class="fw-semibold">${rec.name}</span>
              <span class="badge bg-body-secondary text-body border ms-1">${rec.size}</span>
              <span class="badge text-bg-info ms-1">${rec.vram}</span>
            </div>
            <span class="badge ${isPulled ? "text-bg-success" : "text-bg-secondary"}">${isPulled ? "Installed" : "Not Pulled"}</span>
          </div>
          <p class="small text-muted mb-2">${rec.description}</p>
          <div class="d-flex gap-2">
            <button type="button" class="btn btn-xs ${isSelected ? "btn-primary" : "btn-outline-primary"} select-btn">
              ${isSelected ? "Default Selected" : "Use as Default"}
            </button>
            ${!isPulled ? `<button type="button" class="btn btn-xs btn-outline-success pull-btn" ${data.online ? "" : "disabled"}>Download</button>` : ""}
          </div>
        </div>`;

      col.querySelector(".select-btn").addEventListener("click", () => {
        wizState.selectedModel = rec.name;
        el("wizVisionModel").value = rec.name;
        renderModelCards();
      });
      const pullBtn = col.querySelector(".pull-btn");
      if (pullBtn) {
        pullBtn.addEventListener("click", () => beginModelPull(rec.name));
      }
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
      await startModelPull(modelName);
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
            wizState.selectedModel = modelName;
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
    const notifyVal = notifyLabels[el("wizNotify").value] || "Always";
    const llmSummary = el("wizLlmEnabled").checked
      ? `${el("wizLlmProvider").value} / ${el("wizLlmModel").value.trim()}`
      : "Disabled (Vision OCR only)";
    const items = [
      ["Vision Provider", `${el("wizVisionProvider").value} (${el("wizOllamaUrl").value.trim()})`],
      ["Vision OCR Model", el("wizVisionModel").value.trim() || wizState.selectedModel],
      ["Assistant LLM", llmSummary],
      ["Quality Preset", `${wizState.dpi} DPI · ${wizState.maxDim} px max dimension`],
      ["Archive Directory", el("wizArchiveDir").value.trim()],
      ["Obsidian Vault Mode", el("wizObsidian").checked ? "Enabled (.obsidian + YAML frontmatter)" : "Disabled"],
      ["Desktop Integration", `${el("wizTray").checked ? "Minimize to tray" : "Exit on close"} · OS toasts: ${notifyVal}`],
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

    if (wizState.step === 2) refreshProviderStep(el("wizVisionProvider").value === "ollama");
    if (wizState.step === 3) renderModelCards();
    if (wizState.step === 5) renderSummaryStep();
  }

  async function openWizard() {
    toggleWizardShell(true);
    goToStep(1);
    await Promise.all([loadSettingsForm(), loadSystemStep()]);
    await refreshProviderStep(el("wizVisionProvider").value === "ollama");
  }

  function closeWizard() {
    stopPullPolling();
    toggleWizardShell(false);
  }

  el("openWizardBtn").addEventListener("click", openWizard);
  el("wizExitBtn").addEventListener("click", closeWizard);
  el("wizBackBtn").addEventListener("click", () => goToStep(wizState.step - 1));
  el("wizNextBtn").addEventListener("click", () => goToStep(wizState.step + 1));
  document.querySelectorAll("#wizStepNav [data-wiz-step]").forEach((btn) => {
    btn.addEventListener("click", () => goToStep(Number(btn.dataset.wizStep)));
  });

  el("wizVisionProvider").addEventListener("change", () => {
    syncWizardProviderUi();
    const spec = specFor(el("wizVisionProvider").value);
    if (spec?.suggested_vision_models?.length) {
      wizState.selectedModel = spec.suggested_vision_models[0];
      el("wizVisionModel").value = wizState.selectedModel;
    }
    refreshProviderStep(false);
  });
  el("wizVisionModel").addEventListener("input", (e) => {
    wizState.selectedModel = e.target.value.trim();
  });
  el("wizLlmEnabled").addEventListener("change", syncWizardLlmFields);

  el("wizRecheckBtn").addEventListener("click", async () => {
    const pid = el("wizVisionProvider").value || "ollama";
    await saveSettings({
      vision_provider: pid,
      providers: {
        [pid]: {
          base_url: el("wizOllamaUrl").value.trim(),
          api_key: el("wizApiKey").value.trim(),
        },
      },
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
      if (!res.online) showToast("Ollama", res.message, "warning");
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
    const pid = el("wizVisionProvider").value || "ollama";
    const vModel = el("wizVisionModel").value.trim() || wizState.selectedModel || "glm-ocr";
    await saveSettings({
      setup_completed: true,
      vision_provider: pid,
      vision_model: vModel,
      providers: {
        [pid]: {
          base_url: el("wizOllamaUrl").value.trim(),
          api_key: el("wizApiKey").value.trim(),
        },
      },
      llm_enabled: el("wizLlmEnabled").checked,
      llm_provider: el("wizLlmProvider").value || "ollama",
      llm_model: el("wizLlmModel").value.trim(),
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
      ollama_url: el("setOllamaUrl").value.trim(),
      default_model: el("setDefaultModel").value.trim(),
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
