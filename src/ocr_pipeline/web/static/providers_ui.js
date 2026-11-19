import {
  createCredential,
  deleteCredential,
  deleteModel,
  estimateLlmVram,
  fetchModels,
  fetchProviders,
  fetchProviderStatus,
  fetchSettings,
  pullModel,
  saveSettings,
  startOllama,
  testProvider,
  updateCredential,
} from "./api.js";
import { bsConfirm } from "./dialogs.js";
import {
  CUSTOM_MODEL_VALUE,
  populateRoleModelSelect,
  readRoleModelChoice,
} from "./model_dropdown.js";

const el = (id) => document.getElementById(id);
const LOCAL_CRED_ID = "cred_ollama_local";

export function initProvidersAndModelsView({ showToast, onProvidersChanged }) {
  const pState = {
    registry: [],
    credentials: [],
    settings: null,
    statusData: null,
    localModelsData: null,
    discoveredByCred: {},
    activePane: "visualConfigPane",
  };

  function specFor(pid) {
    return pState.registry.find((r) => r.id === pid) || null;
  }

  function credFor(cid) {
    return pState.credentials.find((c) => c.id === cid) || null;
  }

  function setPane(paneId) {
    pState.activePane = paneId || "visualConfigPane";
    document.querySelectorAll("#modelsView [data-pref-pane]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.prefPane === pState.activePane);
    });
    document.querySelectorAll("#modelsView .pref-pane").forEach((pane) => {
      pane.classList.toggle("d-none", pane.id !== pState.activePane);
    });
  }

  function statusBadgeClass(status) {
    if (status === "ok") return "badge text-bg-success";
    if (status === "connecting") return "badge text-bg-info";
    if (["no_key", "no_url", "model_missing", "disabled"].includes(status)) return "badge text-bg-warning text-dark";
    return "badge text-bg-danger";
  }

  function applyQuickCredState({
    roleChoice,
    statusBadgeEl,
    localNoticeEl,
    remoteFieldsEl,
    baseUrlEl,
    apiKeyEl,
    hintEl,
  }) {
    const cred = credFor(roleChoice.credentialId);
    const spec = cred ? specFor(cred.provider) : null;
    const isLocalSingleton = cred?.id === LOCAL_CRED_ID || cred?.provider === "ollama_local";

    localNoticeEl.classList.toggle("d-none", !isLocalSingleton);
    remoteFieldsEl.classList.toggle("d-none", isLocalSingleton);

    if (!cred) {
      statusBadgeEl.className = "badge text-bg-secondary";
      statusBadgeEl.textContent = "Unconfigured";
      return;
    }

    if (cred.enabled === false) {
      statusBadgeEl.className = "badge text-bg-warning text-dark";
      statusBadgeEl.textContent = "Disabled";
    } else if (isLocalSingleton) {
      const online = Boolean(pState.statusData?.ollama_local?.online);
      statusBadgeEl.className = online ? "badge text-bg-success" : "badge text-bg-danger";
      statusBadgeEl.textContent = online ? "Online" : "Offline";
    } else if (spec?.requires_api_key && !cred.has_key) {
      statusBadgeEl.className = "badge text-bg-warning text-dark";
      statusBadgeEl.textContent = "No Key";
    } else {
      statusBadgeEl.className = "badge text-bg-success";
      statusBadgeEl.textContent = "Ready";
    }

    if (!isLocalSingleton) {
      baseUrlEl.value = cred.base_url || spec?.default_base_url || "";
      apiKeyEl.value = cred.api_key || "";
      hintEl.textContent = spec
        ? `${cred.name} (${spec.label})${spec.env_var ? ` · Env: ${spec.env_var}` : ""}`
        : cred.name;
    }
  }

  function renderRoleSelectors() {
    const s = pState.settings;
    if (!s) return;

    populateRoleModelSelect({
      selectEl: el("visualModelSelect"),
      customPanelEl: el("visualCustomPanel"),
      customCredEl: el("visualCustomCredSelect"),
      customInputEl: el("visualCustomModelInput"),
      credentials: pState.credentials,
      registry: pState.registry,
      discoveredByCred: pState.discoveredByCred,
      role: "vision",
      current: {
        credentialId: s.vision_credential_id || LOCAL_CRED_ID,
        model: s.vision_model || "glm-ocr",
        custom: Boolean(s.vision_custom_model),
      },
    });

    populateRoleModelSelect({
      selectEl: el("llmModelSelect"),
      customPanelEl: el("llmCustomPanel"),
      customCredEl: el("llmCustomCredSelect"),
      customInputEl: el("llmCustomModelInput"),
      credentials: pState.credentials,
      registry: pState.registry,
      discoveredByCred: pState.discoveredByCred,
      role: "llm",
      current: {
        credentialId: s.llm_credential_id || LOCAL_CRED_ID,
        model: s.llm_model || "qwen3:1.7b",
        custom: Boolean(s.llm_custom_model),
      },
    });

    syncVisualQuickCred();
    syncLlmQuickCred();
  }

  function currentVisualChoice() {
    return readRoleModelChoice({
      selectEl: el("visualModelSelect"),
      customCredEl: el("visualCustomCredSelect"),
      customInputEl: el("visualCustomModelInput"),
      credentials: pState.credentials,
    });
  }

  function currentLlmChoice() {
    return readRoleModelChoice({
      selectEl: el("llmModelSelect"),
      customCredEl: el("llmCustomCredSelect"),
      customInputEl: el("llmCustomModelInput"),
      credentials: pState.credentials,
    });
  }

  function syncVisualQuickCred() {
    const choice = currentVisualChoice();
    el("visualCustomPanel").classList.toggle("d-none", el("visualModelSelect").value !== CUSTOM_MODEL_VALUE);
    applyQuickCredState({
      roleChoice: choice,
      statusBadgeEl: el("visualCredStatusBadge"),
      localNoticeEl: el("visualQuickLocalNotice"),
      remoteFieldsEl: el("visualQuickRemoteFields"),
      baseUrlEl: el("visualCredBaseUrl"),
      apiKeyEl: el("visualCredApiKey"),
      hintEl: el("visualQuickCredHint"),
    });
  }

  function syncLlmQuickCred() {
    const choice = currentLlmChoice();
    el("llmCustomPanel").classList.toggle("d-none", el("llmModelSelect").value !== CUSTOM_MODEL_VALUE);
    applyQuickCredState({
      roleChoice: choice,
      statusBadgeEl: el("llmCredStatusBadge"),
      localNoticeEl: el("llmQuickLocalNotice"),
      remoteFieldsEl: el("llmQuickRemoteFields"),
      baseUrlEl: el("llmCredBaseUrl"),
      apiKeyEl: el("llmCredApiKey"),
      hintEl: el("llmQuickCredHint"),
    });
    scheduleVramEstimate(choice.provider === "ollama_local" || choice.provider === "ollama");
  }

  function syncLlmRoleDisabledState() {
    const enabled = el("roleLlmEnabled").checked;
    el("roleLlmConfigFields").classList.toggle("opacity-50", !enabled);
    el("llmModelSelect").disabled = !enabled;
    el("llmCustomCredSelect").disabled = !enabled;
    el("llmCustomModelInput").disabled = !enabled;
    el("roleAutoTag").disabled = !enabled;
    el("roleAutoSummarize").disabled = !enabled;
    el("roleAutoRename").disabled = !enabled;
    el("roleLlmAutoCtx").disabled = !enabled;
    el("sideLlmBadge").textContent = enabled ? "On" : "Off";
  }

  function renderVramPlan(plan) {
    const box = el("llmVramEstimateBox");
    if (!box) return;
    if (!plan) {
      box.textContent = "";
      return;
    }
    box.className = plan.fits_in_vram ? "text-muted" : "text-warning";
    box.textContent = plan.summary_text || "";
  }

  let vramTimer = null;
  function scheduleVramEstimate(probeOllama = false) {
    clearTimeout(vramTimer);
    vramTimer = setTimeout(async () => {
      const choice = currentLlmChoice();
      const installed = (pState.localModelsData?.detailed || []).find(
        (m) => m.name === choice.model || m.name === `${choice.model}:latest`
      );
      try {
        const plan = await estimateLlmVram({
          credential_id: choice.credentialId,
          provider: choice.provider,
          model: choice.model || "qwen3:1.7b",
          auto_ctx: el("roleLlmAutoCtx").checked,
          probe_ollama: probeOllama,
          ...(installed
            ? {
                model_size_gb: installed.size_gb,
                parameter_size: installed.parameter_size,
                quantization: installed.quantization_level,
              }
            : {}),
        });
        renderVramPlan(plan);
      } catch {
        // Ignore transient estimator errors
      }
    }, 220);
  }

  function renderCredentialsPane() {
    const localCred = credFor(LOCAL_CRED_ID);
    const localEnabled = localCred ? localCred.enabled !== false : true;
    const localOnline = Boolean(pState.statusData?.ollama_local?.online);
    const localInstalled = Boolean(pState.statusData?.ollama_local?.installed);

    el("credOllamaLocalToggle").checked = localEnabled;
    el("localOllamaEnabledSwitch").checked = localEnabled;
    el("sideLocalOllamaBadge").textContent = localEnabled ? (localOnline ? "Online" : "Off") : "Disabled";

    const credBadge = el("credLocalOllamaStatusBadge");
    if (!localEnabled) {
      credBadge.className = "badge text-bg-warning text-dark";
      credBadge.textContent = "Disabled";
    } else if (localOnline) {
      credBadge.className = "badge text-bg-success";
      credBadge.textContent = `Online (${(pState.localModelsData?.models || []).length} models)`;
    } else {
      credBadge.className = "badge text-bg-danger";
      credBadge.textContent = localInstalled ? "Offline" : "Not Installed";
    }

    const customCreds = pState.credentials.filter((c) => c.id !== LOCAL_CRED_ID && c.provider !== "ollama_local");
    el("sideCredCountBadge").textContent = String(pState.credentials.length);
    const tbody = el("credentialsTableBody");
    tbody.innerHTML = "";
    if (!customCreds.length) {
      tbody.innerHTML =
        '<tr><td colspan="5" class="text-muted text-center p-3">No extra remote or cloud credentials added yet. Click <strong>+ Add Credential</strong> to add DeepSeek, OpenAI, Anthropic, Google, llama.cpp, or a second Ollama server.</td></tr>';
      return;
    }

    for (const cred of customCreds) {
      const spec = specFor(cred.provider);
      const tr = document.createElement("tr");
      const keyLabel = spec?.requires_api_key
        ? cred.has_key
          ? '<span class="badge text-bg-success">Key Set</span>'
          : '<span class="badge text-bg-warning text-dark">Missing Key</span>'
        : '<span class="badge bg-body-secondary text-body border">No Key Needed</span>';
      tr.innerHTML = `
        <td class="fw-semibold"></td>
        <td><span class="badge bg-body-secondary text-body border"></span></td>
        <td class="text-truncate" style="max-width: 240px;">
          <div class="small text-muted text-truncate cred-url-cell"></div>
          <div>${keyLabel}</div>
        </td>
        <td>
          <div class="form-check form-switch mb-0">
            <input type="checkbox" class="form-check-input cred-enable-toggle" ${cred.enabled !== false ? "checked" : ""} />
          </div>
        </td>
        <td class="text-end text-nowrap">
          <button type="button" class="btn btn-xs btn-outline-primary me-1 cred-test-btn">Test</button>
          <button type="button" class="btn btn-xs btn-outline-secondary me-1 cred-edit-btn">Edit</button>
          <button type="button" class="btn btn-xs btn-outline-danger cred-del-btn">Delete</button>
        </td>`;
      tr.querySelector("td").textContent = cred.name;
      tr.querySelector("td:nth-child(2) span").textContent = spec ? spec.label : cred.provider;
      tr.querySelector(".cred-url-cell").textContent = cred.base_url || "(default)";

      tr.querySelector(".cred-enable-toggle").addEventListener("change", async (e) => {
        try {
          await updateCredential(cred.id, { provider: cred.provider, name: cred.name, enabled: e.target.checked });
          await refresh();
          if (onProvidersChanged) await onProvidersChanged();
        } catch (err) {
          e.target.checked = !e.target.checked;
          showToast("Update Failed", err.message, "danger");
        }
      });

      tr.querySelector(".cred-test-btn").addEventListener("click", async () => {
        try {
          const res = await testProvider({ credential_id: cred.id, provider: cred.provider });
          if (res.online && res.models?.length) pState.discoveredByCred[cred.id] = res.models;
          renderRoleSelectors();
          showToast(cred.name, res.message, res.online ? "success" : "warning");
        } catch (err) {
          showToast("Test Failed", err.message, "danger");
        }
      });

      tr.querySelector(".cred-edit-btn").addEventListener("click", () => openCredentialModal(cred));

      tr.querySelector(".cred-del-btn").addEventListener("click", async () => {
        const ok = await bsConfirm({
          title: "Delete Credential",
          message: `Remove credential "${cred.name}"?`,
          confirmText: "Delete",
          variant: "danger",
        });
        if (!ok) return;
        try {
          await deleteCredential(cred.id);
          showToast("Deleted", `Removed ${cred.name}.`);
          await refresh();
          if (onProvidersChanged) await onProvidersChanged();
        } catch (err) {
          showToast("Delete Failed", err.message, "danger");
        }
      });
      tbody.appendChild(tr);
    }
  }

  function suggestUniqueCredName(providerId) {
    const spec = specFor(providerId);
    const base = spec?.short_label || spec?.label || providerId;
    const taken = new Set(pState.credentials.map((c) => c.name.toLowerCase()));
    if (!taken.has(base.toLowerCase())) return base;
    let idx = 2;
    while (taken.has(`${base} ${idx}`.toLowerCase())) idx += 1;
    return `${base} ${idx}`;
  }

  function syncCredModalProviderMeta() {
    const pid = el("credEditProvider").value;
    const spec = specFor(pid);
    const isEdit = Boolean(el("credEditId").value);
    if (!isEdit) {
      el("credEditBaseUrl").value = spec?.default_base_url || "";
      el("credEditName").value = suggestUniqueCredName(pid);
    }
    el("credEditBaseUrl").placeholder = spec?.default_base_url || "https://your-endpoint/v1";
    el("credEditApiKey").placeholder = spec?.requires_api_key
      ? `API key (or env ${spec.env_var})`
      : `Optional (${spec?.env_var || "none"})`;
    el("credEditEnvHint").textContent = spec
      ? `Protocol: ${spec.protocol} · Visual OCR: ${spec.supports_vision ? "Yes" : "No"} · LLM: ${spec.supports_llm ? "Yes" : "No"}`
      : "";
    el("credEditNameError").classList.add("d-none");
    el("credEditTestResult").textContent = "";
  }

  function openCredentialModal(cred = null) {
    const provSel = el("credEditProvider");
    provSel.innerHTML = "";
    for (const spec of pState.registry) {
      if (spec.singleton || spec.id === "ollama_local") continue;
      provSel.add(new Option(spec.label, spec.id));
    }
    el("credEditId").value = cred ? cred.id : "";
    el("credEditTitle").innerHTML = cred
      ? '<i class="bi bi-key me-1"></i> Edit Provider Credential'
      : '<i class="bi bi-key me-1"></i> Add Provider Credential';
    provSel.value = cred ? cred.provider : "deepseek";
    provSel.disabled = Boolean(cred);
    el("credEditName").value = cred ? cred.name : suggestUniqueCredName(provSel.value);
    el("credEditBaseUrl").value = cred ? cred.base_url || "" : specFor(provSel.value)?.default_base_url || "";
    el("credEditApiKey").value = cred ? cred.api_key || "" : "";
    el("credEditNameError").classList.add("d-none");
    el("credEditTestResult").textContent = "";
    const spec = specFor(provSel.value);
    el("credEditEnvHint").textContent = spec
      ? `Protocol: ${spec.protocol} · Visual OCR: ${spec.supports_vision ? "Yes" : "No"} · LLM: ${spec.supports_llm ? "Yes" : "No"}`
      : "";
    window.bootstrap.Modal.getOrCreateInstance(el("credentialEditorModal")).show();
  }

  function renderLocalOllamaPane() {
    const data = pState.localModelsData || { online: false, installed: false, models: [], detailed: [], recommended: [] };
    const localCred = credFor(LOCAL_CRED_ID);
    const enabled = localCred ? localCred.enabled !== false : true;
    el("localOllamaEnabledSwitch").checked = enabled;
    el("localOllamaUrlInput").value = localCred?.base_url || data.ollama_url || "http://localhost:11434";

    const alertBox = el("localOllamaAlert");
    const startBtn = el("localOllamaStartBtn");
    if (!enabled) {
      alertBox.className = "alert alert-warning py-2 px-3 small mb-0";
      alertBox.textContent = "Ollama Local is disabled. Toggle it on above to use local models.";
      startBtn.classList.add("d-none");
    } else if (data.online) {
      alertBox.className = "alert alert-success py-2 px-3 small mb-0";
      alertBox.textContent = `Connected to ${el("localOllamaUrlInput").value} (${(data.models || []).length} installed model(s)).`;
      startBtn.classList.add("d-none");
    } else if (data.installed) {
      alertBox.className = "alert alert-warning py-2 px-3 small mb-0";
      alertBox.textContent = `Ollama is installed, but not responding at ${el("localOllamaUrlInput").value}.`;
      startBtn.classList.remove("d-none");
    } else {
      alertBox.className = "alert alert-danger py-2 px-3 small mb-0";
      alertBox.textContent = `Ollama was not found locally at ${el("localOllamaUrlInput").value}. Install Ollama or add a remote server in Credential Config.`;
      startBtn.classList.add("d-none");
    }

    const tbody = el("installedModelsBody");
    const installedNames = new Set(data.models || []);
    if (!data.online) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-muted p-3">Local Ollama service is offline.</td></tr>';
    } else if (!data.detailed?.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-muted p-3">No models installed yet. Pull one from the right panel.</td></tr>';
    } else {
      tbody.innerHTML = "";
      for (const m of data.detailed) {
        const isVis =
          pState.settings?.vision_credential_id === LOCAL_CRED_ID &&
          (m.name === pState.settings?.vision_model || m.name === `${pState.settings?.vision_model}:latest`);
        const isLlm =
          pState.settings?.llm_enabled &&
          pState.settings?.llm_credential_id === LOCAL_CRED_ID &&
          (m.name === pState.settings?.llm_model || m.name === `${pState.settings?.llm_model}:latest`);
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="fw-semibold"></td>
          <td>${m.parameter_size || "-"}</td>
          <td>${m.quantization_level || "-"}</td>
          <td>${m.size_gb} GB</td>
          <td class="text-end text-nowrap">
            <button class="btn btn-xs ${isVis ? "btn-primary" : "btn-outline-primary"} me-1 set-vis-btn" type="button"></button>
            <button class="btn btn-xs ${isLlm ? "btn-info" : "btn-outline-secondary"} me-1 set-llm-btn" type="button"></button>
            <button class="btn btn-xs btn-outline-danger del-btn" type="button">Delete</button>
          </td>`;
        tr.querySelector("td").textContent = m.name;
        const visBtn = tr.querySelector(".set-vis-btn");
        visBtn.textContent = isVis ? "Visual ✓" : "Use Visual";
        visBtn.disabled = isVis;
        visBtn.addEventListener("click", async () => {
          await saveSettings({
            vision_credential_id: LOCAL_CRED_ID,
            vision_provider: "ollama_local",
            vision_model: m.name,
            vision_custom_model: false,
          });
          showToast("Visual Model Updated", `${m.name} (Ollama Local) set for OCR.`);
          await refresh();
          if (onProvidersChanged) await onProvidersChanged();
        });

        const llmBtn = tr.querySelector(".set-llm-btn");
        llmBtn.textContent = isLlm ? "LLM ✓" : "Use LLM";
        llmBtn.disabled = isLlm;
        llmBtn.addEventListener("click", async () => {
          await saveSettings({
            llm_enabled: true,
            llm_credential_id: LOCAL_CRED_ID,
            llm_provider: "ollama_local",
            llm_model: m.name,
            llm_custom_model: false,
          });
          showToast("Assistant LLM Updated", `${m.name} (Ollama Local) enabled.`);
          await refresh();
          if (onProvidersChanged) await onProvidersChanged();
        });

        tr.querySelector(".del-btn").addEventListener("click", async () => {
          const ok = await bsConfirm({
            title: "Delete Local Model",
            message: `Delete "${m.name}" from local Ollama storage?`,
            confirmText: "Delete",
            variant: "danger",
          });
          if (!ok) return;
          try {
            await deleteModel(m.name, LOCAL_CRED_ID);
            showToast("Model Deleted", `${m.name} removed.`);
            await refresh();
            if (onProvidersChanged) await onProvidersChanged();
          } catch (err) {
            showToast("Delete Failed", err.message, "danger");
          }
        });
        tbody.appendChild(tr);
      }
    }

    const smallLlms = [
      { name: "qwen3:1.7b", size: "~1.7B", vram: "~1.4 GB", description: "Fast small local LLM for tags, renaming & summaries.", role: "llm" },
      { name: "llama3.2:3b", size: "3B", vram: "~2.0 GB", description: "Compact general local LLM for summaries & OCR cleanup.", role: "llm" },
    ];
    const recItems = [
      ...(data.recommended || []).map((r) => ({ ...r, role: "vision" })),
      ...smallLlms,
    ];
    const recList = el("recommendedModelsList");
    recList.innerHTML = "";
    for (const rec of recItems) {
      const pulled = installedNames.has(rec.name) || installedNames.has(`${rec.name}:latest`);
      const roleTag = rec.role === "llm" ? '<span class="badge text-bg-secondary ms-1">Small LLM</span>' : '<span class="badge text-bg-primary ms-1">Visual OCR</span>';
      const row = document.createElement("div");
      row.className = "list-group-item d-flex justify-content-between align-items-start gap-2";
      row.innerHTML = `
        <div>
          <div class="fw-semibold">${rec.name} <span class="badge bg-body-secondary text-body border">${rec.size}</span> <span class="badge text-bg-info">${rec.vram}</span>${roleTag}</div>
          <div class="text-muted">${rec.description}</div>
        </div>
        <button class="btn btn-sm ${pulled ? "btn-outline-success" : "btn-outline-primary"} text-nowrap" type="button" ${pulled ? "disabled" : ""}>
          ${pulled ? "Installed" : "Pull"}
        </button>`;
      const btn = row.querySelector("button");
      if (!pulled) btn.addEventListener("click", () => handlePull(rec.name));
      recList.appendChild(row);
    }
  }

  async function handlePull(modelName) {
    const status = el("modelActionStatus");
    status.textContent = `Pulling ${modelName}... this may take a minute.`;
    try {
      await pullModel(modelName, LOCAL_CRED_ID);
      status.textContent = `Pulled ${modelName} successfully.`;
      showToast("Model Ready", `${modelName} installed in Ollama Local.`, "success");
      await refresh();
      if (onProvidersChanged) await onProvidersChanged();
    } catch (err) {
      status.textContent = err.message;
      showToast("Pull Failed", err.message, "danger");
    }
  }

  async function refresh(paneOverride = null) {
    if (paneOverride) setPane(paneOverride);
    const [provData, settings, localData, statusData] = await Promise.all([
      fetchProviders(),
      fetchSettings(),
      fetchModels(LOCAL_CRED_ID),
      fetchProviderStatus(false).catch(() => null),
    ]);
    pState.registry = provData.registry || [];
    pState.credentials = provData.credentials || [];
    pState.settings = settings;
    pState.localModelsData = localData;
    pState.statusData = statusData;
    if (localData?.models?.length) {
      pState.discoveredByCred[LOCAL_CRED_ID] = localData.models;
    }

    el("roleLlmEnabled").checked = Boolean(settings.llm_enabled);
    el("roleAutoTag").checked = Boolean(settings.auto_tag_on_ocr);
    el("roleAutoSummarize").checked = Boolean(settings.auto_summarize_on_ocr);
    el("roleAutoRename").checked = Boolean(settings.auto_rename_on_ocr);
    el("roleLlmAutoCtx").checked = settings.llm_auto_ctx !== false;

    renderRoleSelectors();
    syncLlmRoleDisabledState();
    renderCredentialsPane();
    renderLocalOllamaPane();
    renderVramPlan(provData.llm?.vram_plan);
    setPane(pState.activePane);
  }

  // Sidebar navigation
  document.querySelectorAll("#modelsView [data-pref-pane]").forEach((btn) => {
    btn.addEventListener("click", () => setPane(btn.dataset.prefPane));
  });
  el("visualOpenCredsBtn").addEventListener("click", () => setPane("credentialsPane"));
  el("llmOpenCredsBtn").addEventListener("click", () => setPane("credentialsPane"));
  el("visualGoLocalOllamaBtn").addEventListener("click", () => setPane("localOllamaPane"));
  el("llmGoLocalOllamaBtn").addEventListener("click", () => setPane("localOllamaPane"));
  el("credGoLocalOllamaBtn").addEventListener("click", () => setPane("localOllamaPane"));

  // Visual role listeners
  el("visualModelSelect").addEventListener("change", syncVisualQuickCred);
  el("visualCustomCredSelect").addEventListener("change", syncVisualQuickCred);
  el("saveVisualConfigBtn").addEventListener("click", async () => {
    const choice = currentVisualChoice();
    if (!choice.model) {
      showToast("Model Required", "Select or enter a Visual OCR model.", "warning");
      return;
    }
    await saveSettings({
      vision_credential_id: choice.credentialId,
      vision_provider: choice.provider,
      vision_model: choice.model,
      vision_custom_model: choice.custom,
    });
    showToast("Visual Model Saved", `${choice.model} (${choice.credentialName})`, "success");
    await refresh();
    if (onProvidersChanged) await onProvidersChanged();
  });

  // LLM role listeners
  el("roleLlmEnabled").addEventListener("change", syncLlmRoleDisabledState);
  el("llmModelSelect").addEventListener("change", syncLlmQuickCred);
  el("llmCustomCredSelect").addEventListener("change", syncLlmQuickCred);
  el("llmCustomModelInput").addEventListener("input", () => scheduleVramEstimate(false));
  el("roleLlmAutoCtx").addEventListener("change", () => scheduleVramEstimate(false));
  el("saveLlmConfigBtn").addEventListener("click", async () => {
    const enabled = el("roleLlmEnabled").checked;
    const choice = currentLlmChoice();
    if (enabled && !choice.model) {
      showToast("Model Required", "Select or enter an Assistant LLM model.", "warning");
      return;
    }
    await saveSettings({
      llm_enabled: enabled,
      llm_credential_id: choice.credentialId,
      llm_provider: choice.provider,
      llm_model: choice.model,
      llm_custom_model: choice.custom,
      auto_tag_on_ocr: el("roleAutoTag").checked,
      auto_summarize_on_ocr: el("roleAutoSummarize").checked,
      auto_rename_on_ocr: el("roleAutoRename").checked,
      llm_auto_ctx: el("roleLlmAutoCtx").checked,
    });
    showToast("LLM Config Saved", enabled ? `${choice.model} (${choice.credentialName})` : "Assistant LLM disabled.", "success");
    await refresh();
    if (onProvidersChanged) await onProvidersChanged();
  });

  // Shared inline quick credential save/test helper for Visual & LLM panes
  function wireQuickCredActions(roleGetter, baseUrlId, apiKeyId, statusBadgeId, saveBtnId, testBtnId) {
    el(saveBtnId).addEventListener("click", async () => {
      const choice = roleGetter();
      const cred = credFor(choice.credentialId);
      if (!cred) return;
      try {
        await updateCredential(cred.id, {
          name: cred.name,
          provider: cred.provider,
          base_url: el(baseUrlId).value.trim(),
          api_key: el(apiKeyId).value.trim(),
        });
        showToast("Credential Saved", `Updated ${cred.name}.`, "success");
        await refresh();
        if (onProvidersChanged) await onProvidersChanged();
      } catch (err) {
        showToast("Save Failed", err.message, "danger");
      }
    });

    el(testBtnId).addEventListener("click", async () => {
      const choice = roleGetter();
      const cred = credFor(choice.credentialId);
      if (!cred) return;
      const badge = el(statusBadgeId);
      badge.className = "badge text-bg-info";
      badge.textContent = "Testing...";
      try {
        const res = await testProvider({
          credential_id: cred.id,
          provider: cred.provider,
          base_url: el(baseUrlId).value.trim(),
          api_key: el(apiKeyId).value.trim(),
        });
        badge.className = statusBadgeClass(res.status_code || (res.online ? "ok" : "error"));
        badge.textContent = res.online ? "Online" : res.status_text || "Error";
        if (res.online && res.models?.length) {
          pState.discoveredByCred[cred.id] = res.models;
          renderRoleSelectors();
        }
        showToast(cred.name, res.message, res.online ? "success" : "warning");
      } catch (err) {
        badge.className = "badge text-bg-danger";
        badge.textContent = "Error";
        showToast("Test Failed", err.message, "danger");
      }
    });
  }

  wireQuickCredActions(currentVisualChoice, "visualCredBaseUrl", "visualCredApiKey", "visualCredStatusBadge", "visualSaveCredBtn", "visualTestCredBtn");
  wireQuickCredActions(currentLlmChoice, "llmCredBaseUrl", "llmCredApiKey", "llmCredStatusBadge", "llmSaveCredBtn", "llmTestCredBtn");

  // Credentials pane & Modal
  async function toggleLocalOllama(enabled) {
    try {
      await updateCredential(LOCAL_CRED_ID, { provider: "ollama_local", enabled });
      showToast("Ollama Local", enabled ? "Ollama Local enabled." : "Ollama Local disabled.");
      await refresh();
      if (onProvidersChanged) await onProvidersChanged();
    } catch (err) {
      showToast("Update Failed", err.message, "danger");
    }
  }

  el("credOllamaLocalToggle").addEventListener("change", (e) => toggleLocalOllama(e.target.checked));
  el("localOllamaEnabledSwitch").addEventListener("change", (e) => toggleLocalOllama(e.target.checked));
  el("refreshCredentialsBtn").addEventListener("click", () => refresh());
  el("addCredentialBtn").addEventListener("click", () => openCredentialModal(null));
  el("credEditProvider").addEventListener("change", syncCredModalProviderMeta);
  el("credEditName").addEventListener("input", () => el("credEditNameError").classList.add("d-none"));

  el("credEditTestBtn").addEventListener("click", async () => {
    const out = el("credEditTestResult");
    out.className = "small text-muted";
    out.textContent = "Testing connection...";
    try {
      const res = await testProvider({
        credential_id: el("credEditId").value || undefined,
        provider: el("credEditProvider").value,
        base_url: el("credEditBaseUrl").value.trim(),
        api_key: el("credEditApiKey").value.trim(),
      });
      out.className = `small ${res.online ? "text-success" : "text-warning"}`;
      out.textContent = res.message;
    } catch (err) {
      out.className = "small text-danger";
      out.textContent = err.message;
    }
  });

  el("credEditForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = el("credEditId").value.trim();
    const name = el("credEditName").value.trim();
    const errEl = el("credEditNameError");
    const duplicate = pState.credentials.some(
      (c) => c.id !== id && c.name.trim().toLowerCase() === name.toLowerCase()
    );
    if (duplicate) {
      errEl.textContent = `A credential named "${name}" already exists. Names must be unique.`;
      errEl.classList.remove("d-none");
      return;
    }
    const payload = {
      name,
      provider: el("credEditProvider").value,
      base_url: el("credEditBaseUrl").value.trim(),
      api_key: el("credEditApiKey").value.trim(),
    };
    try {
      if (id) await updateCredential(id, payload);
      else await createCredential(payload);
      window.bootstrap.Modal.getOrCreateInstance(el("credentialEditorModal")).hide();
      showToast("Credential Saved", `${name} saved.`, "success");
      await refresh();
      if (onProvidersChanged) await onProvidersChanged();
    } catch (err) {
      errEl.textContent = err.message;
      errEl.classList.remove("d-none");
    }
  });

  // Local Ollama Integration actions
  el("modelsRefreshBtn").addEventListener("click", () => refresh("localOllamaPane"));
  el("saveLocalOllamaUrlBtn").addEventListener("click", async () => {
    const url = el("localOllamaUrlInput").value.trim() || "http://localhost:11434";
    await updateCredential(LOCAL_CRED_ID, { provider: "ollama_local", base_url: url });
    showToast("Ollama Local URL Saved", url, "success");
    await refresh("localOllamaPane");
    if (onProvidersChanged) await onProvidersChanged();
  });
  el("testLocalOllamaBtn").addEventListener("click", async () => {
    const url = el("localOllamaUrlInput").value.trim() || "http://localhost:11434";
    const res = await testProvider({ credential_id: LOCAL_CRED_ID, provider: "ollama_local", base_url: url });
    showToast("Ollama Local", res.message, res.online ? "success" : "warning");
    await refresh("localOllamaPane");
  });
  el("localOllamaStartBtn").addEventListener("click", async () => {
    const btn = el("localOllamaStartBtn");
    btn.disabled = true;
    btn.textContent = "Starting...";
    try {
      const res = await startOllama();
      showToast("Ollama Local", res.message, res.online ? "success" : "warning");
      await refresh("localOllamaPane");
      if (onProvidersChanged) await onProvidersChanged();
    } finally {
      btn.disabled = false;
      btn.textContent = "Start Ollama";
    }
  });
  el("customPullBtn").addEventListener("click", () => {
    const val = el("customPullInput").value.trim();
    if (val) handlePull(val);
  });

  return { refresh, setPane };
}
