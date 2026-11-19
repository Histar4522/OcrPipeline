import {
  deleteModel,
  estimateLlmVram,
  fetchModels,
  fetchProviders,
  fetchSettings,
  pullModel,
  saveSettings,
  testProvider,
} from "./api.js";
import { bsConfirm } from "./dialogs.js";

const el = (id) => document.getElementById(id);

export function initProvidersAndModelsView({ showToast, onProvidersChanged }) {
  const pState = {
    registry: [],
    providers: {},
    settings: null,
    ollamaData: null,
    discoveredModels: {},
  };

  function specFor(pid) {
    return pState.registry.find((r) => r.id === pid) || null;
  }

  function fillDatalist(listId, models) {
    const dl = el(listId);
    if (!dl) return;
    dl.innerHTML = "";
    for (const m of [...new Set(models.filter(Boolean))]) {
      const opt = document.createElement("option");
      opt.value = m;
      dl.appendChild(opt);
    }
  }

  function updateRoleDatalists() {
    const vPid = el("roleVisionProvider").value;
    const vSpec = specFor(vPid);
    const vDiscovered = pState.discoveredModels[vPid] || (vPid === "ollama" ? pState.ollamaData?.models || [] : []);
    fillDatalist("visionModelDatalist", [...vDiscovered, ...(vSpec?.suggested_vision_models || [])]);

    const lPid = el("roleLlmProvider").value;
    const lSpec = specFor(lPid);
    const lDiscovered = pState.discoveredModels[lPid] || (lPid === "ollama" ? pState.ollamaData?.models || [] : []);
    fillDatalist("llmModelDatalist", [...lDiscovered, ...(lSpec?.suggested_llm_models || [])]);
  }

  function syncLlmRoleDisabledState() {
    const enabled = el("roleLlmEnabled").checked;
    el("roleLlmConfigFields").classList.toggle("opacity-50", !enabled);
    el("roleLlmProvider").disabled = !enabled;
    el("roleLlmModel").disabled = !enabled;
    el("roleAutoTag").disabled = !enabled;
    el("roleAutoSummarize").disabled = !enabled;
    el("roleAutoRename").disabled = !enabled;
    el("roleLlmAutoCtx").disabled = !enabled;
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
      const pid = el("roleLlmProvider").value || "ollama";
      const model = el("roleLlmModel").value.trim() || "qwen3:1.7b";
      const installed = (pState.ollamaData?.detailed || []).find(
        (m) => m.name === model || m.name === `${model}:latest`
      );
      try {
        const plan = await estimateLlmVram({
          provider: pid,
          model,
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
        // Ignore transient UI estimate failures
      }
    }, 220);
  }

  function renderProviderConfigEditor(pid) {
    const spec = specFor(pid);
    const cfg = pState.providers[pid] || {};
    el("cfgBaseUrl").value = cfg.base_url || spec?.default_base_url || "";
    el("cfgBaseUrl").placeholder = spec?.default_base_url || "https://your-endpoint/v1";
    el("cfgApiKey").value = cfg.api_key || "";
    el("cfgApiKey").placeholder = spec?.requires_api_key
      ? `API key (or set ${spec.env_var})`
      : `Optional (${spec?.env_var || "none"})`;
    el("cfgEnvHint").textContent = spec
      ? `Protocol: ${spec.protocol} · Vision: ${spec.supports_vision ? "Yes" : "No"} · LLM: ${spec.supports_llm ? "Yes" : "No"}${spec.env_var ? ` · Env fallback: ${spec.env_var}` : ""}`
      : "";
    el("cfgProviderResult").textContent = "";
  }

  async function handlePull(modelName) {
    const status = el("modelActionStatus");
    status.textContent = `Pulling ${modelName}... this may take a minute.`;
    try {
      await pullModel(modelName);
      status.textContent = `Pulled ${modelName} successfully.`;
      showToast("Model Ready", `${modelName} is now installed in Ollama.`, "success");
      await refresh();
      if (onProvidersChanged) await onProvidersChanged();
    } catch (err) {
      status.textContent = err.message;
      showToast("Pull Failed", err.message, "danger");
    }
  }

  function renderOllamaModels(data) {
    pState.ollamaData = data;
    const tbody = el("installedModelsBody");
    const recList = el("recommendedModelsList");
    const installedNames = new Set(data.models || []);

    if (!data.online) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-muted p-3">Local Ollama is offline (optional if using OpenAI, Anthropic, Google, llama.cpp, etc.).</td></tr>';
    } else if (!data.detailed?.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-muted p-3">No Ollama models installed yet.</td></tr>';
    } else {
      tbody.innerHTML = "";
      for (const m of data.detailed) {
        const tr = document.createElement("tr");
        const isVis =
          pState.settings?.vision_provider === "ollama" &&
          (m.name === pState.settings?.vision_model || m.name === `${pState.settings?.vision_model}:latest`);
        const isLlm =
          pState.settings?.llm_enabled &&
          pState.settings?.llm_provider === "ollama" &&
          (m.name === pState.settings?.llm_model || m.name === `${pState.settings?.llm_model}:latest`);
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
        visBtn.textContent = isVis ? "Vision ✓" : "Use Vision";
        visBtn.disabled = isVis;
        visBtn.addEventListener("click", async () => {
          await saveSettings({ vision_provider: "ollama", vision_model: m.name });
          showToast("Vision Model Updated", `Ollama / ${m.name} set for OCR.`);
          await refresh();
          if (onProvidersChanged) await onProvidersChanged();
        });

        const llmBtn = tr.querySelector(".set-llm-btn");
        llmBtn.textContent = isLlm ? "LLM ✓" : "Use LLM";
        llmBtn.disabled = isLlm;
        llmBtn.addEventListener("click", async () => {
          await saveSettings({ llm_enabled: true, llm_provider: "ollama", llm_model: m.name });
          showToast("Assistant LLM Updated", `Ollama / ${m.name} enabled for AI features.`);
          await refresh();
          if (onProvidersChanged) await onProvidersChanged();
        });

        tr.querySelector(".del-btn").addEventListener("click", async () => {
          const ok = await bsConfirm({
            title: "Delete Model",
            message: `Delete Ollama model "${m.name}" from local storage?`,
            confirmText: "Delete",
            variant: "danger",
          });
          if (!ok) return;
          try {
            await deleteModel(m.name);
            showToast("Model Deleted", `${m.name} removed from Ollama.`);
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
      { name: "qwen3:1.7b", size: "~1.7B", vram: "~1.4 GB", description: "Fast small local Assistant LLM for tags, folders, renaming & summaries.", role: "llm" },
      { name: "llama3.2:3b", size: "3B", vram: "~2.0 GB", description: "Compact general local LLM for note organization and OCR cleanup.", role: "llm" },
    ];
    const recItems = [
      ...(data.recommended || []).map((r) => ({ ...r, role: "vision" })),
      ...smallLlms,
    ];
    recList.innerHTML = "";
    for (const rec of recItems) {
      const pulled = installedNames.has(rec.name) || installedNames.has(`${rec.name}:latest`);
      const roleTag = rec.role === "llm" ? '<span class="badge text-bg-secondary ms-1">Small LLM</span>' : '<span class="badge text-bg-primary ms-1">Vision OCR</span>';
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

  async function refresh() {
    const [provData, settings, ollamaData] = await Promise.all([
      fetchProviders(),
      fetchSettings(),
      fetchModels(),
    ]);
    pState.registry = provData.registry || [];
    pState.providers = provData.providers || {};
    pState.settings = settings;

    const visionSelect = el("roleVisionProvider");
    const llmSelect = el("roleLlmProvider");
    const cfgSelect = el("cfgProviderSelect");
    const prevCfg = cfgSelect.value;

    visionSelect.innerHTML = "";
    llmSelect.innerHTML = "";
    cfgSelect.innerHTML = "";
    for (const spec of pState.registry) {
      if (spec.supports_vision) {
        visionSelect.add(new Option(spec.label, spec.id));
      }
      if (spec.supports_llm) {
        llmSelect.add(new Option(spec.label, spec.id));
      }
      const hasKeyNote = pState.providers[spec.id]?.has_key && spec.requires_api_key ? " ✓" : "";
      cfgSelect.add(new Option(`${spec.label}${hasKeyNote}`, spec.id));
    }

    visionSelect.value = settings.vision_provider || "ollama";
    el("roleVisionModel").value = settings.vision_model || "glm-ocr";
    el("roleLlmEnabled").checked = Boolean(settings.llm_enabled);
    llmSelect.value = settings.llm_provider || "ollama";
    el("roleLlmModel").value = settings.llm_model || "";
    el("roleAutoTag").checked = Boolean(settings.auto_tag_on_ocr);
    el("roleAutoSummarize").checked = Boolean(settings.auto_summarize_on_ocr);
    el("roleAutoRename").checked = Boolean(settings.auto_rename_on_ocr);
    el("roleLlmAutoCtx").checked = settings.llm_auto_ctx !== false;

    cfgSelect.value = prevCfg && specFor(prevCfg) ? prevCfg : (settings.vision_provider || "ollama");
    renderProviderConfigEditor(cfgSelect.value);
    syncLlmRoleDisabledState();
    renderOllamaModels(ollamaData);
    updateRoleDatalists();
    renderVramPlan(provData.llm?.vram_plan);
    scheduleVramEstimate(settings.llm_provider === "ollama" && Boolean(settings.llm_model));
  }

  el("roleVisionProvider").addEventListener("change", () => {
    const spec = specFor(el("roleVisionProvider").value);
    if (spec?.suggested_vision_models?.length && !el("roleVisionModel").value.trim()) {
      el("roleVisionModel").value = spec.suggested_vision_models[0];
    }
    updateRoleDatalists();
  });
  el("roleLlmProvider").addEventListener("change", () => {
    updateRoleDatalists();
    scheduleVramEstimate(el("roleLlmProvider").value === "ollama");
  });
  el("roleLlmModel").addEventListener("input", () => scheduleVramEstimate(false));
  el("roleLlmModel").addEventListener("change", () => scheduleVramEstimate(true));
  el("roleLlmAutoCtx").addEventListener("change", () => scheduleVramEstimate(false));
  el("roleLlmEnabled").addEventListener("change", syncLlmRoleDisabledState);
  el("cfgProviderSelect").addEventListener("change", (e) => renderProviderConfigEditor(e.target.value));

  el("saveRolesBtn").addEventListener("click", async () => {
    const vModel = el("roleVisionModel").value.trim();
    if (!vModel) {
      showToast("Vision Model Required", "Please specify a Vision/OCR model.", "warning");
      return;
    }
    await saveSettings({
      vision_provider: el("roleVisionProvider").value,
      vision_model: vModel,
      llm_enabled: el("roleLlmEnabled").checked,
      llm_provider: el("roleLlmProvider").value,
      llm_model: el("roleLlmModel").value.trim(),
      auto_tag_on_ocr: el("roleAutoTag").checked,
      auto_summarize_on_ocr: el("roleAutoSummarize").checked,
      auto_rename_on_ocr: el("roleAutoRename").checked,
      llm_auto_ctx: el("roleLlmAutoCtx").checked,
    });
    showToast("Roles Saved", "Updated active Vision & Assistant LLM settings.", "success");
    await refresh();
    if (onProvidersChanged) await onProvidersChanged();
  });

  el("cfgSaveProviderBtn").addEventListener("click", async () => {
    const pid = el("cfgProviderSelect").value;
    await saveSettings({
      providers: {
        [pid]: {
          base_url: el("cfgBaseUrl").value.trim(),
          api_key: el("cfgApiKey").value.trim(),
        },
      },
    });
    showToast("Provider Saved", `Saved ${specFor(pid)?.label || pid} configuration.`, "success");
    await refresh();
    if (onProvidersChanged) await onProvidersChanged();
  });

  el("cfgTestProviderBtn").addEventListener("click", async () => {
    const pid = el("cfgProviderSelect").value;
    const out = el("cfgProviderResult");
    out.className = "small text-muted mt-2";
    out.textContent = "Testing connection...";
    try {
      const res = await testProvider({
        provider: pid,
        base_url: el("cfgBaseUrl").value.trim(),
        api_key: el("cfgApiKey").value.trim(),
      });
      if (res.online) {
        out.className = "small text-success mt-2";
        pState.discoveredModels[pid] = res.models || [];
        updateRoleDatalists();
      } else {
        out.className = "small text-warning mt-2";
      }
      out.textContent = res.message;
    } catch (err) {
      out.className = "small text-danger mt-2";
      out.textContent = err.message;
    }
  });

  el("modelsRefreshBtn").addEventListener("click", refresh);
  el("customPullBtn").addEventListener("click", () => {
    const val = el("customPullInput").value.trim();
    if (val) handlePull(val);
  });

  return { refresh };
}
