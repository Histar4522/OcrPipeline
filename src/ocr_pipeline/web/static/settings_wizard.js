import { fetchModels, fetchSettings, pullModel, saveSettings, startOllama } from "./api.js";

const el = (id) => document.getElementById(id);

export function initSettingsAndWizard({ showToast, onSettingsSaved }) {
  const wizardModal = new window.bootstrap.Modal(el("wizardModal"));

  async function loadSettingsForm() {
    const s = await fetchSettings();
    el("setOllamaUrl").value = s.ollama_url;
    el("setDefaultModel").value = s.default_model;
    el("setArchiveDir").value = s.archive_dir;
    el("setDpi").value = s.dpi;
    el("setMaxDim").value = s.max_image_dim;
    el("setNumCtx").value = s.num_ctx;
    el("setSystemPrompt").value = s.system_prompt;
    el("setMinimizeTray").checked = Boolean(s.minimize_to_tray);
    el("setDesktopNotify").checked = Boolean(s.desktop_notifications);
    el("wizArchiveDir").value = s.archive_dir;
    el("wizTray").checked = Boolean(s.minimize_to_tray);
    el("wizNotify").checked = Boolean(s.desktop_notifications);
    return s;
  }

  async function checkWizardStatus(autoLaunch = true) {
    const oDesc = el("wizOllamaDesc");
    const mDesc = el("wizModelDesc");
    const pullBtn = el("wizPullGlmBtn");
    const startBtn = el("wizStartOllamaBtn");

    let data = await fetchModels();
    if (!data.online && data.installed && autoLaunch) {
      oDesc.className = "small text-primary";
      oDesc.textContent = "Detected local Ollama installation. Attempting to start Ollama service...";
      startBtn.classList.add("d-none");
      const res = await startOllama();
      if (res.online) {
        showToast("Ollama Started", "Local Ollama service launched automatically.", "success");
        await onSettingsSaved();
      }
      data = await fetchModels();
    }

    if (data.online) {
      oDesc.className = "small text-success";
      oDesc.textContent = `Connected to ${data.ollama_url} (${data.models.length} model(s) installed).`;
      startBtn.classList.add("d-none");
    } else if (data.installed) {
      oDesc.className = "small text-warning";
      oDesc.textContent = `Ollama is installed, but not responding at ${data.ollama_url}. Click Start Ollama or check Settings.`;
      startBtn.classList.remove("d-none");
    } else {
      oDesc.className = "small text-danger";
      oDesc.textContent = `Ollama was not detected at ${data.ollama_url} or in PATH. Install Ollama, then click Re-check.`;
      startBtn.classList.add("d-none");
    }

    const hasGlm = data.models.includes("glm-ocr") || data.models.includes("glm-ocr:latest");
    if (hasGlm) {
      mDesc.className = "small text-success";
      mDesc.textContent = "glm-ocr is installed and ready for fast local OCR.";
      pullBtn.disabled = true;
      pullBtn.textContent = "Installed";
    } else {
      mDesc.className = "small text-muted";
      mDesc.textContent = data.models.length
        ? `Installed: ${data.models.join(", ")}. Pull glm-ocr (~0.9B) for fastest 8 GB VRAM performance.`
        : "No OCR model pulled yet. Click Pull glm-ocr to download (~1 GB).";
      pullBtn.disabled = !data.online;
      pullBtn.textContent = "Pull glm-ocr";
    }
  }

  async function openWizard() {
    await loadSettingsForm();
    wizardModal.show();
    await checkWizardStatus(true);
  }

  el("openWizardBtn").addEventListener("click", openWizard);
  el("wizRecheckBtn").addEventListener("click", () => checkWizardStatus(false));
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
      await checkWizardStatus(false);
    }
  });
  el("wizPullGlmBtn").addEventListener("click", async () => {
    const btn = el("wizPullGlmBtn");
    btn.disabled = true;
    btn.textContent = "Pulling...";
    try {
      await pullModel("glm-ocr");
      await saveSettings({ default_model: "glm-ocr" });
      showToast("Model Ready", "glm-ocr pulled and set as default.", "success");
    } catch (err) {
      showToast("Pull Failed", err.message, "danger");
    } finally {
      await checkWizardStatus();
      await onSettingsSaved();
    }
  });

  el("wizFinishBtn").addEventListener("click", async () => {
    await saveSettings({
      setup_completed: true,
      archive_dir: el("wizArchiveDir").value.trim(),
      minimize_to_tray: el("wizTray").checked,
      desktop_notifications: el("wizNotify").checked,
    });
    wizardModal.hide();
    await loadSettingsForm();
    await onSettingsSaved();
    showToast("Setup Complete", "Preferences saved.", "success");
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
      system_prompt: el("setSystemPrompt").value,
      minimize_to_tray: el("setMinimizeTray").checked,
      desktop_notifications: el("setDesktopNotify").checked,
    });
    el("settingsSaveMsg").textContent = "Saved!";
    setTimeout(() => (el("settingsSaveMsg").textContent = ""), 2500);
    await onSettingsSaved(saved);
    showToast("Settings Saved", "Configuration updated.", "success");
  });

  return { loadSettingsForm, openWizard };
}
