import { deleteModel, fetchModels, pullModel, saveSettings } from "./api.js";
import { bsConfirm } from "./dialogs.js";

const el = (id) => document.getElementById(id);

export function initModelsView({ showToast, onModelsChanged }) {
  async function handlePull(modelName) {
    const status = el("modelActionStatus");
    status.textContent = `Pulling ${modelName}... this may take a minute.`;
    try {
      await pullModel(modelName);
      status.textContent = `Pulled ${modelName} successfully.`;
      showToast("Model Ready", `${modelName} is now installed in Ollama.`, "success");
      await refresh();
      await onModelsChanged();
    } catch (err) {
      status.textContent = err.message;
      showToast("Pull Failed", err.message, "danger");
    }
  }

  async function refresh() {
    const data = await fetchModels();
    const tbody = el("installedModelsBody");
    const recList = el("recommendedModelsList");
    const installedNames = new Set(data.models || []);

    if (!data.online) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-danger p-3">Ollama is offline. Start Ollama to manage models.</td></tr>';
    } else if (!data.detailed?.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-muted p-3">No Ollama models installed yet.</td></tr>';
    } else {
      tbody.innerHTML = "";
      for (const m of data.detailed) {
        const tr = document.createElement("tr");
        const isDefault = m.name === data.default_model || m.name === `${data.default_model}:latest`;
        tr.innerHTML = `
          <td class="fw-semibold"></td>
          <td>${m.parameter_size || "-"}</td>
          <td>${m.quantization_level || "-"}</td>
          <td>${m.size_gb} GB</td>
          <td class="text-end text-nowrap">
            <button class="btn btn-xs btn-outline-primary me-1 set-def-btn" type="button"></button>
            <button class="btn btn-xs btn-outline-danger del-btn" type="button">Delete</button>
          </td>`;
        tr.querySelector("td").textContent = m.name;
        const defBtn = tr.querySelector(".set-def-btn");
        defBtn.textContent = isDefault ? "Default" : "Set Default";
        defBtn.disabled = isDefault;
        defBtn.addEventListener("click", async () => {
          await saveSettings({ default_model: m.name });
          showToast("Default Updated", `${m.name} is now your default model.`);
          await refresh();
          await onModelsChanged();
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
            await onModelsChanged();
          } catch (err) {
            showToast("Delete Failed", err.message, "danger");
          }
        });
        tbody.appendChild(tr);
      }
    }

    recList.innerHTML = "";
    for (const rec of data.recommended || []) {
      const pulled = installedNames.has(rec.name) || installedNames.has(`${rec.name}:latest`);
      const row = document.createElement("div");
      row.className = "list-group-item d-flex justify-content-between align-items-start gap-2";
      row.innerHTML = `
        <div>
          <div class="fw-semibold">${rec.name} <span class="badge bg-body-secondary text-body border">${rec.size}</span> <span class="badge text-bg-info">${rec.vram}</span></div>
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

  el("modelsRefreshBtn").addEventListener("click", refresh);
  el("customPullBtn").addEventListener("click", () => {
    const val = el("customPullInput").value.trim();
    if (val) handlePull(val);
  });

  return { refresh };
}
