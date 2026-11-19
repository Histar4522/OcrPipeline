/** Shared helpers for `model_name (Credential Name)` dropdowns + Custom model fallback. */

export const CUSTOM_MODEL_VALUE = "__custom__";

export function encodeModelChoice(credentialId, model) {
  return `${credentialId}::${model}`;
}

export function decodeModelChoice(value) {
  const idx = String(value || "").indexOf("::");
  if (idx === -1) return null;
  return {
    credentialId: value.slice(0, idx),
    model: value.slice(idx + 2),
  };
}

export function isModelEligibleForRole(providerId, modelName, role) {
  const pid = String(providerId || "").toLowerCase();
  const lower = String(modelName || "").trim().toLowerCase();
  if (!lower) return false;
  if (
    ["embed", "rerank", "whisper", "tts-", "dall-e", "gpt-image", "imagen", "veo", "grok-imagine-", "moderation"].some(
      (tok) => lower.includes(tok)
    )
  ) {
    return false;
  }
  if (pid === "deepseek") {
    if (role === "vision") {
      return !["deepseek-v4-pro", "deepseek-reasoner", "deepseek-chat"].includes(lower);
    }
    if (role === "llm") {
      return !["deepseek-ocr", "deepseek-ocr-text", "deepseek-ocr-rich", "deepseek-ai/deepseek-ocr"].includes(lower);
    }
  }
  if (pid === "qwen" && role === "llm" && lower.includes("-ocr")) {
    return false;
  }
  return true;
}

export function getCandidateModelsForCredential(cred, spec, discoveredByCred, role) {
  const discovered = (discoveredByCred?.[cred.id] || []).filter((m) =>
    isModelEligibleForRole(cred.provider, m, role)
  );
  const suggested =
    role === "vision" ? spec?.suggested_vision_models || [] : spec?.suggested_llm_models || [];
  // Keep discovered/installed models first, then add suggested defaults without duplicates.
  return [...new Set([...discovered, ...suggested].map((m) => String(m).trim()).filter(Boolean))];
}

/**
 * Populate a grouped `<select>` containing `model (Credential Name)` options plus `Custom model…`.
 */
export function populateRoleModelSelect({
  selectEl,
  customPanelEl,
  customCredEl,
  customInputEl,
  credentials = [],
  registry = [],
  discoveredByCred = {},
  role = "vision",
  current = {},
  includeDisabled = false,
}) {
  if (!selectEl) return;
  const specMap = new Map((registry || []).map((s) => [s.id, s]));
  const eligibleCreds = (credentials || []).filter((c) => {
    const spec = specMap.get(c.provider);
    if (!spec) return false;
    if (role === "vision" && !spec.supports_vision) return false;
    if (role === "llm" && !spec.supports_llm) return false;
    return includeDisabled || c.enabled !== false;
  });

  selectEl.innerHTML = "";
  const availableValues = new Set();

  for (const cred of eligibleCreds) {
    const spec = specMap.get(cred.provider);
    const models = getCandidateModelsForCredential(cred, spec, discoveredByCred, role);
    if (!models.length) continue;
    const group = document.createElement("optgroup");
    group.label = cred.name;
    for (const model of models) {
      const val = encodeModelChoice(cred.id, model);
      availableValues.add(val);
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = `${model} (${cred.name})`;
      group.appendChild(opt);
    }
    selectEl.appendChild(group);
  }

  const customOpt = document.createElement("option");
  customOpt.value = CUSTOM_MODEL_VALUE;
  customOpt.textContent = "Custom model…";
  selectEl.appendChild(customOpt);

  if (customCredEl) {
    customCredEl.innerHTML = "";
    const allEnabled = (credentials || []).filter((c) => {
      const spec = specMap.get(c.provider);
      if (!spec) return false;
      if (role === "vision" && !spec.supports_vision) return false;
      if (role === "llm" && !spec.supports_llm) return false;
      return true;
    });
    for (const cred of allEnabled) {
      const opt = document.createElement("option");
      opt.value = cred.id;
      opt.textContent = cred.enabled === false ? `${cred.name} (Disabled)` : cred.name;
      customCredEl.appendChild(opt);
    }
  }

  const targetCredId = current.credentialId || (eligibleCreds[0]?.id || credentials[0]?.id || "");
  const targetModel = String(current.model || "").trim();
  const targetKey = targetCredId && targetModel ? encodeModelChoice(targetCredId, targetModel) : "";

  if (!current.custom && targetKey && availableValues.has(targetKey)) {
    selectEl.value = targetKey;
    if (customPanelEl) customPanelEl.classList.add("d-none");
  } else if (current.custom || (targetModel && !availableValues.has(targetKey))) {
    selectEl.value = CUSTOM_MODEL_VALUE;
    if (customCredEl && targetCredId) customCredEl.value = targetCredId;
    if (customInputEl) customInputEl.value = targetModel;
    if (customPanelEl) customPanelEl.classList.remove("d-none");
  } else {
    const firstOpt = selectEl.querySelector("optgroup option");
    if (firstOpt) {
      selectEl.value = firstOpt.value;
      if (customPanelEl) customPanelEl.classList.add("d-none");
    } else {
      selectEl.value = CUSTOM_MODEL_VALUE;
      if (customCredEl && targetCredId) customCredEl.value = targetCredId;
      if (customPanelEl) customPanelEl.classList.remove("d-none");
    }
  }
}

export function readRoleModelChoice({ selectEl, customCredEl, customInputEl, credentials = [] }) {
  const credMap = new Map((credentials || []).map((c) => [c.id, c]));
  if (selectEl.value === CUSTOM_MODEL_VALUE) {
    const credentialId = customCredEl?.value || credentials[0]?.id || "";
    const cred = credMap.get(credentialId);
    return {
      custom: true,
      credentialId,
      credentialName: cred?.name || "",
      provider: cred?.provider || "ollama_local",
      model: (customInputEl?.value || "").trim(),
    };
  }
  const decoded = decodeModelChoice(selectEl.value);
  if (!decoded) {
    return { custom: true, credentialId: "", credentialName: "", provider: "ollama_local", model: "" };
  }
  const cred = credMap.get(decoded.credentialId);
  return {
    custom: false,
    credentialId: decoded.credentialId,
    credentialName: cred?.name || "",
    provider: cred?.provider || "ollama_local",
    model: decoded.model,
  };
}
