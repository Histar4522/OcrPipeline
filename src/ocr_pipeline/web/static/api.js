/** REST API client for OcrPipeline Desktop & Web App. */

async function requestJson(url, options) {
  const res = await fetch(url, options);
  const text = await res.text();
  const data = text ? JSON.parse(text) : {};
  if (!res.ok) throw new Error(data.detail || text || `HTTP ${res.status}`);
  return data;
}

export const fetchHealth = () => requestJson("/api/health");
export const fetchProviders = () => requestJson("/api/providers");
export const testProvider = (payload) =>
  requestJson("/api/providers/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const estimateLlmVram = (payload) =>
  requestJson("/api/llm/vram-estimate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const fetchSystemInfo = () => requestJson("/api/system-info");
export const openExternalUrl = (url) =>
  requestJson("/api/open-url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
export const fetchSettings = () => requestJson("/api/settings");
export const saveSettings = (payload) =>
  requestJson("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const checkUpdates = () => requestJson("/api/updates/check");
export const installUpdate = (payload) =>
  requestJson("/api/updates/install", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const fetchUpdateInstallStatus = () => requestJson("/api/updates/install-status");

export const fetchModels = () => requestJson("/api/models");
export const startOllama = () => requestJson("/api/ollama/start", { method: "POST" });
export const pullModel = (model) => {
  const fd = new FormData();
  fd.append("model", model);
  return requestJson("/api/models/pull", { method: "POST", body: fd });
};
export const startModelPull = (model) => {
  const fd = new FormData();
  fd.append("model", model);
  return requestJson("/api/models/pull-async", { method: "POST", body: fd });
};
export const fetchModelPullStatus = () => requestJson("/api/models/pull-status");
export const deleteModel = (model) =>
  requestJson(`/api/models/${encodeURIComponent(model)}`, { method: "DELETE" });

export const fetchFolders = () => requestJson("/api/folders");
export const fetchArtifacts = ({ cwd = "", q = "", tag = "", sort = "newest" } = {}) => {
  const params = new URLSearchParams({ cwd, q, tag, sort });
  return requestJson(`/api/artifacts?${params.toString()}`);
};
export const createArtifactFolder = (folder) =>
  requestJson("/api/artifacts/folder", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder }),
  });
export const renameArtifactFolder = (folder, newName) =>
  requestJson("/api/artifacts/folder", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder, new_name: newName }),
  });
export const deleteArtifactFolder = (folder, recursive = false) =>
  requestJson(`/api/artifacts/folder?folder=${encodeURIComponent(folder)}&recursive=${recursive}`, {
    method: "DELETE",
  });
export const fetchArtifactContent = async (path) => {
  const res = await fetch(`/api/artifacts/content?path=${encodeURIComponent(path)}`);
  if (!res.ok) throw new Error("Failed to load artifact");
  return res.text();
};
export const artifactDownloadUrl = (path) =>
  `/api/artifacts/content?path=${encodeURIComponent(path)}&download=true`;
export const moveOrRenameArtifact = (path, { newName, targetFolder } = {}) =>
  requestJson("/api/artifacts/move", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      path,
      ...(newName !== undefined ? { new_name: newName } : {}),
      ...(targetFolder !== undefined ? { target_folder: targetFolder } : {}),
    }),
  });
export const updateArtifactTags = (path, tags) =>
  requestJson("/api/artifacts/tags", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, tags }),
  });
export const openInObsidian = (path = "") =>
  requestJson("/api/artifacts/obsidian", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
export const deleteArtifact = (path) =>
  requestJson(`/api/artifacts?path=${encodeURIComponent(path)}`, { method: "DELETE" });
export const revealArtifact = (path = "") =>
  requestJson("/api/artifacts/reveal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
export const saveArtifactContent = (path, content) =>
  requestJson("/api/artifacts/content", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, content }),
  });
export const aiSuggestTags = (path, apply = true) =>
  requestJson("/api/artifacts/ai/tags", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, apply }),
  });
export const aiSuggestRename = (path) =>
  requestJson("/api/artifacts/ai/rename", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
export const aiSummarizeArtifact = (path) =>
  requestJson("/api/artifacts/ai/summarize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
export const aiCleanupArtifact = (path) =>
  requestJson("/api/artifacts/ai/cleanup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
export const aiSummarizeFolder = (cwd = "") =>
  requestJson("/api/artifacts/ai/folder-summary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cwd }),
  });

export const fetchJobs = () => requestJson("/api/jobs");
export const fetchJob = (jobId) => requestJson(`/api/jobs/${encodeURIComponent(jobId)}`);
export const startJob = (formData) => requestJson("/api/jobs", { method: "POST", body: formData });
export const cancelJob = (jobId) =>
  requestJson(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
export const fetchJobLog = async (jobId) => {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/log`);
  return res.text();
};
export const fetchJobFile = async (jobId, filePath) => {
  const encoded = filePath.split("/").map(encodeURIComponent).join("/");
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/files/${encoded}`);
  if (!res.ok) throw new Error("Failed to load file");
  return res.text();
};
export const fileDownloadUrl = (jobId, filePath) => {
  const encoded = filePath.split("/").map(encodeURIComponent).join("/");
  return `/api/jobs/${encodeURIComponent(jobId)}/files/${encoded}?download=true`;
};
