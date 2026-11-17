/** REST API client for OcrPipeline Desktop & Web App. */

async function requestJson(url, options) {
  const res = await fetch(url, options);
  const text = await res.text();
  const data = text ? JSON.parse(text) : {};
  if (!res.ok) throw new Error(data.detail || text || `HTTP ${res.status}`);
  return data;
}

export const fetchHealth = () => requestJson("/api/health");
export const fetchSettings = () => requestJson("/api/settings");
export const saveSettings = (payload) =>
  requestJson("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const fetchModels = () => requestJson("/api/models");
export const pullModel = (model) => {
  const fd = new FormData();
  fd.append("model", model);
  return requestJson("/api/models/pull", { method: "POST", body: fd });
};
export const deleteModel = (model) =>
  requestJson(`/api/models/${encodeURIComponent(model)}`, { method: "DELETE" });

export const fetchArtifacts = () => requestJson("/api/artifacts");
export const fetchArtifactContent = async (path) => {
  const res = await fetch(`/api/artifacts/content?path=${encodeURIComponent(path)}`);
  if (!res.ok) throw new Error("Failed to load artifact");
  return res.text();
};
export const artifactDownloadUrl = (path) =>
  `/api/artifacts/content?path=${encodeURIComponent(path)}&download=true`;
export const renameArtifact = (path, newName) =>
  requestJson("/api/artifacts/rename", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, new_name: newName }),
  });
export const deleteArtifact = (path) =>
  requestJson(`/api/artifacts?path=${encodeURIComponent(path)}`, { method: "DELETE" });
export const revealArtifact = (path = "") =>
  requestJson("/api/artifacts/reveal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
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
