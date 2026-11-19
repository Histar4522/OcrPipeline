import {
  checkUpdates,
  fetchUpdateInstallStatus,
  installUpdate,
  openExternalUrl,
  saveSettings,
} from "./api.js";
import { renderMarkdownContent } from "./markdown.js";

const el = (id) => document.getElementById(id);

function openUrl(url) {
  if (!url) return;
  openExternalUrl(url).catch(() => window.open(url, "_blank", "noopener,noreferrer"));
}

export function initUpdatesView({ showToast, onOpenUpdateSection }) {
  const updState = {
    data: null,
    pollTimer: null,
    autoInstalledTag: null,
  };

  function stopPolling() {
    if (updState.pollTimer) {
      clearInterval(updState.pollTimer);
      updState.pollTimer = null;
    }
  }

  function renderSelectedReleasePreview(tag) {
    const releases = updState.data?.releases || [];
    const rel = releases.find((r) => r.tag_name === tag) || releases[0] || null;
    const previewEl = el("releaseHistoryPreview");
    const metaEl = el("previewReleaseMeta");
    const linkBtn = el("previewReleaseLink");
    if (!rel) {
      metaEl.textContent = "Check for updates to load GitHub release notes.";
      linkBtn.classList.add("d-none");
      renderMarkdownContent(previewEl, "", false, "No release notes loaded yet.");
      return;
    }
    const assetNote = rel.asset ? ` · Installer: ${rel.asset.name}` : "";
    const statusNote = rel.is_current ? " · Installed" : rel.is_newer ? " · New release" : "";
    metaEl.textContent = `${rel.tag_name}${rel.published_at ? ` · ${rel.published_at}` : ""}${statusNote}${assetNote}`;
    linkBtn.classList.toggle("d-none", !rel.html_url);
    linkBtn.dataset.url = rel.html_url || "";
    renderMarkdownContent(previewEl, rel.body || "_No release notes provided._", false);
  }

  function renderResult(data) {
    updState.data = data;
    el("updateCurrentVersion").textContent = `v${data.current_version}`;
    el("updateLastChecked").textContent = data.checked_at ? `Last checked: ${data.checked_at}` : "";

    const badge = el("updateStatusBadge");
    const navBtn = el("navUpdateBtn");
    const newBox = el("newReleaseBox");

    if (data.update_available && data.latest) {
      badge.className = "badge text-bg-warning text-dark";
      badge.textContent = `Update Available: ${data.latest.tag_name}`;
      navBtn.classList.remove("d-none");
      navBtn.textContent = `Update ${data.latest.tag_name}`;

      newBox.classList.remove("d-none");
      el("newReleaseTitle").textContent = data.latest.name || data.latest.tag_name;
      const mb = data.latest.asset?.size ? ` (${(data.latest.asset.size / (1024 * 1024)).toFixed(1)} MB)` : "";
      el("newReleaseMeta").textContent = `Latest: ${data.latest.tag_name}${data.latest.published_at ? ` · Published ${data.latest.published_at}` : ""}${data.latest.asset ? ` · ${data.latest.asset.name}${mb}` : ""}`;
      el("newReleaseLink").dataset.url = data.latest.html_url || "";
      const installBtn = el("installUpdateBtn");
      installBtn.textContent = data.latest.asset?.download_url ? "Download & Install Update" : "Open Release Page";
      renderMarkdownContent(el("newReleaseNotes"), data.latest.body || "_No release notes provided._", false);
    } else {
      badge.className = "badge text-bg-success";
      badge.textContent = "Up to date";
      navBtn.classList.add("d-none");
      newBox.classList.add("d-none");
    }

    const sel = el("releaseHistorySelect");
    const prev = sel.value;
    sel.innerHTML = "";
    const releases = data.releases || [];
    for (const r of releases) {
      const opt = document.createElement("option");
      opt.value = r.tag_name;
      const tagBadge = r.is_current ? " (Current)" : r.is_newer ? " (New)" : "";
      opt.textContent = `${r.tag_name}${r.published_at ? ` · ${r.published_at}` : ""}${tagBadge}`;
      sel.appendChild(opt);
    }
    if (releases.length) {
      const defaultTag = releases.some((r) => r.tag_name === prev)
        ? prev
        : (data.latest?.tag_name || releases[0].tag_name);
      sel.value = defaultTag;
      renderSelectedReleasePreview(defaultTag);
    } else {
      renderSelectedReleasePreview("");
    }
  }

  async function startInstall(release) {
    if (!release?.asset?.download_url) {
      openUrl(release?.html_url);
      return;
    }
    const panel = el("updateProgressPanel");
    const bar = el("updateProgressBar");
    const statusText = el("updateProgressText");
    const pctText = el("updateProgressPercent");
    const installBtn = el("installUpdateBtn");

    panel.classList.remove("d-none");
    bar.className = "progress-bar progress-bar-striped progress-bar-animated";
    bar.style.width = "0%";
    pctText.textContent = "0%";
    statusText.textContent = `Downloading ${release.asset.name}...`;
    installBtn.disabled = true;

    try {
      await installUpdate({
        tag: release.tag_name,
        download_url: release.asset.download_url,
        asset_name: release.asset.name,
      });
    } catch (err) {
      bar.className = "progress-bar bg-danger";
      statusText.textContent = err.message;
      installBtn.disabled = false;
      showToast("Update Failed", err.message, "danger");
      return;
    }

    stopPolling();
    updState.pollTimer = setInterval(async () => {
      try {
        const st = await fetchUpdateInstallStatus();
        const pct = Math.min(100, Math.round(st.percent || 0));
        const sizeNote = st.total_mb > 0 ? ` (${st.completed_mb} / ${st.total_mb} MB)` : "";
        bar.style.width = `${pct}%`;
        pctText.textContent = `${pct}%`;
        if (!st.done) {
          statusText.textContent = `Downloading ${release.asset.name}${sizeNote}...`;
          return;
        }
        stopPolling();
        installBtn.disabled = false;
        if (st.error) {
          bar.className = "progress-bar bg-danger";
          statusText.textContent = st.error;
          showToast("Update Failed", st.error, "danger");
        } else {
          bar.className = "progress-bar bg-success";
          statusText.textContent = st.message || "Update ready.";
          showToast("Update Ready", st.message || "Installer launched.", "success");
        }
      } catch {
        stopPolling();
        installBtn.disabled = false;
      }
    }, 500);
  }

  async function check({ silent = false, autoUpdate = false } = {}) {
    const btn = el("checkUpdatesBtn");
    const errBox = el("updateErrorBox");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Checking...';
    errBox.classList.add("d-none");

    try {
      const data = await checkUpdates();
      renderResult(data);
      if (data.update_available && data.latest) {
        if (autoUpdate && data.latest.asset?.download_url && updState.autoInstalledTag !== data.latest.tag_name) {
          updState.autoInstalledTag = data.latest.tag_name;
          showToast("Auto-Updating", `Downloading ${data.latest.tag_name}...`, "info");
          await startInstall(data.latest);
        } else if (silent) {
          showToast("Update Available", `${data.latest.tag_name} is available in Settings → Update.`, "info");
        }
      } else if (!silent) {
        showToast("Up to Date", `OcrPipeline v${data.current_version} is the latest release.`, "success");
      }
    } catch (err) {
      if (!silent) {
        errBox.textContent = err.message;
        errBox.classList.remove("d-none");
      }
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i>Check for Updates';
    }
  }

  function syncSettingsForm(s) {
    el("setAutoCheckUpdates").checked = s.auto_check_updates !== false;
    el("setAutoUpdate").checked = Boolean(s.auto_update);
  }

  el("checkUpdatesBtn").addEventListener("click", () => check({ silent: false, autoUpdate: false }));
  el("releaseHistorySelect").addEventListener("change", (e) => renderSelectedReleasePreview(e.target.value));
  el("installUpdateBtn").addEventListener("click", () => {
    if (updState.data?.latest) startInstall(updState.data.latest);
  });
  el("newReleaseLink").addEventListener("click", () => openUrl(el("newReleaseLink").dataset.url));
  el("previewReleaseLink").addEventListener("click", () => openUrl(el("previewReleaseLink").dataset.url));
  el("navUpdateBtn").addEventListener("click", () => {
    if (onOpenUpdateSection) onOpenUpdateSection();
  });

  el("setAutoCheckUpdates").addEventListener("change", async (e) => {
    await saveSettings({ auto_check_updates: e.target.checked });
    showToast("Update Settings", "Saved startup update check preference.");
  });
  el("setAutoUpdate").addEventListener("change", async (e) => {
    await saveSettings({ auto_update: e.target.checked });
    showToast("Update Settings", e.target.checked ? "Auto-update enabled." : "Auto-update disabled.");
  });

  renderSelectedReleasePreview("");
  return { check, syncSettingsForm, getData: () => updState.data };
}
