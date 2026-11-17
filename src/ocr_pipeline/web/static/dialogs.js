/** Reusable Bootstrap 5 modal dialogs replacing native browser alert/confirm/prompt. */

import { createTagInput } from "./tag_input.js";

const el = (id) => document.getElementById(id);

export function bsConfirm({
  title = "Confirm",
  message = "",
  confirmText = "Confirm",
  variant = "danger",
} = {}) {
  return new Promise((resolve) => {
    const modalEl = el("bsConfirmModal");
    const modal = window.bootstrap.Modal.getOrCreateInstance(modalEl);
    el("bsConfirmTitle").textContent = title;
    el("bsConfirmMessage").textContent = message;

    const okBtn = el("bsConfirmOkBtn");
    okBtn.textContent = confirmText;
    okBtn.className = `btn btn-sm btn-${variant}`;

    let settled = false;
    const cleanup = (result) => {
      if (settled) return;
      settled = true;
      okBtn.removeEventListener("click", onOk);
      modalEl.removeEventListener("hidden.bs.modal", onHidden);
      resolve(result);
    };
    const onOk = () => {
      cleanup(true);
      modal.hide();
    };
    const onHidden = () => cleanup(false);

    okBtn.addEventListener("click", onOk);
    modalEl.addEventListener("hidden.bs.modal", onHidden);
    modal.show();
  });
}

export function bsPrompt({
  title = "Input",
  label = "",
  value = "",
  placeholder = "",
  confirmText = "Save",
  required = true,
} = {}) {
  return new Promise((resolve) => {
    const modalEl = el("bsPromptModal");
    const modal = window.bootstrap.Modal.getOrCreateInstance(modalEl);
    el("bsPromptTitle").textContent = title;
    el("bsPromptLabel").textContent = label;
    const input = el("bsPromptInput");
    input.value = value;
    input.placeholder = placeholder;
    input.required = required;
    el("bsPromptOkBtn").textContent = confirmText;

    const form = el("bsPromptForm");
    let settled = false;
    const cleanup = (result) => {
      if (settled) return;
      settled = true;
      form.removeEventListener("submit", onSubmit);
      modalEl.removeEventListener("hidden.bs.modal", onHidden);
      modalEl.removeEventListener("shown.bs.modal", onShown);
      resolve(result);
    };
    const onSubmit = (e) => {
      e.preventDefault();
      cleanup(input.value);
      modal.hide();
    };
    const onHidden = () => cleanup(null);
    const onShown = () => {
      input.focus();
      input.select();
    };

    form.addEventListener("submit", onSubmit);
    modalEl.addEventListener("hidden.bs.modal", onHidden);
    modalEl.addEventListener("shown.bs.modal", onShown);
    modal.show();
  });
}

export function bsMoveArtifact({ path = "", currentFolder = "", folders = [] } = {}) {
  return new Promise((resolve) => {
    const modalEl = el("moveArtifactModal");
    const modal = window.bootstrap.Modal.getOrCreateInstance(modalEl);
    el("moveArtifactFileLabel").textContent = path;

    const select = el("moveFolderSelect");
    const newInput = el("moveNewFolderInput");
    newInput.value = "";
    select.innerHTML = '<option value="">(Archive Root)</option>';
    for (const folder of folders) {
      const opt = document.createElement("option");
      opt.value = folder;
      opt.textContent = folder;
      select.appendChild(opt);
    }
    select.value = folders.includes(currentFolder) ? currentFolder : "";

    const form = el("moveArtifactForm");
    let settled = false;
    const cleanup = (result) => {
      if (settled) return;
      settled = true;
      form.removeEventListener("submit", onSubmit);
      modalEl.removeEventListener("hidden.bs.modal", onHidden);
      resolve(result);
    };
    const onSubmit = (e) => {
      e.preventDefault();
      const custom = newInput.value.trim();
      cleanup(custom || select.value);
      modal.hide();
    };
    const onHidden = () => cleanup(null);

    form.addEventListener("submit", onSubmit);
    modalEl.addEventListener("hidden.bs.modal", onHidden);
    modal.show();
  });
}

let _modalTagEditor = null;

export function bsEditTags({ path = "", initialTags = [], suggestions = [] } = {}) {
  return new Promise((resolve) => {
    const modalEl = el("editTagsModal");
    const modal = window.bootstrap.Modal.getOrCreateInstance(modalEl);
    el("editTagsFileLabel").textContent = path;
    if (!_modalTagEditor) {
      _modalTagEditor = createTagInput(el("editTagsContainer"));
    }
    _modalTagEditor.setSuggestions(suggestions);
    _modalTagEditor.setTags(initialTags);

    const form = el("editTagsForm");
    let settled = false;
    const cleanup = (result) => {
      if (settled) return;
      settled = true;
      form.removeEventListener("submit", onSubmit);
      modalEl.removeEventListener("hidden.bs.modal", onHidden);
      modalEl.removeEventListener("shown.bs.modal", onShown);
      resolve(result);
    };
    const onSubmit = (e) => {
      e.preventDefault();
      cleanup(_modalTagEditor.getTags());
      modal.hide();
    };
    const onHidden = () => cleanup(null);
    const onShown = () => _modalTagEditor.focus();

    form.addEventListener("submit", onSubmit);
    modalEl.addEventListener("hidden.bs.modal", onHidden);
    modalEl.addEventListener("shown.bs.modal", onShown);
    modal.show();
  });
}

export async function triggerDownload(url, defaultFilename, showToast) {
  if (window.pywebview?.api?.save_url) {
    try {
      const res = await window.pywebview.api.save_url(url, defaultFilename);
      if (res?.saved && showToast) showToast("Saved", res.path, "success");
      else if (res?.error && showToast) showToast("Save Failed", res.error, "danger");
    } catch (err) {
      if (showToast) showToast("Save Failed", err?.message || String(err), "danger");
    }
    return;
  }
  const link = document.createElement("a");
  link.href = url;
  link.download = defaultFilename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}
