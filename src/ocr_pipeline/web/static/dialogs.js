/** Reusable Bootstrap 5 modal dialogs replacing native browser alert/confirm/prompt. */

import { createTagInput } from "./tag_input.js";

const el = (id) => document.getElementById(id);
const modalQueues = new WeakMap();

/** Serialize calls per modal element and resolve only after Bootstrap finishes hiding. */
function runModal(modalEl, configure) {
  const prev = modalQueues.get(modalEl) || Promise.resolve();
  const run = prev.catch(() => {}).then(
    () =>
      new Promise((resolve) => {
        const modal = window.bootstrap.Modal.getOrCreateInstance(modalEl);
        let shown = false;
        let hidePending = false;
        let decided = false;
        let result;

        const finish = (val) => {
          if (decided) return;
          decided = true;
          result = val;
          if (shown) modal.hide();
          else hidePending = true;
        };

        const { defaultValue, onShown, bind } = configure(finish);
        result = defaultValue;
        const teardown = bind?.();

        const handleShown = () => {
          shown = true;
          if (hidePending) modal.hide();
          else onShown?.();
        };

        const handleHidden = () => {
          modalEl.removeEventListener("shown.bs.modal", handleShown);
          modalEl.removeEventListener("hidden.bs.modal", handleHidden);
          teardown?.();
          resolve(result);
        };

        modalEl.addEventListener("shown.bs.modal", handleShown);
        modalEl.addEventListener("hidden.bs.modal", handleHidden);
        modal.show();
      })
  );
  modalQueues.set(modalEl, run);
  return run;
}

export function bsConfirm({
  title = "Confirm",
  message = "",
  confirmText = "Confirm",
  variant = "danger",
} = {}) {
  const modalEl = el("bsConfirmModal");
  return runModal(modalEl, (finish) => {
    el("bsConfirmTitle").textContent = title;
    el("bsConfirmMessage").textContent = message;

    const okBtn = el("bsConfirmOkBtn");
    okBtn.textContent = confirmText;
    okBtn.className = `btn btn-sm btn-${variant}`;

    const onOk = () => finish(true);
    return {
      defaultValue: false,
      bind: () => {
        okBtn.addEventListener("click", onOk);
        return () => okBtn.removeEventListener("click", onOk);
      },
    };
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
  const modalEl = el("bsPromptModal");
  return runModal(modalEl, (finish) => {
    el("bsPromptTitle").textContent = title;
    el("bsPromptLabel").textContent = label;
    const input = el("bsPromptInput");
    input.value = value;
    input.placeholder = placeholder;
    input.required = required;
    el("bsPromptOkBtn").textContent = confirmText;

    const form = el("bsPromptForm");
    const onSubmit = (e) => {
      e.preventDefault();
      finish(input.value);
    };
    return {
      defaultValue: null,
      onShown: () => {
        input.focus();
        input.select();
      },
      bind: () => {
        form.addEventListener("submit", onSubmit);
        return () => form.removeEventListener("submit", onSubmit);
      },
    };
  });
}

let _modalTagEditor = null;

export function bsEditTags({ path = "", initialTags = [], suggestions = [] } = {}) {
  const modalEl = el("editTagsModal");
  return runModal(modalEl, (finish) => {
    el("editTagsFileLabel").textContent = path;
    if (!_modalTagEditor) {
      _modalTagEditor = createTagInput(el("editTagsContainer"));
    }
    _modalTagEditor.setSuggestions(suggestions);
    _modalTagEditor.setTags(initialTags);

    const form = el("editTagsForm");
    const onSubmit = (e) => {
      e.preventDefault();
      finish(_modalTagEditor.getTags());
    };
    return {
      defaultValue: null,
      onShown: () => _modalTagEditor.focus(),
      bind: () => {
        form.addEventListener("submit", onSubmit);
        return () => form.removeEventListener("submit", onSubmit);
      },
    };
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
