/** Reusable Bootstrap 5 tag chip/pill editor component. */

function normalizeTag(raw) {
  return String(raw || "").trim().replace(/^#+/, "").replace(/\s+/g, "-");
}

export function createTagInput(container, { placeholder = "Type a tag and press Enter...", maxSuggestions = 3 } = {}) {
  let tags = [];
  let suggestions = [];

  container.innerHTML = `
    <div class="border rounded p-2 bg-body">
      <div class="d-flex flex-wrap align-items-center gap-1 mb-2 tag-chips"></div>
      <div class="input-group input-group-sm">
        <input type="text" class="form-control tag-draft-input" placeholder="${placeholder}" />
        <button type="button" class="btn btn-outline-primary tag-add-btn">Add</button>
        <button type="button" class="btn btn-outline-secondary tag-clear-btn d-none">Clear All</button>
      </div>
      <div class="d-flex flex-wrap gap-1 mt-2 d-none tag-suggestions"></div>
    </div>`;

  const chipsEl = container.querySelector(".tag-chips");
  const inputEl = container.querySelector(".tag-draft-input");
  const addBtn = container.querySelector(".tag-add-btn");
  const clearBtn = container.querySelector(".tag-clear-btn");
  const suggEl = container.querySelector(".tag-suggestions");

  function render() {
    chipsEl.innerHTML = "";
    if (!tags.length) {
      chipsEl.innerHTML = '<span class="small text-muted">No tags</span>';
    } else {
      for (const tag of tags) {
        const pill = document.createElement("span");
        pill.className = "badge rounded-pill text-bg-primary d-inline-flex align-items-center gap-1 py-1 px-2";
        const label = document.createElement("span");
        label.textContent = `#${tag}`;
        const rm = document.createElement("button");
        rm.type = "button";
        rm.className = "btn-close btn-close-white";
        rm.style.fontSize = "0.55rem";
        rm.ariaLabel = `Remove ${tag}`;
        rm.addEventListener("click", () => {
          tags = tags.filter((t) => t !== tag);
          render();
        });
        pill.append(label, rm);
        chipsEl.appendChild(pill);
      }
    }
    clearBtn.classList.toggle("d-none", tags.length === 0);

    const available = suggestions.filter((s) => !tags.includes(s));
    suggEl.innerHTML = "";
    suggEl.classList.toggle("d-none", available.length === 0);
    if (available.length) {
      const lbl = document.createElement("span");
      lbl.className = "small text-muted me-1 align-self-center";
      lbl.textContent = "Suggestions:";
      suggEl.appendChild(lbl);
      for (const s of available.slice(0, maxSuggestions)) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "btn btn-xs btn-outline-secondary rounded-pill";
        btn.textContent = `+ #${s}`;
        btn.addEventListener("click", () => addTag(s));
        suggEl.appendChild(btn);
      }
    }
  }

  function addTag(raw, clearInput = false) {
    const clean = normalizeTag(raw);
    if (!clean) return false;
    if (!tags.includes(clean)) tags = [...tags, clean];
    if (clearInput) inputEl.value = "";
    render();
    return true;
  }

  function commitDraft() {
    if (!inputEl.value.trim()) return;
    for (const part of inputEl.value.split(",")) addTag(part, false);
    inputEl.value = "";
    render();
    inputEl.focus();
  }

  addBtn.addEventListener("click", (e) => {
    e.preventDefault();
    commitDraft();
  });
  clearBtn.addEventListener("click", () => {
    tags = [];
    inputEl.value = "";
    render();
  });
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      commitDraft();
    } else if (e.key === "Backspace" && !inputEl.value && tags.length) {
      tags = tags.slice(0, -1);
      render();
    }
  });

  render();

  return {
    getTags: () => {
      commitDraft();
      return [...tags];
    },
    setTags: (next) => {
      tags = [];
      for (const item of next || []) {
        const clean = normalizeTag(item);
        if (clean && !tags.includes(clean)) tags.push(clean);
      }
      inputEl.value = "";
      render();
    },
    setSuggestions: (next) => {
      suggestions = (next || []).map(normalizeTag).filter(Boolean);
      render();
    },
    focus: () => inputEl.focus(),
  };
}
