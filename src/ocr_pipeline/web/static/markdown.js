/** Shared Markdown + LaTeX (KaTeX) + Bootstrap table renderer. */

const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;

function splitFrontmatter(text) {
  const m = FRONTMATTER_RE.exec(text);
  if (!m) return { metaLines: [], body: text };
  const metaLines = m[1]
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"));
  return { metaLines, body: text.slice(m[0].length) };
}

function protectMathAndCode(markdown) {
  const placeholders = [];
  const stash = (raw, isMath = false) => {
    const id = `@@OCR_TOKEN_${placeholders.length}@@`;
    placeholders.push({ id, raw, isMath });
    return id;
  };

  // Preserve fenced and inline code before scanning for math delimiters
  let protectedText = markdown.replace(/```[\s\S]*?```|`[^`\n]+`/g, (m) => stash(m, false));

  // Protect block & inline LaTeX so marked does not eat underscores, asterisks, or backslashes
  protectedText = protectedText.replace(
    /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?!\s)(?:\\.|[^$\\\n])+(?<!\s)\$/g,
    (m) => stash(m, true)
  );

  return { protectedText, placeholders };
}

function restoreTokens(html, placeholders) {
  let output = html;
  for (const token of placeholders) {
    if (token.isMath) {
      const escaped = token.raw
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      output = output.replaceAll(token.id, escaped);
    } else {
      output = output.replaceAll(token.id, () => token.raw);
    }
  }
  return output;
}

function enhanceRenderedDom(container) {
  for (const table of container.querySelectorAll("table")) {
    table.classList.add("table", "table-sm", "table-bordered", "table-striped", "align-middle", "my-3");
    if (!table.parentElement?.classList.contains("table-responsive")) {
      const wrapper = document.createElement("div");
      wrapper.className = "table-responsive";
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    }
  }

  for (const bq of container.querySelectorAll("blockquote")) {
    bq.classList.add("blockquote-ocr");
  }

  if (typeof window.renderMathInElement === "function") {
    window.renderMathInElement(container, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }
}

export function renderMarkdownContent(container, rawText, rawMode, emptyMessage = "No content.") {
  if (!rawText) {
    container.innerHTML = `<span class="text-muted small">${emptyMessage}</span>`;
    return;
  }

  if (rawMode) {
    container.innerHTML = '<pre class="mb-0 small"></pre>';
    container.querySelector("pre").textContent = rawText;
    return;
  }

  const { metaLines, body } = splitFrontmatter(rawText);
  const { protectedText, placeholders } = protectMathAndCode(body);

  if (window.marked?.setOptions) {
    window.marked.setOptions({ gfm: true, breaks: false });
  }
  let html = window.marked ? window.marked.parse(protectedText) : protectedText;
  html = restoreTokens(html, placeholders);

  container.innerHTML = "";
  if (metaLines.length) {
    const bar = document.createElement("div");
    bar.className = "frontmatter-bar small text-muted border rounded px-2 py-1 mb-3 bg-body-tertiary";
    bar.textContent = metaLines.join(" · ");
    container.appendChild(bar);
  }

  const bodyEl = document.createElement("div");
  bodyEl.className = "markdown-body";
  bodyEl.innerHTML = html;
  container.appendChild(bodyEl);

  enhanceRenderedDom(bodyEl);
}
