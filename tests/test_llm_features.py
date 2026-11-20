"""Tests for optional LLM-powered artifact tagging, renaming, organization, summary & cleanup."""

import json
from pathlib import Path
import threading

import httpx
import pytest

from ocr_pipeline.llm_context import (
    calculate_vram_plan,
    choose_adequate_context,
    compute_cleanup_budgets,
    estimate_model_max_output_tokens,
    inspect_ollama_model_specs,
    split_markdown_chunks,
    split_markdown_pages_for_cleanup,
)
from ocr_pipeline.ollama_client import JobCancelledError
from ocr_pipeline.artifacts import scan_artifacts, update_artifact_summary
from ocr_pipeline.llm_features import (
    _parse_json_object,
    analyze_artifact_all,
    cleanup_ocr_markdown,
    suggest_filename,
    suggest_tags,
    summarize_document,
    summarize_folder_contents,
)
from ocr_pipeline.providers import ProviderEndpoint


def test_parse_json_strips_think_and_fences() -> None:
    raw = '<think>reasoning...</think>\n```json\n{"tags": ["physics", "optics"]}\n```'
    assert _parse_json_object(raw) == {"tags": ["physics", "optics"]}


def test_llm_artifact_helpers(tmp_path: Path, monkeypatch: object) -> None:
    ep = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b")

    monkeypatch.setattr(
        "ocr_pipeline.llm_features.chat_text",
        lambda endpoint, sys, usr, **kw: json.dumps({"tags": ["Calculus", "#Math Notes", "calculus"]}),
    )  # type: ignore[attr-defined]
    assert suggest_tags("# Integrals", ["calculus"], ep) == ["calculus", "math-notes"]

    monkeypatch.setattr(
        "ocr_pipeline.llm_features.chat_text",
        lambda endpoint, sys, usr, **kw: json.dumps({"filename": "Lecture 04: Wave/Optics?.md"}),
    )  # type: ignore[attr-defined]
    assert suggest_filename("# Wave Optics", "scan1.md", ep) == "Lecture_04_Wave_Optics.md"

    monkeypatch.setattr(
        "ocr_pipeline.llm_features.chat_text",
        lambda endpoint, sys, usr, **kw: json.dumps({"summary": "Introduces Snell's law.\nExplains refraction."}),
    )  # type: ignore[attr-defined]
    assert summarize_document("# Refraction", ep) == "Introduces Snell's law. Explains refraction."

    monkeypatch.setattr(
        "ocr_pipeline.llm_features.chat_text",
        lambda endpoint, sys, usr, **kw: "```markdown\n# Clean Title\nFixed paragraph.\n```",
    )  # type: ignore[attr-defined]
    cleaned = cleanup_ocr_markdown("---\ntitle: \"Sample\"\n---\n\n# Clean   Ttile\nFixed\nparagraph.", ep)
    assert 'title: "Sample"' in cleaned
    assert "# Clean Title\nFixed paragraph." in cleaned


def test_ai_rename_avoids_uuid_echo_and_retries(monkeypatch: object) -> None:
    ep = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b")
    uuid_name = "2f20d00c-e300-4604-8270-a005313f78b5.md"
    seen_prompts: list[str] = []

    def fake_chat(endpoint: ProviderEndpoint, sys: str, usr: str, **kw: object) -> str:
        seen_prompts.append(usr)
        if len(seen_prompts) == 1:
            return json.dumps({"filename": uuid_name})
        return json.dumps({"title": "Race and Antiracist Essay"})

    monkeypatch.setattr("ocr_pipeline.llm_features.chat_text", fake_chat)  # type: ignore[attr-defined]
    suggested = suggest_filename("RACE: A power construct of collected difference.", uuid_name, ep)
    # Initial prompt never exposes current UUID filename; retry produces clean title
    assert uuid_name not in seen_prompts[0]
    assert suggested == "Race_and_Antiracist_Essay.md"

    # Also retries if suggest_tags returns placeholder slugs like ["topic-one", "topic-two"]
    tag_calls = 0

    def fake_tag_retry(endpoint: ProviderEndpoint, sys: str, usr: str, **kw: object) -> str:
        nonlocal tag_calls
        tag_calls += 1
        if tag_calls == 1:
            return json.dumps({"tags": ["topic-one", "topic-two"]})
        return json.dumps({"keywords": ["spanish grammar", "verb_conjugation"]})

    monkeypatch.setattr("ocr_pipeline.llm_features.chat_text", fake_tag_retry)  # type: ignore[attr-defined]
    assert suggest_tags("Spanish ser vs estar study guide.", [], ep) == ["spanish-grammar", "verb-conjugation"]
    assert tag_calls == 2


def test_analyze_and_folder_content_summary(tmp_path: Path, monkeypatch: object) -> None:
    archive = tmp_path / "archive"
    (archive / "Chemistry").mkdir(parents=True)
    acid = archive / "Chemistry" / "acid.md"
    acid.write_text("# Acid Base Titration\npH curves and buffers", encoding="utf-8")
    update_artifact_summary(acid, "Covers pH titration curves.")
    bases = archive / "Chemistry" / "bases.md"
    bases.write_text("# Weak Bases\nKb equilibrium constants", encoding="utf-8")

    ep = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b")
    calls: list[str] = []

    def fake_chat(endpoint: ProviderEndpoint, sys: str, usr: str, **kw: object) -> str:
        calls.append(sys)
        if kw.get("json_mode"):
            if "filename" in sys:
                return json.dumps(
                    {
                        "tags": ["chemistry", "titration"],
                        "filename": "Acid_Base_Titration.md",
                        "summary": "Covers acid-base titration curves.",
                    }
                )
            return json.dumps({"summary": "Explains Kb base equilibrium."})
        return "### Overview\nFolder covers acid-base equilibria.\n\n### Key Topics\n- Titration\n- Kb"

    monkeypatch.setattr("ocr_pipeline.llm_features.chat_text", fake_chat)  # type: ignore[attr-defined]

    single = analyze_artifact_all(
        "# Acid Base Titration",
        current_name="raw_scan.md",
        known_tags=[],
        endpoint=ep,
    )
    assert single["filename"] == "Acid_Base_Titration.md"
    assert single["tags"] == ["chemistry", "titration"]

    calls.clear()
    folder_res = summarize_folder_contents(archive, "Chemistry", ep)
    # Reused acid.md.meta summary, generated summary for bases.md, plus 1 folder synthesis call = 2 calls
    assert len(calls) == 2
    assert folder_res["doc_count"] == 2
    assert "### Overview" in folder_res["summary"]
    assert (archive / "Chemistry" / "bases.md.meta").is_file()
    assert (archive / "Chemistry" / ".folder.meta").is_file()
    scanned = scan_artifacts(archive, cwd="Chemistry")
    assert scanned["folder_summary"]["doc_count"] == 2


def test_vram_calculation_and_context_selection() -> None:
    gpu_8gb = {"gpu_name": "NVIDIA RTX 4060", "vram_gb": 8.0, "ram_gb": 16.0}
    gpu_4gb = {"gpu_name": "NVIDIA GTX 1650", "vram_gb": 4.0, "ram_gb": 16.0}

    ep_small = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b", auto_ctx=True)
    plan_8 = calculate_vram_plan(ep_small, system_info=gpu_8gb)
    plan_4 = calculate_vram_plan(ep_small, system_info=gpu_4gb)
    assert plan_8.fits_in_vram is True
    assert plan_8.recommended_max_ctx >= 24576
    assert plan_4.recommended_max_ctx < plan_8.recommended_max_ctx
    assert plan_4.vram_at_recommended_gb <= plan_4.budget_gb  # type: ignore[operator]

    # A 7,500-token document (~27,000 ASCII chars) should automatically scale up num_ctx on 8 GB VRAM
    medium_doc = ("Linear algebra eigenvalues and eigenvectors. " * 620).strip()
    sel = choose_adequate_context(ep_small, medium_doc, expected_output_tokens=512, system_info=gpu_8gb)
    assert sel.chosen_num_ctx >= 12288
    assert sel.needs_splitting is False

    # Manual auto_ctx=False respects manual cap and triggers map-reduce splitting instead
    ep_manual = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b", num_ctx=4096, auto_ctx=False)
    sel_manual = choose_adequate_context(ep_manual, medium_doc, expected_output_tokens=512, system_info=gpu_8gb)
    assert sel_manual.chosen_num_ctx == 4096
    assert sel_manual.needs_splitting is True

    # Cloud endpoint consumes 0 GB local VRAM
    ep_cloud = ProviderEndpoint("openai", "https://api.openai.com/v1", "sk-test", "gpt-4.1-mini")
    cloud_plan = calculate_vram_plan(ep_cloud, system_info=gpu_4gb)
    assert cloud_plan.is_local is False
    assert cloud_plan.vram_at_recommended_gb == 0.0

    # Ollama /api/show inspection calculates exact FP16 GQA KV cache bytes/token
    def show_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "details": {"parameter_size": "1.7B", "quantization_level": "Q4_K_M"},
                "model_info": {
                    "general.architecture": "qwen3",
                    "qwen3.block_count": 28,
                    "qwen3.attention.head_count_kv": 8,
                    "qwen3.attention.key_length": 128,
                    "qwen3.context_length": 40960,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(show_handler)) as client:
        specs = inspect_ollama_model_specs("http://127.0.0.1:59998", "qwen3:1.7b", client=client)
    assert specs["kv_bytes_per_token"] == 2 * 28 * 8 * 128 * 2
    assert specs["context_length"] == 40960


def test_split_markdown_chunks_preserves_code_and_headings() -> None:
    doc = (
        "# Chapter 1: Physics\n\n"
        "Intro paragraph.\n\n"
        "## Section 1.2: Numerics\n\n"
        "```python\n" + ("x = x + 1\n" * 60) + "```\n\n"
        + ("Wave interference equation explanation. " * 220)
    )
    chunks = split_markdown_chunks(doc, max_chunk_tokens=450, overlap_tokens=60)
    assert len(chunks) >= 2
    # Code fence should remain intact inside a single chunk
    code_chunks = [c["text"] for c in chunks if "```python" in c["text"]]
    assert len(code_chunks) == 1
    assert code_chunks[0].count("```") == 2
    assert any("Section 1.2" in c["heading_context"] for c in chunks)


def test_large_document_split_and_summarize_and_analyze(monkeypatch: object) -> None:
    sections = []
    for idx in range(1, 16):
        topic = "MIDTERM_SPECIAL_TOPIC_FOURIER" if idx == 8 else f"Topic {idx}"
        sections.append(f"## Section {idx}: {topic}\n\n" + (f"Detailed notes for {topic}. " * 120))
    huge_doc = "---\ntitle: \"Huge Textbook\"\n---\n\n" + "\n\n".join(sections)

    ep = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b", num_ctx=4096, auto_ctx=False)
    prompts_seen: list[tuple[int, str, str]] = []

    def fake_chat(endpoint: ProviderEndpoint, system: str, user: str, **kw: object) -> str:
        prompts_seen.append((endpoint.num_ctx, system, user))
        if kw.get("json_mode"):
            if "tags" in system and "filename" in system:
                return json.dumps(
                    {
                        "tags": ["fourier", "signal-processing"],
                        "filename": "Huge_Textbook_Notes.md",
                        "folder": "",
                        "summary": "Covers all 15 sections including Fourier analysis.",
                        "reason": "Root",
                    }
                )
            return json.dumps({"summary": "Synthesized summary across all 15 sections."})
        return "Section covers key mathematical concepts."

    monkeypatch.setattr("ocr_pipeline.llm_features.chat_text", fake_chat)  # type: ignore[attr-defined]
    summary = summarize_document(huge_doc, ep, system_info={"vram_gb": 4.0, "ram_gb": 16.0})
    assert summary == "Synthesized summary across all 15 sections."
    # Map phase ran across chunks AND captured middle section 8 (not truncated!)
    assert len(prompts_seen) >= 4
    assert any("MIDTERM_SPECIAL_TOPIC_FOURIER" in usr for _, _, usr in prompts_seen)

    prompts_seen.clear()
    analysis = analyze_artifact_all(
        huge_doc,
        current_name="scan.md",
        known_tags=["fourier"],
        existing_folders=[],
        endpoint=ep,
        system_info={"vram_gb": 4.0, "ram_gb": 16.0},
    )
    assert "fourier" in analysis["tags"]
    assert analysis["filename"] == "Huge_Textbook_Notes.md"
    assert len(prompts_seen) >= 4


def test_large_document_chunked_cleanup(monkeypatch: object) -> None:
    pages = [
        f"<!-- Page {i} -->\n\nParagraph {i} with broken\nline wrap and typo ttitle." * 25
        for i in range(1, 14)
    ]
    huge_raw = "---\ntitle: \"Long OCR\"\n---\n\n" + "\n\n".join(pages)

    ep = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b", num_ctx=4096, auto_ctx=False)
    cleanup_calls: list[str] = []

    def fake_cleanup(endpoint: ProviderEndpoint, system: str, user: str, **kw: object) -> str:
        cleanup_calls.append(user)
        return user.replace("ttitle", "title")

    monkeypatch.setattr("ocr_pipeline.llm_features.chat_text", fake_cleanup)  # type: ignore[attr-defined]
    cleaned = cleanup_ocr_markdown(huge_raw, ep, system_info={"vram_gb": 4.0, "ram_gb": 16.0})
    assert len(cleanup_calls) >= 3
    assert 'title: "Long OCR"' in cleaned
    assert "<!-- Page 1 -->" in cleaned
    assert "<!-- Page 13 -->" in cleaned
    assert "ttitle" not in cleaned
    assert "typo title." in cleaned


def test_cleanup_budgets_output_caps_and_auto_ctx_downsizing() -> None:
    gpu_8gb = {"gpu_name": "NVIDIA RTX 4060", "vram_gb": 8.0, "ram_gb": 16.0}
    assert estimate_model_max_output_tokens("openai", "gpt-4.1-mini", 32768) == 16384
    assert estimate_model_max_output_tokens("openai", "gpt-4-turbo", 32768) == 4096
    assert estimate_model_max_output_tokens("anthropic", "claude-haiku-4-5", 32768) == 8192
    assert estimate_model_max_output_tokens("ollama", "qwen3:1.7b", 16384) == 4096
    assert estimate_model_max_output_tokens("ollama", "qwen2.5:7b", 16384) == 8192

    long_body = ("Quadratic reciprocity theorem proof and numerical example.\n" * 600).strip()

    # Local 1.7B model with auto_ctx=True is output-capped at 4096 tokens, so num_ctx downsizes to 8192 to save VRAM.
    ep_local = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b", auto_ctx=True)
    local_budget = compute_cleanup_budgets(ep_local, long_body, system_info=gpu_8gb)
    assert local_budget.max_output_tokens == 4096
    assert local_budget.max_slice_input_tokens <= int((local_budget.max_output_tokens - 256) / 1.25)
    assert local_budget.needs_splitting is True
    assert local_budget.chosen_num_ctx == 8192

    # Manual num_ctx=4096 caps chosen_num_ctx and reduces max_output_tokens + max_slice_input_tokens proportionally.
    ep_manual = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b", num_ctx=4096, auto_ctx=False)
    manual_budget = compute_cleanup_budgets(ep_manual, long_body, system_info=gpu_8gb)
    assert manual_budget.chosen_num_ctx == 4096
    assert manual_budget.max_output_tokens <= 2048
    assert manual_budget.max_slice_input_tokens <= int((manual_budget.max_output_tokens - 256) / 1.25)

    # Cloud model with 8k output limit never creates slices larger than its output budget can emit.
    ep_ds = ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "sk-test", "deepseek-flash", auto_ctx=True)
    ds_budget = compute_cleanup_budgets(ep_ds, long_body * 3, system_info=gpu_8gb)
    assert ds_budget.max_output_tokens == 8192
    assert ds_budget.max_slice_input_tokens == int((8192 - 256) / 1.25)


def test_split_markdown_pages_packing_oversize_and_error_skip() -> None:
    p1 = "<!-- Page 1 -->\n\n" + ("Alpha line on page one. " * 22)
    p2 = "<!-- Page 2 -->\n\n" + ("Beta line on page two. " * 22)
    p3 = "<!-- Page 3 -->\n\n<!-- OCR ERROR on Page 3: timeout -->"
    p4 = "<!-- Page 4 -->\n\n" + ("\n\n".join(f"Gamma paragraph {k}. " * 14 for k in range(1, 7)))
    doc = "\n\n---\n\n".join([p1, p2, p3, p4])

    slices = split_markdown_pages_for_cleanup(doc, max_slice_tokens=420)
    labels = [s["page_label"] for s in slices]
    # p1 + p2 pack together; error page p3 stands alone with skip_llm=True; oversized p4 splits into parts
    assert labels[0] == "Pages 1-2"
    assert slices[0]["skip_llm"] is False
    assert slices[0]["join_with_hr"] is False
    assert "<!-- Page 1 -->" in slices[0]["text"] and "<!-- Page 2 -->" in slices[0]["text"]

    assert labels[1] == "Page 3"
    assert slices[1]["skip_llm"] is True
    assert slices[1]["join_with_hr"] is True

    p4_parts = [s for s in slices if s["page_label"].startswith("Page 4 (part ")]
    assert len(p4_parts) >= 2
    assert p4_parts[0]["text"].startswith("<!-- Page 4 -->")
    assert p4_parts[0]["join_with_hr"] is True
    assert all(part["join_with_hr"] is False for part in p4_parts[1:])

    # Page-less fallback splits structurally with Part X/Y labels
    pageless = "\n\n".join(f"## Topic {i}\n\n" + ("Content words here. " * 45) for i in range(1, 6))
    pageless_slices = split_markdown_pages_for_cleanup(pageless, max_slice_tokens=420)
    assert len(pageless_slices) >= 2
    assert pageless_slices[0]["page_label"].startswith("Part 1/")


def test_cleanup_progress_output_clamp_and_cancellation(monkeypatch: object) -> None:
    pages = [
        "<!-- Page 1 -->\n\nFirst page brken text.",
        "<!-- Page 2 -->\n\n<!-- OCR ERROR on Page 2: timeout -->",
        "<!-- Page 3 -->\n\nThird page ttitle text.",
    ]
    doc = "\n\n---\n\n".join(pages)
    ep = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b", num_ctx=4096, auto_ctx=False)

    seen_calls: list[tuple[int, str]] = []
    progress_events: list[tuple[int, int, str]] = []

    def fake_chat(endpoint: ProviderEndpoint, system: str, user: str, **kw: object) -> str:
        seen_calls.append((int(kw.get("max_tokens") or 0), user))
        return user.replace("brken", "broken").replace("ttitle", "title")

    monkeypatch.setattr("ocr_pipeline.llm_features.chat_text", fake_chat)  # type: ignore[attr-defined]
    cleaned = cleanup_ocr_markdown(
        doc,
        ep,
        system_info={"vram_gb": 8.0, "ram_gb": 16.0},
        on_progress=lambda i, t, lbl: progress_events.append((i, t, lbl)),
    )
    # Error page skipped LLM cleanly, --- separators preserved, and max_tokens respected ceiling
    assert len(seen_calls) == 2
    assert all(tok <= 2048 for tok, _ in seen_calls)
    assert all("OCR ERROR" not in usr for _, usr in seen_calls)
    assert [e[2] for e in progress_events] == ["Page 1", "Page 2", "Page 3"]
    assert "\n\n---\n\n<!-- Page 2 -->\n\n<!-- OCR ERROR on Page 2: timeout -->\n\n---\n\n" in cleaned
    assert "First page broken text." in cleaned
    assert "Third page title text." in cleaned

    cancel_ev = threading.Event()
    calls_before_cancel = 0

    def cancel_on_second_progress(idx: int, total: int, label: str) -> None:
        if idx >= 2:
            cancel_ev.set()

    def fake_chat_count(endpoint: ProviderEndpoint, system: str, user: str, **kw: object) -> str:
        nonlocal calls_before_cancel
        calls_before_cancel += 1
        return user

    monkeypatch.setattr("ocr_pipeline.llm_features.chat_text", fake_chat_count)  # type: ignore[attr-defined]
    with pytest.raises(JobCancelledError):
        cleanup_ocr_markdown(
            doc,
            ep,
            system_info={"vram_gb": 8.0, "ram_gb": 16.0},
            cancel_event=cancel_ev,
            on_progress=cancel_on_second_progress,
        )
    assert calls_before_cancel == 1
