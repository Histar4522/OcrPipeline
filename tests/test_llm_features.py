"""Tests for optional LLM-powered artifact tagging, renaming, organization, summary & cleanup."""

import json
from pathlib import Path

import httpx

from ocr_pipeline.llm_context import (
    calculate_vram_plan,
    choose_adequate_context,
    inspect_ollama_model_specs,
    split_markdown_chunks,
)
from ocr_pipeline.llm_features import (
    _parse_json_object,
    analyze_artifact_all,
    cleanup_ocr_markdown,
    plan_folder_organization,
    suggest_filename,
    suggest_folder,
    suggest_tags,
    summarize_document,
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
        lambda endpoint, sys, usr, **kw: json.dumps({"folder": "BrandNewFolder", "reason": "Matches waves"}),
    )  # type: ignore[attr-defined]
    blocked_new = suggest_folder("# Waves", ["Physics/Optics"], ["waves"], ep, allow_new_folders=False)
    assert blocked_new["folder"] == ""
    allowed_new = suggest_folder("# Waves", ["Physics/Optics"], ["waves"], ep, allow_new_folders=True)
    assert allowed_new["folder"] == "BrandNewFolder"
    assert allowed_new["is_new"] is True

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


def test_analyze_and_batch_plan(tmp_path: Path, monkeypatch: object) -> None:
    archive = tmp_path / "archive"
    (archive / "Chemistry").mkdir(parents=True)
    (archive / "raw_scan.md").write_text("# Acid Base Titration\npH curves and buffers", encoding="utf-8")

    ep = ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b")
    monkeypatch.setattr(
        "ocr_pipeline.llm_features.chat_text",
        lambda endpoint, sys, usr, **kw: json.dumps(
            {
                "tags": ["chemistry", "titration"],
                "filename": "Acid_Base_Titration.md",
                "folder": "Chemistry",
                "summary": "Covers acid-base titration curves.",
                "reason": "Chemistry lab notes.",
            }
        ),
    )  # type: ignore[attr-defined]

    single = analyze_artifact_all(
        "# Acid Base Titration",
        current_name="raw_scan.md",
        known_tags=[],
        existing_folders=["Chemistry"],
        endpoint=ep,
    )
    assert single["filename"] == "Acid_Base_Titration.md"
    assert single["folder"] == "Chemistry"
    assert single["tags"] == ["chemistry", "titration"]

    plan = plan_folder_organization(archive, "", ep, allow_new_folders=False)
    assert len(plan) == 1
    assert plan[0]["suggested_folder"] == "Chemistry"
    assert plan[0]["suggested_name"] == "Acid_Base_Titration.md"


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
