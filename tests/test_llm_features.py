"""Tests for optional LLM-powered artifact tagging, renaming, organization, summary & cleanup."""

import json
from pathlib import Path

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
