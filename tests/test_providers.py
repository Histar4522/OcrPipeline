"""Tests for multi-provider Vision OCR & text LLM adapters."""

import json
from pathlib import Path

import httpx
import pytest

from ocr_pipeline.config import OcrConfig
from ocr_pipeline.ollama_client import ProviderError
from ocr_pipeline.providers import (
    PROVIDER_REGISTRY,
    ProviderEndpoint,
    chat_text,
    check_endpoint_ready,
    test_provider_connection as probe_provider_connection,
    vision_ocr_image_b64,
    vision_ocr_with_config,
)


def test_provider_registry_covers_expected_providers() -> None:
    expected = {"ollama", "llama_cpp", "openai", "anthropic", "google", "deepseek", "xai", "qwen", "custom"}
    assert expected.issubset(PROVIDER_REGISTRY.keys())
    assert PROVIDER_REGISTRY["deepseek"].supports_vision is False
    assert PROVIDER_REGISTRY["deepseek"].supports_llm is True
    assert PROVIDER_REGISTRY["ollama"].is_local is True
    assert PROVIDER_REGISTRY["llama_cpp"].is_local is True


def test_vision_ocr_protocols_via_mock_transport() -> None:
    seen: list[tuple[str, dict, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append((str(request.url), dict(request.headers), body))
        url_str = str(request.url)
        if "/api/chat" in url_str:
            return httpx.Response(200, json={"message": {"content": "ollama ok"}})
        if "/chat/completions" in url_str:
            return httpx.Response(200, json={"choices": [{"message": {"content": "openai ok"}}]})
        if "/messages" in url_str:
            return httpx.Response(200, json={"content": [{"type": "text", "text": "anthropic ok"}]})
        if ":generateContent" in url_str:
            return httpx.Response(
                200, json={"candidates": [{"content": {"parts": [{"text": "gemini ok"}]}}]}
            )
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        out_ollama = vision_ocr_with_config(
            "QUJD",
            OcrConfig(provider="ollama", model="glm-ocr", base_url="http://localhost:11434"),
            client=client,
        )
        assert out_ollama == "ollama ok"

        out_openai = vision_ocr_image_b64(
            "QUJD",
            ProviderEndpoint("openai", "https://api.openai.com/v1", "sk-test", "gpt-4.1-mini"),
            "sys",
            "usr",
            client=client,
        )
        assert out_openai == "openai ok"

        out_anth = vision_ocr_image_b64(
            "QUJD",
            ProviderEndpoint("anthropic", "https://api.anthropic.com/v1", "ak-test", "claude-sonnet-4-5"),
            "sys",
            "usr",
            client=client,
        )
        assert out_anth == "anthropic ok"

        out_gem = vision_ocr_image_b64(
            "QUJD",
            ProviderEndpoint("google", "https://generativelanguage.googleapis.com/v1beta", "gk-test", "gemini-2.5-flash"),
            "sys",
            "usr",
            client=client,
        )
        assert out_gem == "gemini ok"

    assert seen[1][1]["authorization"] == "Bearer sk-test"
    assert seen[1][2]["messages"][1]["content"][1]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert seen[2][1]["x-api-key"] == "ak-test"
    assert seen[2][2]["messages"][0]["content"][0]["source"]["data"] == "QUJD"
    assert "key=gk-test" in seen[3][0]
    assert seen[3][2]["contents"][0]["parts"][1]["inline_data"]["data"] == "QUJD"


def test_chat_text_json_mode_across_protocols() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        url_str = str(request.url)
        if "/api/chat" in url_str:
            assert body["format"] == "json"
            return httpx.Response(200, json={"message": {"content": '{"ok": "ollama"}'}})
        if "/chat/completions" in url_str:
            assert body["response_format"] == {"type": "json_object"}
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": "compat"}'}}]})
        if "/messages" in url_str:
            return httpx.Response(200, json={"content": [{"type": "text", "text": '{"ok": "anthropic"}'}]})
        if ":generateContent" in url_str:
            assert body["generationConfig"]["responseMimeType"] == "application/json"
            return httpx.Response(
                200, json={"candidates": [{"content": {"parts": [{"text": '{"ok": "google"}'}]}}]}
            )
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert chat_text(ProviderEndpoint("ollama", "http://localhost:11434", "", "qwen3:1.7b"), "s", "u", json_mode=True, client=client) == '{"ok": "ollama"}'
        assert chat_text(ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "dk", "deepseek-chat"), "s", "u", json_mode=True, client=client) == '{"ok": "compat"}'
        assert chat_text(ProviderEndpoint("anthropic", "https://api.anthropic.com/v1", "ak", "claude-haiku-4-5"), "s", "u", json_mode=True, client=client) == '{"ok": "anthropic"}'
        assert chat_text(ProviderEndpoint("google", "https://generativelanguage.googleapis.com/v1beta", "gk", "gemini-2.5-flash-lite"), "s", "u", json_mode=True, client=client) == '{"ok": "google"}'


def test_connection_checks_and_validation(monkeypatch: object) -> None:
    with pytest.raises(ProviderError, match="does not support Vision"):
        check_endpoint_ready(
            ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "dk", "deepseek-chat"),
            require_vision=True,
        )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # type: ignore[attr-defined]
    res = probe_provider_connection("openai", "", "")
    assert res["online"] is False
    assert "API key" in res["message"]

    monkeypatch.setenv("OPENAI_API_KEY", "env-key-123")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "ocr_pipeline.providers.httpx.get",
        lambda *a, **k: httpx.Response(
            200, json={"data": [{"id": "gpt-4.1-mini"}, {"id": "gpt-4o"}]}, request=httpx.Request("GET", "https://api.openai.com/v1/models")
        ),
    )  # type: ignore[attr-defined]
    ok = probe_provider_connection("openai", "", "")
    assert ok["online"] is True
    assert ok["models"] == ["gpt-4.1-mini", "gpt-4o"]
