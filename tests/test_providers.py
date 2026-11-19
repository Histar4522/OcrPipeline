"""Tests for multi-provider Vision OCR & text LLM adapters."""

import base64
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
    classify_http_error,
    clean_ocr_output,
    probe_endpoint_status,
    test_provider_connection as probe_provider_connection,
    vision_ocr_image_b64,
    vision_ocr_with_config,
)


def test_provider_registry_covers_expected_providers() -> None:
    expected = {
        "ollama_local",
        "ollama",
        "llama_cpp",
        "openai",
        "anthropic",
        "google",
        "deepseek",
        "xai",
        "qwen",
        "custom",
    }
    assert expected.issubset(PROVIDER_REGISTRY.keys())
    assert PROVIDER_REGISTRY["ollama_local"].singleton is True
    assert PROVIDER_REGISTRY["deepseek"].supports_vision is True
    assert "deepseek-ocr" in PROVIDER_REGISTRY["deepseek"].suggested_vision_models
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
    assert "temperature" not in seen[2][2]
    assert seen[2][2]["messages"][0]["content"][0]["source"]["data"] == "QUJD"
    assert seen[3][1]["x-goog-api-key"] == "gk-test"
    assert seen[3][2]["contents"][0]["parts"][1]["inlineData"]["data"] == "QUJD"


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
        assert chat_text(ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "dk", "deepseek-flash"), "s", "u", json_mode=True, client=client) == '{"ok": "compat"}'
        assert chat_text(ProviderEndpoint("anthropic", "https://api.anthropic.com/v1", "ak", "claude-haiku-4-5"), "s", "u", json_mode=True, client=client) == '{"ok": "anthropic"}'
        assert chat_text(ProviderEndpoint("google", "https://generativelanguage.googleapis.com/v1beta", "gk", "gemini-2.5-flash-lite"), "s", "u", json_mode=True, client=client) == '{"ok": "google"}'


def test_connection_checks_and_validation(monkeypatch: object) -> None:
    # DeepSeek OCR is accepted as a Vision OCR endpoint when API key & model are set
    check_endpoint_ready(
        ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "dk", "deepseek-ocr"),
        require_vision=True,
    )
    assert classify_http_error(429, "rate limit exceeded") == "out_of_quota"
    assert classify_http_error(402, "insufficient quota") == "out_of_quota"
    assert classify_http_error(401, "unauthorized") == "invalid_key"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # type: ignore[attr-defined]
    res = probe_provider_connection("openai", "", "")
    assert res["online"] is False
    assert res["status_code"] == "no_key"
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
    assert ok["status_code"] == "ok"
    assert ok["status_text"] == ""
    assert ok["models"] == ["gpt-4.1-mini", "gpt-4o"]

    # Quota exhaustion surfaced cleanly by probe_endpoint_status
    monkeypatch.setattr(
        "ocr_pipeline.providers.httpx.get",
        lambda *a, **k: httpx.Response(
            429, json={"error": {"message": "You exceeded your current quota"}}, request=httpx.Request("GET", "https://api.deepseek.com/v1/models")
        ),
    )  # type: ignore[attr-defined]
    quota_res = probe_endpoint_status(ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "dk", "deepseek-ocr"))
    assert quota_res["status"] == "out_of_quota"
    assert quota_res["status_text"] == "Out of Quota"

    disabled_res = probe_endpoint_status(ProviderEndpoint("ollama_local", "http://localhost:11434", "", "glm-ocr"), enabled=False)
    assert disabled_res["status"] == "disabled"
    assert disabled_res["status_text"] == "Disabled"


def test_deepseek_ocr_tool_and_flash_vision() -> None:
    calls: list[tuple[str, dict, bytes]] = []
    png_raw = base64.b64encode(b"\x89PNG\r\n\x1a\nTEST").decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((str(request.url), dict(request.headers), request.content))
        url_str = str(request.url)
        if url_str.endswith("/v1/tools/ocr"):
            return httpx.Response(
                200,
                json={
                    "id": "ocr-1",
                    "object": "ocr.result",
                    "page_count": 2,
                    "pages": [
                        {"index": 1, "text": "<｜ref｜>text<｜/ref｜><｜box｜>[[10,20,30,40]]<｜/box｜> Second block", "finish_reason": "stop"},
                        {"index": 0, "text": "<｜ref｜>sub_title<｜/ref｜><｜box｜>[[1,2,3,4]]<｜/box｜> First block", "finish_reason": "stop"},
                    ],
                    "usage": {"input_tokens": 120, "output_tokens": 45},
                },
            )
        if url_str.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "flash ok"}}]})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        # Both https://api.deepseek.com and https://api.deepseek.com/v1 resolve to /v1/tools/ocr
        metrics: dict[str, object] = {}
        out1 = vision_ocr_image_b64(
            png_raw,
            ProviderEndpoint("deepseek", "https://api.deepseek.com", "dk-1", "deepseek-ocr"),
            "sys",
            "usr",
            client=client,
            metrics_out=metrics,
        )
        assert out1 == "## First block\n\nSecond block"
        assert metrics["prompt_eval_count"] == 120
        assert metrics["eval_count"] == 45

        out2 = vision_ocr_image_b64(
            png_raw,
            ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "dk-2", "deepseek-ocr-text"),
            "sys",
            "usr",
            client=client,
        )
        assert out2 == "## First block\n\nSecond block"

        out_flash = vision_ocr_image_b64(
            png_raw,
            ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "dk-3", "deepseek-flash"),
            "sys",
            "usr",
            client=client,
        )
        assert out_flash == "flash ok"

    assert calls[0][0] == "https://api.deepseek.com/v1/tools/ocr"
    assert calls[0][1]["authorization"] == "Bearer dk-1"
    assert "multipart/form-data" in calls[0][1]["content-type"]
    assert b"rich_layout_recognition" in calls[0][2]
    assert b"\x89PNG\r\n\x1a\nTEST" in calls[0][2]

    assert calls[1][0] == "https://api.deepseek.com/v1/tools/ocr"
    assert b"text_recognition" in calls[1][2]

    assert calls[2][0] == "https://api.deepseek.com/v1/chat/completions"
    flash_body = json.loads(calls[2][2].decode("utf-8"))
    assert flash_body["thinking"] == {"type": "disabled"}
    assert flash_body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    with pytest.raises(ProviderError, match="does not support Vision"):
        vision_ocr_image_b64(
            png_raw,
            ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "dk", "deepseek-v4-pro"),
            "sys",
            "usr",
        )
    with pytest.raises(ProviderError, match="does not support Vision"):
        check_endpoint_ready(
            ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "dk", "deepseek-v4-pro"),
            require_vision=True,
        )
    with pytest.raises(ProviderError, match="dedicated OCR tool"):
        chat_text(
            ProviderEndpoint("deepseek", "https://api.deepseek.com/v1", "dk", "deepseek-ocr"),
            "sys",
            "usr",
        )


def test_openai_qwen_google_anthropic_quirks(monkeypatch: object) -> None:
    seen: list[tuple[str, dict, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append((str(request.url), dict(request.headers), body))
        url_str = str(request.url)
        if "/chat/completions" in url_str:
            return httpx.Response(200, json={"choices": [{"message": {"content": "compat ok"}}]})
        if ":generateContent" in url_str:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "hidden thought", "thought": True},
                                    {"text": "visible answer"},
                                ]
                            }
                        }
                    ]
                },
            )
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert (
            chat_text(
                ProviderEndpoint("openai", "https://api.openai.com/v1", "sk", "gpt-5-mini"),
                "sys",
                "usr",
                max_tokens=512,
                client=client,
            )
            == "compat ok"
        )
        assert (
            chat_text(
                ProviderEndpoint("openai", "https://api.openai.com/v1", "sk", "gpt-4.1-mini"),
                "sys",
                "usr",
                max_tokens=512,
                client=client,
            )
            == "compat ok"
        )
        assert (
            vision_ocr_image_b64(
                "QUJD",
                ProviderEndpoint("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qk", "qwen-vl-ocr"),
                "System rules",
                "User prompt",
                client=client,
            )
            == "compat ok"
        )
        assert (
            chat_text(
                ProviderEndpoint("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "qk", "qwen3.8-flash"),
                "sys",
                "usr",
                client=client,
            )
            == "compat ok"
        )
        gem_out = vision_ocr_image_b64(
            "QUJD",
            ProviderEndpoint("google", "https://generativelanguage.googleapis.com/v1beta", "gk", "gemini-3.8-flash"),
            "sys",
            "usr",
            client=client,
        )
        assert gem_out == "visible answer"

    # GPT-5 omits temperature and uses max_completion_tokens; GPT-4.1 keeps temperature + max_completion_tokens
    assert "temperature" not in seen[0][2]
    assert "max_tokens" not in seen[0][2]
    assert seen[0][2]["max_completion_tokens"] == 512
    assert seen[1][2]["temperature"] == 0.1
    assert seen[1][2]["max_completion_tokens"] == 512

    # Qwen OCR omits role=system message & folds prompt into user content, and omits enable_thinking
    assert all(m["role"] != "system" for m in seen[2][2]["messages"])
    assert "System rules" in seen[2][2]["messages"][0]["content"][0]["text"]
    assert "enable_thinking" not in seen[2][2]
    # Qwen3 text LLM sets enable_thinking=False
    assert seen[3][2]["enable_thinking"] is False

    # Model discovery filtering for Google, OpenAI, and xAI
    monkeypatch.setattr(
        "ocr_pipeline.providers.httpx.get",
        lambda url, **k: httpx.Response(
            200,
            json={
                "models": [
                    {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
                    {"name": "models/imagen-3.0-generate-002", "supportedGenerationMethods": ["generateContent"]},
                ]
            },
            request=httpx.Request("GET", str(url)),
        ),
    )  # type: ignore[attr-defined]
    gem_list = probe_provider_connection("google", "", "gk-test")
    assert gem_list["models"] == ["gemini-3.8-flash"]

    monkeypatch.setattr(
        "ocr_pipeline.providers.httpx.get",
        lambda url, **k: httpx.Response(
            200,
            json={
                "data": [
                    {"id": "gpt-5-mini"},
                    {"id": "whisper-1"},
                    {"id": "text-embedding-3-small"},
                    {"id": "gpt-image-1"},
                ]
            },
            request=httpx.Request("GET", str(url)),
        ),
    )  # type: ignore[attr-defined]
    oa_list = probe_provider_connection("openai", "", "sk-test")
    assert oa_list["models"] == ["gpt-5-mini"]

    monkeypatch.setattr(
        "ocr_pipeline.providers.httpx.get",
        lambda url, **k: httpx.Response(
            200,
            json={"data": [{"id": "grok-4.7"}, {"id": "grok-imagine-image-2.0"}]},
            request=httpx.Request("GET", str(url)),
        ),
    )  # type: ignore[attr-defined]
    xai_list = probe_provider_connection("xai", "", "xk-test")
    assert xai_list["models"] == ["grok-4.7"]


def test_clean_ocr_output_strips_provider_reference_and_box_marks() -> None:
    # DeepSeek /v1/tools/ocr fullwidth <｜ref｜>/<｜box｜> output matching user sample
    ds_raw = (
        "<｜ref｜>sub_title<｜/ref｜><｜box｜>[[291,88,821,138]]<｜/box｜> Balanced Chemical Equations \n"
        "<｜ref｜>equation<｜/ref｜><｜box｜>[[114,138,856,287]]<｜/box｜> $$\n"
        "6CO_2 + 6H_2O \\rightarrow C_6H_{12}O_6 + 6O_2\n"
        "$$\n"
        "<｜ref｜>text<｜/ref｜><｜box｜>[[114,287,221,309]]<｜/box｜> Bicarbonate "
    )
    ds_clean = clean_ocr_output(ds_raw)
    assert "<｜ref｜>" not in ds_clean
    assert "<｜box｜>" not in ds_clean
    assert ds_clean.startswith("## Balanced Chemical Equations\n$$")
    assert "$$" in ds_clean and ds_clean.endswith("Bicarbonate")

    # DeepSeek-OCR already-headed sub_title should not double `##`
    assert clean_ocr_output("<|ref|>sub_title<|/ref|><|det|>[[10,20,30,40]]<|/det|> ## Existing Heading") == "## Existing Heading"
    assert clean_ocr_output("<|ref|>title<|/ref|><|det|>[[10,20,30,40]]<|/det|> Document Title") == "# Document Title"

    # DeepSeek-OCR span-grounding mode keeps inner wrapped text and removes box coords
    span_raw = "<|ref|>The perils of vibe coding<|/ref|><|det|>[[352,30,624,111]]<|/det|> details here"
    assert clean_ocr_output(span_raw) == "The perils of vibe coding details here"

    # Qwen-VL tokens (object_ref_start / box_start / XML ref+box / im_end)
    qwen_raw = (
        "<|object_ref_start|>Invoice<|object_ref_end|><|box_start|>(10,20),(300,80)<|box_end|>\n"
        "<ref>Total</ref><box>[[1, 2, 3, 4]]</box><|im_end|>"
    )
    assert clean_ocr_output(qwen_raw) == "Invoice\nTotal"

    # GLM-OCR box, loc tokens, <think>, and outer ```markdown fence unwrapping
    fenced_raw = (
        "<think>internal reasoning</think>\n"
        "```markdown\n"
        "<|begin_of_box|>[[10,20,30,40]]<|end_of_box|><loc0123># Clean Title\n"
        "Line a < x and y > z\n"
        "```"
    )
    assert clean_ocr_output(fenced_raw) == "# Clean Title\nLine a < x and y > z"

    # Preserve physics/math notation (<|psi|>) and legitimate code blocks inside a page
    safe_raw = "State $|\\psi\\rangle$ or `<|psi|>` stays.\n\n```python\nprint('hi')\n```"
    assert clean_ocr_output(safe_raw) == safe_raw
