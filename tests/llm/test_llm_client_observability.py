from __future__ import annotations

import pytest

from ai_runtime.llm import llm_client


class _Usage:
    input_tokens = 11
    output_tokens = 7
    total_tokens = 18


class _Response:
    output_text = "ok"
    usage = _Usage()


class _Responses:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def create(self, **kwargs):
        if self.fail:
            raise RuntimeError("openai unavailable")
        return _Response()


class _Client:
    def __init__(self, *, fail: bool = False) -> None:
        self.responses = _Responses(fail=fail)


def test_openai_runtime_observability_logs_safe_usage_metadata(monkeypatch) -> None:
    logged: list[dict] = []

    def fake_log_info(*args, **kwargs) -> None:
        logged.append(kwargs.get("extra") or {})

    monkeypatch.setattr(llm_client.config, "ENV", "local")
    monkeypatch.setattr(llm_client.logger, "info", fake_log_info)

    response = llm_client._create_openai_response_with_observability(
        client=_Client(),
        model="gpt-test",
        prompt="secret prompt body must not be logged",
        metadata={"source": "rag_llm", "chatbot_type": "main_health_chatbot"},
    )

    assert response.output_text == "ok"
    assert logged
    payload = logged[-1]["llm_runtime"]
    assert payload == {
        "llm_provider": "openai",
        "llm_model": "gpt-test",
        "llm_call_path": "main_health_chatbot.rag_llm",
        "latency_ms": payload["latency_ms"],
        "success": True,
        "error_type": None,
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    assert "secret prompt body" not in str(payload)


def test_openai_runtime_observability_logs_error_type_without_prompt(monkeypatch) -> None:
    logged: list[dict] = []

    def fake_log_info(*args, **kwargs) -> None:
        logged.append(kwargs.get("extra") or {})

    monkeypatch.setattr(llm_client.config, "ENV", "local")
    monkeypatch.setattr(llm_client.logger, "info", fake_log_info)

    with pytest.raises(RuntimeError):
        llm_client._create_openai_response_with_observability(
            client=_Client(fail=True),
            model="gpt-test",
            prompt="secret prompt body must not be logged",
            metadata={"source": "analysis_explanation_rewrite"},
        )

    assert logged
    payload = logged[-1]["llm_runtime"]
    assert payload["success"] is False
    assert payload["error_type"] == "RuntimeError"
    assert payload["llm_call_path"] == "analysis_explanation_rewrite"
    assert payload["prompt_tokens"] is None
    assert "secret prompt body" not in str(payload)
