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
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("openai unavailable")
        return _Response()


class _Client:
    def __init__(self, *, fail: bool = False) -> None:
        self.responses = _Responses(fail=fail)


class _FakeObservation:
    def __init__(self) -> None:
        self.output = None

    def update(self, **kwargs):
        self.output = kwargs.get("output")


class _FakeObservationContext:
    def __init__(self) -> None:
        self.observation = _FakeObservation()

    def __enter__(self):
        return self.observation

    def __exit__(self, *args):
        return False


class _FakeLangfuse:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def start_as_current_observation(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeObservationContext()


class _TemperatureUnsupportedResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "temperature" in kwargs:
            raise TypeError("Responses.create() got an unexpected keyword argument 'temperature'")
        return _Response()


class _TemperatureUnsupportedClient:
    def __init__(self) -> None:
        self.responses = _TemperatureUnsupportedResponses()


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


def test_openai_runtime_passes_configured_temperature(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr(llm_client.config, "OPENAI_TEMPERATURE", 0.0)

    response = llm_client._create_openai_response_with_observability(
        client=client,
        model="gpt-test",
        prompt="short prompt",
        metadata={"source": "temperature_smoke"},
    )

    assert response.output_text == "ok"
    assert client.responses.calls[-1]["temperature"] == 0.0
    assert "top_p" not in client.responses.calls[-1]


def test_openai_runtime_retries_without_temperature_when_unsupported(monkeypatch) -> None:
    client = _TemperatureUnsupportedClient()
    monkeypatch.setattr(llm_client.config, "OPENAI_TEMPERATURE", 0.0)

    response = llm_client._create_openai_response_with_observability(
        client=client,
        model="gpt-test",
        prompt="short prompt",
        metadata={"source": "temperature_fallback_smoke"},
    )

    assert response.output_text == "ok"
    assert len(client.responses.calls) == 2
    assert client.responses.calls[0]["temperature"] == 0.0
    assert "temperature" not in client.responses.calls[1]
    assert "top_p" not in client.responses.calls[0]
    assert "top_p" not in client.responses.calls[1]


def test_langfuse_generation_input_uses_redacted_prompt_but_openai_receives_original(
    monkeypatch,
) -> None:
    client = _Client()
    langfuse = _FakeLangfuse()
    monkeypatch.setattr(llm_client, "build_langfuse_client", lambda: langfuse)

    prompt = (
        "이름: 홍길동, 이메일 hong@example.com, 전화 010-1234-5678, 주민번호 900101-1234567. "
        "혈압 120/80 mmHg, 공복혈당 105 mg/dL, HbA1c 6.5%, LDL 130, "
        "복용약: 메트포르민 500mg 정보를 참고해 설명해줘."
    )

    response = llm_client.create_openai_response(
        client=client,
        model="gpt-test",
        prompt=prompt,
        metadata={"source": "redaction_smoke"},
    )

    assert response.output_text == "ok"
    assert client.responses.calls[-1]["input"] == prompt
    langfuse_input = langfuse.calls[-1]["input"]
    assert "홍길동" not in langfuse_input
    assert "hong@example.com" not in langfuse_input
    assert "010-1234-5678" not in langfuse_input
    assert "900101-1234567" not in langfuse_input
    assert "120/80" not in langfuse_input
    assert "105" not in langfuse_input
    assert "6.5" not in langfuse_input
    assert "130" not in langfuse_input
    assert "메트포르민" not in langfuse_input
    assert "[email]" in langfuse_input
    assert "[phone]" in langfuse_input
    assert "[rrn]" in langfuse_input
    assert "[health_value]" in langfuse_input
    assert "[medication]" in langfuse_input
