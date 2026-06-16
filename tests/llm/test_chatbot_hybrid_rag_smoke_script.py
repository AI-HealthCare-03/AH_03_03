import json
from types import SimpleNamespace

import pytest

from scripts.qa import smoke_chatbot_hybrid_rag_runtime as smoke


def test_smoke_requires_explicit_openai_confirm(monkeypatch) -> None:
    monkeypatch.setenv("CHATBOT_SMOKE_TOKEN", "secret-token")
    args = SimpleNamespace(confirm_openai_call=False, token_env="CHATBOT_SMOKE_TOKEN", limit=1)

    with pytest.raises(smoke.SmokeConfigurationError):
        smoke.validate_args(args)


def test_smoke_requires_token_without_printing_value(monkeypatch) -> None:
    monkeypatch.delenv("CHATBOT_SMOKE_TOKEN", raising=False)
    args = SimpleNamespace(confirm_openai_call=True, token_env="CHATBOT_SMOKE_TOKEN", limit=1)

    with pytest.raises(smoke.SmokeConfigurationError) as exc_info:
        smoke.validate_args(args)

    assert "secret-token" not in str(exc_info.value)
    assert "CHATBOT_SMOKE_TOKEN" in str(exc_info.value)


def test_call_chatbot_parses_trace_header_and_checks_public_answer(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200
        headers = {
            "X-Chatbot-Rag-Trace": json.dumps(
                {
                    "rag_strategy": "hybrid_parallel",
                    "keyword_returned_count": 2,
                    "vector_returned_count": 2,
                    "merged_count": 3,
                    "final_count": 2,
                    "fallback_used": False,
                    "fallback_reason": None,
                }
            )
        }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "answer": "혈당 관리는 생활습관을 참고해 보세요. 이 정보는 진단이 아니며, 정확한 진단과 치료는 의료진 상담이 필요합니다.",
                    "source": "rag_llm",
                    "recommended_actions": [],
                    "safety_notice": "notice",
                },
                ensure_ascii=False,
            ).encode()

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.headers["Authorization"]
        captured["smoke_trace"] = request.headers["X-chatbot-smoke-trace"]
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(smoke, "urlopen", fake_urlopen)

    record = smoke.call_chatbot(
        base_url="http://localhost:8080",
        token="secret-token",
        question="혈당 관리",
        case_id="C01",
        timeout=3,
    )

    assert captured["authorization"] == "Bearer secret-token"
    assert captured["smoke_trace"] == "true"
    assert record["source"] == "rag_llm"
    assert record["rag_trace"]["rag_strategy"] == "hybrid_parallel"
    assert record["rag_trace"]["keyword_returned_count"] == 2
    assert record["rag_trace"]["vector_returned_count"] == 2
    assert record["checks"]["no_json_code_block"] is True
    assert record["checks"]["no_internal_terms"] is True
    assert record["passed"] is True


def test_answer_checks_detect_json_and_internal_terms() -> None:
    checks = smoke.check_answer(
        '근거 수준이 제한적이므로 참고용입니다. ```json {"answer": "x", "source": "rag_llm", '
        '"caution_message": "notice"} ``` chunk_key score embedding'
    )

    assert checks["no_json_code_block"] is False
    assert checks["no_internal_terms"] is False


def test_markdown_output_does_not_include_secret_values() -> None:
    result = {
        "generated_at": "2026-06-16T00:00:00+00:00",
        "question_count": 1,
        "db_write_performed": False,
        "summary": {"passed_count": 1, "failed_count": 0},
        "records": [
            {
                "case_id": "C01",
                "question": "혈당 관리",
                "status_code": 200,
                "source": "rag_llm",
                "rag_trace": {"rag_strategy": "hybrid_parallel"},
                "checks": {"no_json_code_block": True, "no_internal_terms": True},
                "answer_preview": "자연어 답변",
            }
        ],
    }

    rendered = smoke.render_markdown(result)

    assert "secret-token" not in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert "hybrid_parallel" in rendered
