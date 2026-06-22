from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.dtos.chatbot import ChatbotAskRequest, ChatbotContextType
from app.services import chatbot as chatbot_service

STATIC_SAFETY_NOTICE = "본 서비스는 의료 진단이나 처방을 대신하지 않으며, 건강관리 참고 정보로 활용해 주세요."


def _fail_if_graph_called(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("Static intent responses must not call LangGraph or LLM flow.")


@pytest.mark.asyncio
async def test_static_greeting_does_not_call_llm_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatbot_service, "run_chatbot_graph_async", _fail_if_graph_called)

    response, runtime_trace = await chatbot_service.ask_chatbot_with_runtime_trace(
        ChatbotAskRequest(message="안녕하세요", context_type=ChatbotContextType.MAIN),
        user_id=7,
    )

    assert response.source == "static_greeting"
    assert response.context_type == ChatbotContextType.MAIN
    assert STATIC_SAFETY_NOTICE in response.answer
    assert response.recommended_actions
    assert runtime_trace == {"static_intent": "greeting", "source": "static_greeting"}


@pytest.mark.asyncio
async def test_static_service_intro_returns_fixed_response_without_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatbot_service, "run_chatbot_graph_async", _fail_if_graph_called)

    response = await chatbot_service.ask_chatbot(
        ChatbotAskRequest(message="Health Ladder가 뭐야?", context_type=ChatbotContextType.GENERAL),
        user_id=7,
    )

    assert response.source == "static_service_intro"
    assert "Health Ladder" in response.answer
    assert "건강관리 서비스" in response.answer
    assert STATIC_SAFETY_NOTICE in response.answer


@pytest.mark.asyncio
async def test_static_help_returns_fixed_response_without_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatbot_service, "run_chatbot_graph_async", _fail_if_graph_called)

    response = await chatbot_service.ask_chatbot(
        ChatbotAskRequest(message="사용법 알려줘", context_type=ChatbotContextType.GENERAL),
        user_id=7,
    )

    assert response.source == "static_help"
    assert "건강정보 입력" in response.answer
    assert "식단 기록" in response.answer
    assert STATIC_SAFETY_NOTICE in response.answer


@pytest.mark.asyncio
async def test_general_health_question_falls_through_to_existing_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_graph(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            answer="혈압 관리는 저염식과 규칙적인 활동을 함께 점검해보세요.",
            source="rule_engine",
            recommended_actions=["건강정보 기록하기"],
            caution_message="본 서비스는 진단/처방이 아니며, 치료 변경은 반드시 의료진과 상담해야 합니다.",
            trace_metadata={"retrieval": {"fallback_reason": "rag_disabled"}},
        )

    monkeypatch.setattr(chatbot_service, "run_chatbot_graph_async", fake_graph)
    monkeypatch.setattr(chatbot_service.config, "CHATBOT_USE_REAL_LLM", False)
    monkeypatch.setattr(chatbot_service, "load_chatbot_domain_context", _fake_no_context)

    response, runtime_trace = await chatbot_service.ask_chatbot_with_runtime_trace(
        ChatbotAskRequest(message="혈압 관리는 어떻게 하나요?", context_type=ChatbotContextType.MAIN),
        user_id=7,
    )

    assert len(calls) == 1
    assert calls[0]["user_message"] == "혈압 관리는 어떻게 하나요?"
    assert calls[0]["context_type"] == ChatbotContextType.MAIN.value
    assert response.source == "rule_engine"
    assert response.recommended_actions == ["건강정보 기록하기"]
    assert runtime_trace["fallback_reason"] == "rag_disabled"


async def _fake_no_context(*args: Any, **kwargs: Any) -> None:
    return None
