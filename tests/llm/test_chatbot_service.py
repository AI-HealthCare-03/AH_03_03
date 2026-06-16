from __future__ import annotations

import pytest

from app.dtos.chatbot import ChatbotAskRequest, ChatbotContextType
from app.services.chatbot import ask_chatbot


@pytest.mark.asyncio
async def test_chatbot_uses_local_llm_rule_router_without_dummy_source(monkeypatch) -> None:
    monkeypatch.setattr("app.services.chatbot.config.CHATBOT_USE_REAL_LLM", False)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = await ask_chatbot(
        ChatbotAskRequest(message="혈당 관리는 어떻게 하나요?", context_type=ChatbotContextType.MAIN)
    )

    assert response.source == "rule_engine"
    assert "진단이 아니" in response.answer
    assert "의료진 상담" in response.safety_notice


@pytest.mark.asyncio
async def test_chatbot_response_contract_keeps_frontend_fields(monkeypatch) -> None:
    monkeypatch.setattr("app.services.chatbot.config.CHATBOT_USE_REAL_LLM", False)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)

    response = await ask_chatbot(
        ChatbotAskRequest(message="만성질환 예방은 어떻게 시작하면 좋나요?", context_type=ChatbotContextType.GENERAL)
    )
    payload = response.model_dump()

    assert set(payload) == {"answer", "source", "context_type", "recommended_actions", "safety_notice"}
    assert response.source == "rule_engine_unmatched"
    assert response.context_type == ChatbotContextType.GENERAL
    assert isinstance(response.recommended_actions, list)
    assert response.safety_notice


@pytest.mark.asyncio
async def test_chatbot_can_return_rag_llm_source_without_api_contract_change(monkeypatch) -> None:
    monkeypatch.setattr("app.services.chatbot.config.CHATBOT_USE_REAL_LLM", False)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", True)

    response = await ask_chatbot(
        ChatbotAskRequest(message="공복혈당 관리 방법을 알려줘", context_type=ChatbotContextType.MAIN)
    )

    assert response.source == "rag_llm"
    assert response.context_type == ChatbotContextType.MAIN
    assert response.recommended_actions
    assert "진단이 아니" in response.answer


@pytest.mark.asyncio
async def test_chatbot_real_llm_flag_without_openai_key_falls_back_to_rule_engine(monkeypatch) -> None:
    monkeypatch.setattr("app.services.chatbot.config.CHATBOT_USE_REAL_LLM", True)
    monkeypatch.setattr("app.services.chatbot.config.OPENAI_API_KEY", None)
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = await ask_chatbot(
        ChatbotAskRequest(message="혈압 관리는 어떻게 하나요?", context_type=ChatbotContextType.MAIN)
    )

    assert response.source == "rule_engine"
    assert "진단이 아니" in response.answer


@pytest.mark.asyncio
async def test_chatbot_real_llm_flag_uses_openai_rewrite_when_configured(monkeypatch) -> None:
    monkeypatch.setattr("app.services.chatbot.config.CHATBOT_USE_REAL_LLM", True)
    monkeypatch.setattr("app.services.chatbot.config.OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr("ai_runtime.llm.graph.nodes.config.RAG_ENABLED", False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        "ai_runtime.llm.llm_generator.call_llm_json",
        lambda *args, **kwargs: '{"answer": "혈압 관리는 저염식과 규칙적인 활동을 함께 점검해보세요."}',
    )

    response = await ask_chatbot(
        ChatbotAskRequest(message="혈압 관리는 어떻게 하나요?", context_type=ChatbotContextType.MAIN)
    )

    assert response.source == "openai_rewrite"
    assert "진단이 아니" in response.answer


@pytest.mark.asyncio
async def test_chatbot_crisis_keyword_prioritizes_immediate_support(monkeypatch) -> None:
    monkeypatch.setattr("app.services.chatbot.config.CHATBOT_USE_REAL_LLM", True)
    monkeypatch.setattr("app.services.chatbot.config.OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Crisis safety response must not be sent to LLM rewrite")

    monkeypatch.setattr("ai_runtime.llm.llm_generator.call_llm_json", fail_if_called)

    response = await ask_chatbot(
        ChatbotAskRequest(message="요즘 죽고 싶다는 생각이 들어요", context_type=ChatbotContextType.MAIN)
    )

    assert response.source == "safety_policy"
    assert "챌린지 추천보다 안전 확보가 우선" in response.answer
    assert "보호자" in response.answer
    assert "전문기관" in response.answer
    assert "진단이 아니" in response.answer
    assert all("챌린지" not in action for action in response.recommended_actions)


@pytest.mark.asyncio
async def test_chatbot_depression_keyword_recommends_self_care_and_professional_support() -> None:
    response = await ask_chatbot(
        ChatbotAskRequest(message="번아웃이 심하고 너무 무기력해요", context_type=ChatbotContextType.MAIN)
    )

    assert response.source == "safety_policy"
    assert "자기관리 챌린지" in response.answer
    assert "전문 상담" in response.answer
    assert any("전문 상담" in action for action in response.recommended_actions)


@pytest.mark.asyncio
async def test_chatbot_stress_keyword_allows_low_burden_self_care_challenge() -> None:
    response = await ask_chatbot(
        ChatbotAskRequest(message="스트레스랑 불안 때문에 잠이 잘 안 와요", context_type=ChatbotContextType.MAIN)
    )

    assert response.source == "safety_policy"
    assert "정신건강 관련 자기관리 챌린지" in response.answer
    assert "전문 상담" in response.answer
    assert any("호흡" in action or "수면" in action for action in response.recommended_actions)
