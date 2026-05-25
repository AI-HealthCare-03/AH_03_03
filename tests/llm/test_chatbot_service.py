from __future__ import annotations

import pytest

from app.dtos.chatbot import ChatbotAskRequest, ChatbotContextType
from app.services.chatbot import ask_chatbot


@pytest.mark.asyncio
async def test_chatbot_uses_local_llm_rule_router_without_dummy_source() -> None:
    response = await ask_chatbot(
        ChatbotAskRequest(message="혈당 관리는 어떻게 하나요?", context_type=ChatbotContextType.MAIN)
    )

    assert response.source == "rule_engine"
    assert response.source != "DUMMY_LLM"
    assert "진단이 아니" in response.answer
    assert "의료진 상담" in response.safety_notice


@pytest.mark.asyncio
async def test_chatbot_real_llm_flag_without_openai_key_falls_back_to_rule_engine(monkeypatch) -> None:
    monkeypatch.setattr("app.services.chatbot.config.CHATBOT_USE_REAL_LLM", True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = await ask_chatbot(
        ChatbotAskRequest(message="혈압 관리는 어떻게 하나요?", context_type=ChatbotContextType.MAIN)
    )

    assert response.source == "rule_engine"
    assert "진단이 아니" in response.answer


@pytest.mark.asyncio
async def test_chatbot_real_llm_flag_uses_openai_rewrite_when_configured(monkeypatch) -> None:
    monkeypatch.setattr("app.services.chatbot.config.CHATBOT_USE_REAL_LLM", True)
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
