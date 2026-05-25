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
