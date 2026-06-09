import os

from ai_runtime.llm.graph import run_chatbot_graph_async
from app.core import config
from app.dtos.chatbot import ChatbotAskRequest, ChatbotAskResponse

SAFETY_NOTICE = "본 서비스는 진단/처방이 아니며, 치료 변경은 반드시 의료진과 상담해야 합니다."


async def ask_chatbot(request: ChatbotAskRequest) -> ChatbotAskResponse:
    use_real_llm = config.CHATBOT_USE_REAL_LLM and bool(os.getenv("OPENAI_API_KEY"))
    routed = await run_chatbot_graph_async(
        user_message=request.message,
        user_context={"target_id": request.target_id},
        context_type=request.context_type.value,
        use_real_llm=use_real_llm,
        use_rag=True,
    )
    return ChatbotAskResponse(
        answer=routed.answer,
        source=routed.source,
        context_type=request.context_type,
        recommended_actions=routed.recommended_actions,
        safety_notice=routed.caution_message,
    )
