from ai_runtime.llm.graph import run_chatbot_graph_async
from app.core import config
from app.core.providers import has_openai_config
from app.dtos.chatbot import ChatbotAskRequest, ChatbotAskResponse

SAFETY_NOTICE = "본 서비스는 진단/처방이 아니며, 치료 변경은 반드시 의료진과 상담해야 합니다."


async def ask_chatbot(request: ChatbotAskRequest) -> ChatbotAskResponse:
    response, _ = await ask_chatbot_with_runtime_trace(request)
    return response


async def ask_chatbot_with_runtime_trace(request: ChatbotAskRequest) -> tuple[ChatbotAskResponse, dict]:
    use_real_llm = config.CHATBOT_USE_REAL_LLM and has_openai_config(config)
    routed = await run_chatbot_graph_async(
        user_message=request.message,
        user_context={"target_id": request.target_id},
        context_type=request.context_type.value,
        use_real_llm=use_real_llm,
        use_rag=True,
    )
    return (
        ChatbotAskResponse(
            answer=routed.answer,
            source=routed.source,
            context_type=request.context_type,
            recommended_actions=routed.recommended_actions,
            safety_notice=routed.caution_message,
        ),
        build_chatbot_rag_runtime_trace(routed.trace_metadata),
    )


def build_chatbot_rag_runtime_trace(trace_metadata: dict | None) -> dict:
    retrieval = (trace_metadata or {}).get("retrieval") or {}
    if not isinstance(retrieval, dict):
        return {}

    return {
        "rag_strategy": retrieval.get("retriever_strategy") or retrieval.get("strategy") or retrieval.get("source"),
        "keyword_returned_count": retrieval.get("keyword_returned_count"),
        "vector_returned_count": retrieval.get("vector_returned_count"),
        "merged_count": retrieval.get("merged_count"),
        "final_count": retrieval.get("final_count") or retrieval.get("document_count"),
        "fallback_used": retrieval.get("fallback_used", retrieval.get("fallback")),
        "fallback_reason": retrieval.get("fallback_reason"),
    }
