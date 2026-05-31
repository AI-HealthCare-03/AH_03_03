from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .builder import build_health_chatbot_graph
from .state import HealthChatbotGraphState


@dataclass(frozen=True)
class ChatbotGraphResult:
    answer: str
    source: str
    intent: str | None
    safety_level: str | None
    recommended_actions: list[str]
    caution_message: str
    is_safe: bool
    fallback_reason: str | None = None
    safety_result: dict[str, Any] = field(default_factory=dict)
    reference_sources: list[dict[str, Any]] = field(default_factory=list)
    reference_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_metadata: dict[str, Any] = field(default_factory=dict)


def run_chatbot_graph(
    *,
    user_message: str | None,
    user_context: dict[str, Any] | None = None,
    context_type: str | None = None,
    use_real_llm: bool = False,
    use_rag: bool = True,
) -> ChatbotGraphResult:
    graph = build_health_chatbot_graph()
    initial_state: HealthChatbotGraphState = {
        "user_message": user_message,
        "user_context": user_context or {},
        "context_type": context_type,
        "intent": None,
        "safety_level": None,
        "safety_response": None,
        "should_bypass_llm": False,
        "retrieved_docs": [],
        "reference_sources": [],
        "reference_summary": None,
        "llm_answer": None,
        "final_answer": None,
        "recommended_actions": [],
        "fallback_reason": None,
        "metadata": {},
        "trace_metadata": {},
        "source": "langgraph_chatbot",
        "caution_message": "",
        "is_safe": True,
        "safety_result": {},
        "use_real_llm": use_real_llm,
        "use_rag": use_rag,
    }
    final_state = graph.invoke(initial_state)
    return ChatbotGraphResult(
        answer=str(final_state.get("final_answer") or final_state.get("llm_answer") or ""),
        source=str(final_state.get("source") or "langgraph_chatbot"),
        intent=final_state.get("intent"),
        safety_level=final_state.get("safety_level"),
        recommended_actions=list(final_state.get("recommended_actions") or []),
        caution_message=str(final_state.get("caution_message") or ""),
        is_safe=bool(final_state.get("is_safe", True)),
        fallback_reason=final_state.get("fallback_reason"),
        safety_result=dict(final_state.get("safety_result") or {}),
        reference_sources=list(final_state.get("reference_sources") or []),
        reference_summary=final_state.get("reference_summary"),
        metadata=dict(final_state.get("metadata") or {}),
        trace_metadata=dict(final_state.get("trace_metadata") or {}),
    )
