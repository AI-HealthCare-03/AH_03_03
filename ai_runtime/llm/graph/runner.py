from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from inspect import iscoroutine
from typing import Any
from uuid import uuid4

from ai_runtime.llm.schemas import AnalysisExplanationInput, ExplanationOutput, RetrievedContext

from .builder import build_analysis_explanation_graph, build_health_chatbot_graph
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


@dataclass(frozen=True)
class AnalysisExplanationGraphResult:
    explanation: ExplanationOutput
    analysis_type: str | None
    risk_factors: list[dict[str, Any]]
    management_priorities: list[str]
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_metadata: dict[str, Any] = field(default_factory=dict)


async def run_chatbot_graph_async(
    *,
    user_message: str | None,
    user_context: dict[str, Any] | None = None,
    context_type: str | None = None,
    use_real_llm: bool = False,
    use_rag: bool = True,
) -> ChatbotGraphResult:
    graph = build_health_chatbot_graph()
    graph_run_id = uuid4().hex
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
        "metadata": {"graph_run_id": graph_run_id},
        "trace_metadata": {"graph_run_id": graph_run_id},
        "source": "langgraph_chatbot",
        "caution_message": "",
        "is_safe": True,
        "safety_result": {},
        "use_real_llm": use_real_llm,
        "use_rag": use_rag,
    }
    final_state = await graph.ainvoke(initial_state)
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


def run_chatbot_graph(
    *,
    user_message: str | None,
    user_context: dict[str, Any] | None = None,
    context_type: str | None = None,
    use_real_llm: bool = False,
    use_rag: bool = True,
) -> ChatbotGraphResult:
    return _run_coroutine_sync(
        run_chatbot_graph_async(
            user_message=user_message,
            user_context=user_context,
            context_type=context_type,
            use_real_llm=use_real_llm,
            use_rag=use_rag,
        )
    )


async def run_analysis_explanation_graph_async(
    *,
    input_data: AnalysisExplanationInput | None,
    contexts: list[RetrievedContext] | None = None,
    analysis_type: str | None = None,
    use_real_llm: bool = False,
) -> AnalysisExplanationGraphResult:
    graph = build_analysis_explanation_graph()
    graph_run_id = uuid4().hex
    initial_state: HealthChatbotGraphState = {
        "user_message": None,
        "user_context": {},
        "context_type": "ANALYSIS",
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
        "metadata": {"graph_run_id": graph_run_id},
        "trace_metadata": {"graph_run_id": graph_run_id},
        "source": "analysis_explanation_graph",
        "caution_message": "",
        "is_safe": True,
        "safety_result": {},
        "use_real_llm": bool(use_real_llm),
        "use_rag": bool(contexts),
        "analysis_result": input_data.model_dump() if input_data is not None else None,
        "analysis_type": analysis_type,
        "analysis_explanation": None,
        "risk_factors": [],
        "management_priorities": [],
        "analysis_contexts": [context.model_dump() for context in (contexts or [])],
    }
    final_state = await graph.ainvoke(initial_state)
    explanation_payload = final_state.get("analysis_explanation") or {}
    explanation = ExplanationOutput.model_validate(explanation_payload)
    return AnalysisExplanationGraphResult(
        explanation=explanation,
        analysis_type=final_state.get("analysis_type"),
        risk_factors=list(final_state.get("risk_factors") or []),
        management_priorities=list(final_state.get("management_priorities") or []),
        fallback_reason=final_state.get("fallback_reason"),
        metadata=dict(final_state.get("metadata") or {}),
        trace_metadata=dict(final_state.get("trace_metadata") or {}),
    )


def run_analysis_explanation_graph(
    *,
    input_data: AnalysisExplanationInput | None,
    contexts: list[RetrievedContext] | None = None,
    analysis_type: str | None = None,
    use_real_llm: bool = False,
) -> AnalysisExplanationGraphResult:
    return _run_coroutine_sync(
        run_analysis_explanation_graph_async(
            input_data=input_data,
            contexts=contexts,
            analysis_type=analysis_type,
            use_real_llm=use_real_llm,
        )
    )


def _run_coroutine_sync(coroutine) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    if iscoroutine(coroutine):
        coroutine.close()
    raise RuntimeError("LangGraph runner is already in an event loop; use the async graph runner instead.")
