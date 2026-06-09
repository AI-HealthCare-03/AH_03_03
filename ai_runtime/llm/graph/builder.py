from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .analysis_nodes import build_analysis_explanation
from .nodes import (
    CAUTION_MESSAGE,
    _metadata_with_node,
    build_recommended_actions,
    check_grounding_or_fallback,
    check_mental_health_safety,
    classify_intent,
    format_final_response,
    generate_llm_answer,
    normalize_input,
    retrieve_rag_context,
    sanitize_for_trace,
    should_bypass_llm,
    trace_graph_node,
)
from .state import HealthChatbotGraphState


@lru_cache
def build_health_chatbot_graph():
    graph = StateGraph(HealthChatbotGraphState)
    graph.add_node("normalize_input", _with_node_fallback("normalize_input", normalize_input))
    graph.add_node(
        "check_mental_health_safety",
        _with_node_fallback("check_mental_health_safety", check_mental_health_safety),
    )
    graph.add_node("classify_intent", _with_node_fallback("classify_intent", classify_intent))
    graph.add_node("retrieve_rag_context", _with_node_fallback("retrieve_rag_context", retrieve_rag_context))
    graph.add_node("generate_llm_answer", _with_node_fallback("generate_llm_answer", generate_llm_answer))
    graph.add_node(
        "check_grounding_or_fallback",
        _with_node_fallback("check_grounding_or_fallback", check_grounding_or_fallback),
    )
    graph.add_node(
        "build_recommended_actions", _with_node_fallback("build_recommended_actions", build_recommended_actions)
    )
    graph.add_node("format_final_response", _with_node_fallback("format_final_response", format_final_response))

    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "check_mental_health_safety")
    graph.add_conditional_edges(
        "check_mental_health_safety",
        should_bypass_llm,
        {
            "bypass": "build_recommended_actions",
            "continue": "classify_intent",
        },
    )
    graph.add_edge("classify_intent", "retrieve_rag_context")
    graph.add_edge("retrieve_rag_context", "generate_llm_answer")
    graph.add_edge("generate_llm_answer", "check_grounding_or_fallback")
    graph.add_edge("check_grounding_or_fallback", "build_recommended_actions")
    graph.add_edge("build_recommended_actions", "format_final_response")
    graph.add_edge("format_final_response", END)
    return graph.compile()


@lru_cache
def build_analysis_explanation_graph():
    graph = StateGraph(HealthChatbotGraphState)
    graph.add_node("build_analysis_explanation", build_analysis_explanation)
    graph.add_edge(START, "build_analysis_explanation")
    graph.add_edge("build_analysis_explanation", END)
    return graph.compile()


def _with_node_fallback(
    node_name: str,
    node_func: Callable[[HealthChatbotGraphState], HealthChatbotGraphState],
) -> Callable[[HealthChatbotGraphState], HealthChatbotGraphState]:
    def wrapped(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
        try:
            return node_func(state)
        except Exception as exc:
            fallback_state = _node_exception_fallback_state(node_name=node_name, state=state, exc=exc)
            trace_graph_node(
                node_name,
                fallback_state,
                {
                    "fallback_required": True,
                    "exception_type": type(exc).__name__,
                },
            )
            return fallback_state

    wrapped.__name__ = getattr(node_func, "__name__", node_name)
    return wrapped


def _node_exception_fallback_state(
    *,
    node_name: str,
    state: HealthChatbotGraphState,
    exc: Exception,
) -> HealthChatbotGraphState:
    exception_message = sanitize_for_trace(str(exc), limit=200)
    error_payload = {
        "node": node_name,
        "exception_type": type(exc).__name__,
        "exception_message": exception_message,
        "fallback_required": True,
    }
    fallback_answer = (
        "요청을 처리하는 중 일부 단계에서 오류가 발생했습니다. "
        f"현재는 안전한 기본 안내로 대신 답변드립니다. {CAUTION_MESSAGE}"
    )
    metadata = _metadata_with_node(state.get("metadata", {}), node_name)
    graph_errors = [*list(state.get("graph_errors") or []), error_payload]
    return {
        **state,
        "metadata": {
            **metadata,
            "graph_error": error_payload,
            "graph_errors": graph_errors,
        },
        "trace_metadata": {
            **state.get("trace_metadata", {}),
            "graph_error": error_payload,
        },
        "graph_error": error_payload,
        "graph_errors": graph_errors,
        "fallback_required": True,
        "fallback_reason": state.get("fallback_reason") or f"{node_name}_exception_fallback",
        "llm_answer": state.get("llm_answer") or fallback_answer,
        "final_answer": state.get("final_answer") if node_name != "format_final_response" else fallback_answer,
        "recommended_actions": state.get("recommended_actions") or [],
        "source": "rule_based_graph_fallback",
        "caution_message": state.get("caution_message") or CAUTION_MESSAGE,
        "is_safe": False,
        "safety_result": {
            **state.get("safety_result", {}),
            "is_safe": False,
            "metadata": {
                **state.get("safety_result", {}).get("metadata", {}),
                "graph_node": node_name,
                "fallback_required": True,
            },
        },
    }
