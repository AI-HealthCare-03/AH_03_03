from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .analysis_nodes import build_analysis_explanation
from .nodes import (
    build_recommended_actions,
    check_grounding_or_fallback,
    check_mental_health_safety,
    classify_intent,
    format_final_response,
    generate_llm_answer,
    normalize_input,
    retrieve_rag_context,
    should_bypass_llm,
)
from .state import HealthChatbotGraphState


@lru_cache
def build_health_chatbot_graph():
    graph = StateGraph(HealthChatbotGraphState)
    graph.add_node("normalize_input", normalize_input)
    graph.add_node("check_mental_health_safety", check_mental_health_safety)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_rag_context", retrieve_rag_context)
    graph.add_node("generate_llm_answer", generate_llm_answer)
    graph.add_node("check_grounding_or_fallback", check_grounding_or_fallback)
    graph.add_node("build_recommended_actions", build_recommended_actions)
    graph.add_node("format_final_response", format_final_response)

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
