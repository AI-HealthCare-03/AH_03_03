from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class HealthChatbotGraphState(TypedDict):
    user_message: str | None
    user_context: dict[str, Any]
    intent: str | None
    safety_level: str | None
    safety_response: str | None
    should_bypass_llm: bool
    retrieved_docs: list[dict[str, Any]]
    reference_sources: list[dict[str, Any]]
    reference_summary: str | None
    llm_answer: str | None
    final_answer: str | None
    recommended_actions: list[str]
    fallback_reason: str | None
    metadata: dict[str, Any]
    trace_metadata: dict[str, Any]
    source: str
    caution_message: str
    is_safe: bool
    safety_result: dict[str, Any]
    use_real_llm: bool
    use_rag: bool
    context_type: NotRequired[str | None]
    analysis_result: NotRequired[dict[str, Any] | None]
    analysis_type: NotRequired[str | None]
    analysis_explanation: NotRequired[dict[str, Any] | None]
    risk_factors: NotRequired[list[dict[str, Any]]]
    management_priorities: NotRequired[list[str]]
    analysis_contexts: NotRequired[list[Any]]
