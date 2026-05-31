from __future__ import annotations

from time import perf_counter
from typing import Any

from ai_runtime.llm.explanation_service import SAFETY_NOTICE, generate_analysis_explanation
from ai_runtime.llm.prompt_templates import ANALYSIS_EXPLANATION_PROMPT_VERSION
from ai_runtime.llm.rag.rag_context_builder import build_reference_sources, build_reference_summary
from ai_runtime.llm.schemas import AnalysisExplanationInput, ExplanationOutput, RetrievedContext

from .nodes import _metadata_with_node, trace_graph_node
from .state import HealthChatbotGraphState


def build_analysis_explanation(state: HealthChatbotGraphState) -> HealthChatbotGraphState:
    """Build safe analysis-result explanation without changing API contracts."""
    node_started_at = perf_counter()
    input_data, fallback_reason = _analysis_input_from_state(state)
    analysis_type = state.get("analysis_type") or _infer_analysis_type(input_data)
    reference_contexts = _analysis_contexts_from_state(state)

    if input_data is None:
        explanation = _fallback_explanation("analysis_result_missing")
        fallback_reason = fallback_reason or "analysis_result_missing"
    else:
        explanation = generate_analysis_explanation(input_data)
        if reference_contexts:
            explanation = explanation.model_copy(
                update={
                    "reference_summary": build_reference_summary(reference_contexts),
                    "reference_sources": build_reference_sources(reference_contexts),
                }
            )

    risk_factors = _risk_factors_from_input(input_data)
    management_priorities = _management_priorities_from_explanation(explanation)
    prompt_version = ANALYSIS_EXPLANATION_PROMPT_VERSION
    next_state: HealthChatbotGraphState = {
        **state,
        "metadata": _metadata_with_node(
            state.get("metadata", {}), "build_analysis_explanation", started_at=node_started_at
        ),
        "analysis_type": analysis_type,
        "analysis_explanation": explanation.model_dump(),
        "risk_factors": risk_factors,
        "management_priorities": management_priorities,
        "fallback_reason": state.get("fallback_reason") or fallback_reason,
        "source": explanation.source,
        "trace_metadata": {
            **state.get("trace_metadata", {}),
            "analysis_explanation": {
                "analysis_type": analysis_type,
                "risk_factor_count": len(risk_factors),
                "management_priority_count": len(management_priorities),
                "prompt_version": prompt_version,
                "fallback_reason": state.get("fallback_reason") or fallback_reason,
                "source": explanation.source,
                "reference_source_ids": [
                    source.get("id") for source in (explanation.reference_sources or []) if source.get("id")
                ],
                "reference_source_types": [
                    source.get("source_org")
                    for source in (explanation.reference_sources or [])
                    if source.get("source_org")
                ],
            },
        },
    }
    trace_graph_node(
        "build_analysis_explanation",
        next_state,
        {
            "analysis_type": analysis_type,
            "risk_factor_count": len(risk_factors),
            "prompt_version": prompt_version,
            "fallback_reason": next_state.get("fallback_reason"),
            "source": explanation.source,
        },
    )
    return next_state


def _analysis_input_from_state(
    state: HealthChatbotGraphState,
) -> tuple[AnalysisExplanationInput | None, str | None]:
    payload = state.get("analysis_result")
    if isinstance(payload, AnalysisExplanationInput):
        return payload, None
    if not isinstance(payload, dict):
        return None, "analysis_result_missing"
    try:
        return AnalysisExplanationInput.model_validate(payload), None
    except Exception:
        return None, "analysis_result_invalid"


def _analysis_contexts_from_state(state: HealthChatbotGraphState) -> list[RetrievedContext]:
    contexts = []
    for item in state.get("analysis_contexts", []) or []:
        if isinstance(item, RetrievedContext):
            contexts.append(item)
        elif isinstance(item, dict):
            contexts.append(RetrievedContext.model_validate(item))
    return contexts


def _infer_analysis_type(input_data: AnalysisExplanationInput | None) -> str | None:
    if input_data is None:
        return None
    if input_data.model_name or input_data.model_version:
        return "precision"
    return "basic"


def _risk_factors_from_input(input_data: AnalysisExplanationInput | None) -> list[dict[str, Any]]:
    if input_data is None:
        return []
    return [
        {
            "name": factor.name,
            "reason": factor.reason,
        }
        for factor in input_data.factors
    ]


def _management_priorities_from_explanation(explanation: ExplanationOutput) -> list[str]:
    priorities = [explanation.recommended_action.strip()]
    return [priority for priority in priorities if priority]


def _fallback_explanation(reason: str) -> ExplanationOutput:
    return ExplanationOutput(
        summary="분석 결과 설명에 필요한 정보가 부족합니다.",
        caution="현재 입력만으로는 위험요인과 관리 우선순위를 자세히 정리하기 어렵습니다.",
        recommended_action="건강정보와 검진 기록을 확인한 뒤 다시 분석해 보세요.",
        safety_notice=SAFETY_NOTICE,
        source="rule_based_explanation_fallback",
        reference_summary=None,
        reference_sources=[],
    )
