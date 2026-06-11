from __future__ import annotations

from ai_runtime.llm.graph import run_analysis_explanation_graph, run_chatbot_graph
from ai_runtime.llm.prompt_templates import ANALYSIS_EXPLANATION_PROMPT_VERSION
from ai_runtime.llm.schemas import AnalysisExplanationInput, HealthRiskFactor, RetrievedContext


def test_basic_analysis_explanation_node_is_safe_and_non_diagnostic() -> None:
    result = run_analysis_explanation_graph(
        input_data=AnalysisExplanationInput(
            disease_type="DIABETES",
            risk_score=0.42,
            risk_level="CAUTION",
            factors=[HealthRiskFactor(name="생활습관", value=None, reason="positive")],
        ),
        analysis_type="basic",
        use_real_llm=False,
    )

    explanation = result.explanation
    assert result.analysis_type == "basic"
    assert explanation.source == "rule_based_explanation"
    assert "주의" in explanation.summary
    assert "참고" in explanation.caution
    assert "진단이 아니" in explanation.safety_notice
    assert "의료진 상담" in explanation.safety_notice
    assert "당뇨입니다" not in explanation.summary
    assert "복용하세요" not in explanation.recommended_action
    assert result.trace_metadata["analysis_explanation"]["prompt_version"] == ANALYSIS_EXPLANATION_PROMPT_VERSION
    assert result.trace_metadata["analysis_explanation"]["risk_factor_count"] == 1
    assert result.trace_metadata["analysis_explanation"]["analysis_type"] == "basic"


def test_precision_analysis_explanation_focuses_risk_factors_and_priorities() -> None:
    result = run_analysis_explanation_graph(
        input_data=AnalysisExplanationInput(
            disease_type="HYPERTENSION",
            risk_score=0.73,
            risk_level="HIGH_CAUTION",
            model_name="catboost",
            model_version="htn_catboost_final",
            factors=[
                HealthRiskFactor(name="수축기 혈압", value="132", reason="positive"),
                HealthRiskFactor(name="나트륨 섭취", value=None, reason="positive"),
            ],
        ),
        use_real_llm=False,
    )

    explanation = result.explanation
    assert result.analysis_type == "precision"
    assert "고혈압" in explanation.summary
    assert "CatBoost 모델" in explanation.summary
    assert "수축기 혈압" in explanation.caution
    assert "나트륨" in explanation.recommended_action
    assert result.risk_factors == [
        {"name": "수축기 혈압", "reason": "positive"},
        {"name": "나트륨 섭취", "reason": "positive"},
    ]
    assert result.management_priorities
    assert result.trace_metadata["analysis_explanation"]["risk_factor_count"] == 2
    assert result.trace_metadata["analysis_explanation"]["management_priority_count"] == 1


def test_analysis_explanation_node_returns_safe_fallback_without_result() -> None:
    result = run_analysis_explanation_graph(input_data=None, use_real_llm=False)

    explanation = result.explanation
    assert explanation.source == "rule_based_explanation_fallback"
    assert result.fallback_reason == "analysis_result_missing"
    assert result.analysis_type is None
    assert result.risk_factors == []
    assert result.management_priorities
    assert "정보가 부족" in explanation.summary
    assert "진단이 아니" in explanation.safety_notice
    assert result.trace_metadata["analysis_explanation"]["fallback_reason"] == "analysis_result_missing"


def test_analysis_explanation_node_keeps_reference_sources_without_document_body() -> None:
    result = run_analysis_explanation_graph(
        input_data=AnalysisExplanationInput(disease_type="DIABETES", risk_level="LOW"),
        contexts=[
            RetrievedContext(
                title="당뇨 관리 기준",
                content="민감한 본문 전체는 trace에 남기지 않는다.",
                source_name="대한당뇨병학회",
                url="https://example.test/diabetes",
                metadata={"id": "diabetes", "year": 2024, "status": "candidate_unreviewed"},
            )
        ],
        use_real_llm=False,
    )

    explanation = result.explanation
    assert explanation.reference_summary
    assert explanation.reference_sources[0]["id"] == "diabetes"
    trace = result.trace_metadata["analysis_explanation"]
    assert trace["reference_source_ids"] == ["diabetes"]
    assert trace["reference_source_types"] == ["대한당뇨병학회"]
    assert "민감한 본문" not in str(trace)


def test_crisis_chatbot_path_does_not_run_analysis_explanation_node() -> None:
    result = run_chatbot_graph(
        user_message="요즘 죽고 싶다는 생각이 들어요",
        context_type="MAIN",
        use_real_llm=True,
    )

    assert result.source == "safety_policy"
    assert "build_analysis_explanation" not in result.metadata["node_path"]
    assert "analysis_explanation" not in result.trace_metadata
