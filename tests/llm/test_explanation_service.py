from ai_worker.llm.explanation_service import (
    generate_analysis_explanation,
    generate_diet_score_explanation,
    generate_explanation_with_context,
    retrieve_health_context,
)
from ai_worker.llm.schemas import AnalysisExplanationInput, DietScoreExplanationInput, HealthRiskFactor


def test_analysis_explanation_is_rule_based_and_safe() -> None:
    explanation = generate_analysis_explanation(
        AnalysisExplanationInput(
            disease_type="DIABETES",
            risk_score=0.72,
            risk_level="HIGH",
            model_name="catboost",
            model_version="dm_catboost_final",
            factors=[HealthRiskFactor(name="공복혈당", value="132", reason="positive")],
        )
    )

    assert explanation.source == "rule_based_explanation"
    assert "당뇨" in explanation.summary
    assert "공복혈당" in explanation.caution
    assert "진단이 아니" in explanation.safety_notice
    assert "의료진 상담" in explanation.safety_notice


def test_diet_score_explanation_mentions_lowest_disease_score() -> None:
    explanation = generate_diet_score_explanation(
        DietScoreExplanationInput(disease_scores={"DM": 45, "HTN": 70, "DL": 80, "OBE": 77, "ANEM": 66})
    )

    assert explanation.source == "rule_based_explanation"
    assert "당뇨" in explanation.summary
    assert "혈당" in explanation.summary
    assert "진단이 아니" in explanation.safety_notice
    assert "의료진 상담" in explanation.safety_notice


def test_rag_ready_interface_uses_empty_context_fallback() -> None:
    contexts = retrieve_health_context("혈당 관리", disease_type="DIABETES")
    explanation = generate_explanation_with_context(
        AnalysisExplanationInput(disease_type="DIABETES", risk_level="LOW"),
        contexts=contexts,
        use_real_llm=False,
    )

    assert contexts == []
    assert explanation.source == "rule_based_explanation"
    assert "진단이 아니" in explanation.safety_notice
