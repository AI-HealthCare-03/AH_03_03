from ai_runtime.llm.explanation_service import (
    generate_analysis_explanation,
    generate_diet_score_explanation,
    generate_explanation_with_context,
    retrieve_health_context,
)
from ai_runtime.llm.schemas import AnalysisExplanationInput, DietScoreExplanationInput, HealthRiskFactor


def test_analysis_explanation_is_rule_based_and_safe() -> None:
    explanation = generate_analysis_explanation(
        AnalysisExplanationInput(
            disease_type="DIABETES",
            risk_score=0.72,
            risk_level="HIGH_CAUTION",
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
    assert explanation.reference_sources == []


def test_diet_score_explanation_mentions_lowest_disease_score() -> None:
    explanation = generate_diet_score_explanation(
        DietScoreExplanationInput(disease_scores={"DM": 45, "HTN": 70, "DL": 80, "OBE": 77, "ANEM": 66})
    )

    assert explanation.source == "rule_based_explanation"
    assert "당뇨" in explanation.summary
    assert "혈당" in explanation.summary
    assert "진단이 아니" in explanation.safety_notice
    assert "의료진 상담" in explanation.safety_notice


def test_rag_ready_interface_adds_keyword_context_references(monkeypatch) -> None:
    import ai_runtime.llm.explanation_service as explanation_service

    monkeypatch.setattr(explanation_service.config, "RAG_ENABLED", True)

    contexts = retrieve_health_context("혈당 관리", disease_type="DIABETES")
    explanation = generate_explanation_with_context(
        AnalysisExplanationInput(disease_type="DIABETES", risk_level="LOW"),
        contexts=contexts,
        use_real_llm=False,
    )

    source_ids = [context.metadata["id"] for context in contexts]
    assert "diabetes" in source_ids
    assert "safety_disclaimer" in source_ids
    assert explanation.source == "rule_based_explanation"
    assert explanation.reference_summary
    assert explanation.reference_sources
    assert "진단이 아니" in explanation.safety_notice


def test_retrieve_health_context_returns_empty_when_rag_disabled(monkeypatch) -> None:
    import ai_runtime.llm.explanation_service as explanation_service

    def fail_if_called(*args, **kwargs):
        raise AssertionError("RAG retrieval should be skipped when disabled")

    monkeypatch.setattr(explanation_service.config, "RAG_ENABLED", False)
    monkeypatch.setattr(explanation_service, "retrieve_keyword_rag_contexts", fail_if_called)
    monkeypatch.setattr(explanation_service, "trace_keyword_rag_retrieval", fail_if_called)

    assert explanation_service.retrieve_health_context("혈당 관리", disease_type="DIABETES") == []


def test_retrieve_health_context_returns_empty_when_rag_loader_fails(monkeypatch) -> None:
    import ai_runtime.llm.explanation_service as explanation_service

    def raise_rag_error(*args, **kwargs):
        raise RuntimeError("index broken")

    monkeypatch.setattr(explanation_service.config, "RAG_ENABLED", True)
    monkeypatch.setattr(explanation_service, "retrieve_keyword_rag_contexts", raise_rag_error)

    assert explanation_service.retrieve_health_context("혈당 관리", disease_type="DIABETES") == []
