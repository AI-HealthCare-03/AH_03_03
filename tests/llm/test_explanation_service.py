from ai_runtime.llm.explanation_service import (
    _build_analysis_explanation_rewrite_prompt,
    generate_analysis_explanation,
    generate_diet_score_explanation,
    generate_explanation_with_context,
    retrieve_health_context,
)
from ai_runtime.llm.schemas import (
    AnalysisExplanationInput,
    DietScoreExplanationInput,
    ExplanationOutput,
    HealthRiskFactor,
)


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
    assert "diet_caution" in source_ids
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


def test_analysis_explanation_llm_rewrite_flag_default_is_false() -> None:
    import ai_runtime.llm.explanation_service as explanation_service

    assert explanation_service.config.ANALYSIS_EXPLANATION_LLM_REWRITE_ENABLED is False


def test_analysis_explanation_rewrite_disabled_does_not_call_openai(monkeypatch) -> None:
    import ai_runtime.llm.explanation_service as explanation_service

    def fail_if_called(*args, **kwargs):
        raise AssertionError("OpenAI rewrite must not be called when analysis rewrite flag is disabled")

    monkeypatch.setattr(explanation_service.config, "ANALYSIS_EXPLANATION_LLM_REWRITE_ENABLED", False)
    monkeypatch.setattr(explanation_service, "call_llm_json", fail_if_called)

    explanation = generate_explanation_with_context(
        AnalysisExplanationInput(
            disease_type="HYPERTENSION",
            risk_level="CAUTION",
            factors=[HealthRiskFactor(name="수축기 혈압", value="132", reason="positive")],
        ),
        contexts=[],
    )

    assert explanation.source == "rule_based_explanation"
    assert "고혈압" in explanation.summary


def test_analysis_explanation_rewrite_prompt_limits_llm_to_rewrite_only() -> None:
    prompt = _build_analysis_explanation_rewrite_prompt(
        input_data=AnalysisExplanationInput(
            disease_type="DIABETES",
            risk_level="CAUTION",
            factors=[HealthRiskFactor(name="공복혈당", value="132", reason="positive")],
        ),
        explanation=ExplanationOutput(
            summary="당뇨 관련 위험도는 주의로 분류되었습니다.",
            caution="공복혈당 결과는 참고 신호입니다.",
            recommended_action="식후 활동과 당류 섭취를 관리해 보세요.",
            safety_notice="이 설명은 진단이 아니며, 건강관리 참고용입니다.",
        ),
    )

    assert "새로 판단하지 않는다" in prompt
    assert "진단, 확진, 치료, 처방 판단을 하지 않는다" in prompt
    assert "입력에 없는 질환, 약, 검사, 수치, 챌린지를 추가하지 않는다" in prompt
    assert "수치가 있으면 입력으로 제공된 수치만 사용한다" in prompt


def test_analysis_explanation_rewrite_enabled_uses_fake_llm_without_new_judgment(monkeypatch) -> None:
    import ai_runtime.llm.explanation_service as explanation_service

    def fake_call_llm_json(*args, **kwargs):
        return (
            '{"summary":"입력된 결과 기준으로 혈압 관리에 참고할 부분을 정리했어요.",'
            '"caution":"수축기 혈압 항목은 생활관리 참고 신호로 살펴보면 좋습니다.",'
            '"recommended_action":"혈압 기록과 나트륨 섭취를 함께 점검해 보세요."}'
        )

    monkeypatch.setattr(explanation_service.config, "ANALYSIS_EXPLANATION_LLM_REWRITE_ENABLED", True)
    monkeypatch.setattr(explanation_service.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(explanation_service, "call_llm_json", fake_call_llm_json)

    explanation = generate_explanation_with_context(
        AnalysisExplanationInput(
            disease_type="HYPERTENSION",
            risk_level="CAUTION",
            factors=[HealthRiskFactor(name="수축기 혈압", value="132", reason="positive")],
        ),
        contexts=[],
    )

    assert explanation.summary == "입력된 결과 기준으로 혈압 관리에 참고할 부분을 정리했어요."
    assert "처방" not in explanation.summary
    assert "진단" not in explanation.summary
    assert "생활관리 참고" in explanation.caution
