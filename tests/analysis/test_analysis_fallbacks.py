from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from app.models.analysis import AnalysisMode, AnalysisType, RiskLevel
from app.services import analysis as analysis_service


def _health_record(**overrides):
    values = {
        "id": 101,
        "family_dm": "NO",
        "family_htn": "NO",
        "family_dyslipidemia": "NO",
        "bmi": Decimal("26.0"),
        "waist_cm": Decimal("88.0"),
        "fasting_glucose": 108,
        "hba1c": Decimal("5.8"),
        "ldl_cholesterol": 132,
        "hdl_cholesterol": 46,
        "triglyceride": 158,
        "total_cholesterol": 210,
        "systolic_bp": 132,
        "diastolic_bp": 84,
        "smoking_status": "NEVER",
        "drinking_frequency": "NONE",
        "drinking_amount": "NONE",
        "walking_days_per_week": 5,
        "strength_days_per_week": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ml_prediction_failure_returns_empty_for_rule_based_fallback(monkeypatch) -> None:
    from ai_runtime.ml.inference import disease_risk_service

    def raise_prediction_error(*args, **kwargs):
        raise RuntimeError("catboost failed")

    monkeypatch.setattr(disease_risk_service, "predict_chronic_disease_risks", raise_prediction_error)

    result = analysis_service._predict_ml_outputs(SimpleNamespace(id=1), _health_record())

    assert result == {}


def test_precision_scores_include_obesity_rule_based_target() -> None:
    scores = analysis_service._calculate_analysis_scores(
        _health_record(),
        mode=AnalysisMode.PRECISION,
        user=SimpleNamespace(birthday=None),
    )

    assert AnalysisType.OBESITY in scores
    assert isinstance(scores[AnalysisType.OBESITY], Decimal)


def test_analysis_explanation_falls_back_when_context_generation_fails(monkeypatch) -> None:
    def raise_explanation_error(*args, **kwargs):
        raise RuntimeError("rag failed")

    monkeypatch.setattr(analysis_service, "generate_explanation_with_context", raise_explanation_error)

    explanation = analysis_service._analysis_explanation(
        SimpleNamespace(
            id=1,
            analysis_type=AnalysisType.DIABETES,
            risk_score=Decimal("0.72"),
            risk_level=RiskLevel.HIGH,
            model_name="catboost",
            model_version="dm_catboost_final",
        ),
        [],
    )

    assert explanation["source"] == "rule_based_explanation"
    assert "진단이 아니" in explanation["safety_notice"]
    assert "의료진 상담" in explanation["safety_notice"]
