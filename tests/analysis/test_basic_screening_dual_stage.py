from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from ai_runtime.ml.inference import screening_risk_service
from ai_runtime.ml.inference.screening_predictor import ScreeningPrediction
from app.models.analysis import AnalysisMode, AnalysisType, RiskLevel
from app.services import analysis as analysis_service


def _health_record(**overrides: Any) -> SimpleNamespace:
    values = {
        "id": 101,
        "height_cm": Decimal("172.0"),
        "weight_kg": Decimal("77.0"),
        "family_dm": "NO",
        "family_htn": "NO",
        "family_dyslipidemia": "NO",
        "occupation_code": "PROFESSIONAL",
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


@pytest.mark.asyncio
async def test_basic_analysis_adds_screening_service_band_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots: list[Any] = []
    screening_calls: list[AnalysisType] = []

    async def fake_get_user(**kwargs: Any) -> SimpleNamespace:
        assert kwargs == {"id": 7}
        return SimpleNamespace(id=7, birthday=date(1975, 1, 1), gender="MALE")

    def fake_screening_dual_stage(**kwargs: Any) -> dict[str, Any] | None:
        analysis_type = kwargs["analysis_type"]
        if analysis_type == AnalysisType.OBESITY:
            return None
        screening_calls.append(analysis_type)
        return {
            "status": "applied",
            "disease_code": {
                AnalysisType.HYPERTENSION: "HTN",
                AnalysisType.DIABETES: "DM",
                AnalysisType.DYSLIPIDEMIA: "DL",
            }[analysis_type],
            "base_risk_level": kwargs["base_risk_level"].value,
            "base_high": False,
            "screening_high": True,
            "risk_level": "CAUTION",
            "service_band": "CAUTION",
            "service_band_label": "주의",
            "service_band_percent": 65,
            "legacy_risk_level": "CAUTION",
            "screening_missing_features": ["성별"],
            "screening_neutralized_features": ["성별"],
            "screening_model_count": 5,
        }

    async def fake_create_analysis_result(user_id: int, request: Any) -> SimpleNamespace:
        return SimpleNamespace(
            id=len(snapshots) + 1,
            user_id=user_id,
            health_record_id=request.health_record_id,
            analysis_type=request.analysis_type,
            analysis_mode=request.analysis_mode,
            risk_score=request.risk_score,
            risk_level=request.risk_level,
            summary=request.summary,
            model_name=request.model_name,
            model_version=request.model_version,
        )

    async def fake_create_analysis_factors(analysis_result_id: int, factors: list[Any]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                factor_key=factor.factor_key,
                factor_name=factor.factor_name,
                factor_value=factor.factor_value,
                contribution_score=factor.contribution_score,
                direction=factor.direction,
            )
            for factor in factors
        ]

    async def fake_create_analysis_snapshot(analysis_result_id: int, request: Any) -> SimpleNamespace:
        snapshots.append(request)
        return SimpleNamespace(id=analysis_result_id)

    monkeypatch.setattr(analysis_service.User, "get_or_none", fake_get_user)
    monkeypatch.setattr(analysis_service, "_predict_basic_screening_dual_stage", fake_screening_dual_stage)
    monkeypatch.setattr(analysis_service, "create_analysis_result", fake_create_analysis_result)
    monkeypatch.setattr(analysis_service, "create_analysis_factors", fake_create_analysis_factors)
    monkeypatch.setattr(analysis_service, "create_analysis_snapshot", fake_create_analysis_snapshot)
    monkeypatch.setattr(analysis_service, "_create_challenge_recommendations", _empty_recommendations)
    monkeypatch.setattr(analysis_service, "_analysis_explanation", lambda result, factors: {})

    results = await analysis_service.run_analysis(7, _health_record(), AnalysisMode.BASIC)

    assert set(screening_calls) == {
        AnalysisType.HYPERTENSION,
        AnalysisType.DIABETES,
        AnalysisType.DYSLIPIDEMIA,
    }
    assert len(results) == 4
    assert all("probability" not in result for result in results)
    dual_stage_results = [result for result in results if result["analysis_type"] != AnalysisType.OBESITY]
    assert all(result["service_band"] == "CAUTION" for result in dual_stage_results)
    assert all(result["service_band_label"] == "주의" for result in dual_stage_results)
    assert all(result["service_band_percent"] == 65 for result in dual_stage_results)
    assert all(result["risk_level"] == RiskLevel.CAUTION for result in dual_stage_results)
    assert all(result["risk_level"] in RiskLevel for result in results)

    dual_stage_snapshots = [
        snapshot for snapshot in snapshots if snapshot.input_payload["analysis_type"] != AnalysisType.OBESITY.value
    ]
    assert all(
        snapshot.output_payload["final_outputs"]["service_band"] == "CAUTION" for snapshot in dual_stage_snapshots
    )
    assert all(
        snapshot.model_payload["screening_dual_stage"]["screening_neutralized_features"] == ["성별"]
        for snapshot in dual_stage_snapshots
    )
    assert all("probability" not in str(snapshot.model_payload) for snapshot in dual_stage_snapshots)


def test_screening_dual_stage_failure_keeps_basic_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analysis_service,
        "load_screening_artifact",
        lambda disease_code: SimpleNamespace(feature_columns=["나이", "BMI"]),
    )

    def raise_screening_error(**kwargs: Any) -> None:
        _ = kwargs
        raise RuntimeError("screening failed")

    monkeypatch.setattr(analysis_service, "predict_screening_dual_stage_risk", raise_screening_error)

    result = analysis_service._predict_basic_screening_dual_stage(
        user=SimpleNamespace(id=7, birthday=date(1975, 1, 1), gender="MALE"),
        health_record=_health_record(),
        analysis_type=AnalysisType.HYPERTENSION,
        base_risk_level=RiskLevel.CAUTION,
        analysis_mode=AnalysisMode.BASIC,
    )

    assert result is not None
    assert result["status"] == "fallback_basic_only"
    assert result["base_risk_level"] == RiskLevel.CAUTION.value
    assert result["fallback_reason"] == "RuntimeError"


def test_screening_dual_stage_does_not_apply_to_non_target_disease() -> None:
    result = analysis_service._predict_basic_screening_dual_stage(
        user=SimpleNamespace(id=7, birthday=date(1975, 1, 1), gender="MALE"),
        health_record=_health_record(),
        analysis_type=AnalysisType.OBESITY,
        base_risk_level=RiskLevel.LOW,
        analysis_mode=AnalysisMode.BASIC,
    )

    assert result is None


@pytest.mark.parametrize(
    ("bmi", "expected_risk_level"),
    [
        (Decimal("18.4"), RiskLevel.LOW),
        (Decimal("22.9"), RiskLevel.LOW),
        (Decimal("23.0"), RiskLevel.ATTENTION),
        (Decimal("24.9"), RiskLevel.ATTENTION),
        (Decimal("25.0"), RiskLevel.CAUTION),
        (Decimal("29.9"), RiskLevel.CAUTION),
        (Decimal("30.0"), RiskLevel.HIGH_CAUTION),
        (Decimal("34.9"), RiskLevel.HIGH_CAUTION),
        (Decimal("35.0"), RiskLevel.HIGH_CAUTION),
    ],
)
def test_basic_obesity_uses_final_bmi_rule(bmi: Decimal, expected_risk_level: RiskLevel) -> None:
    record = _health_record(bmi=bmi)
    score = analysis_service._basic_obesity_score(record, SimpleNamespace(birthday=None))

    assert analysis_service._risk_level_for_analysis_score(AnalysisType.OBESITY, score, record) == expected_risk_level


def test_basic_obesity_can_calculate_bmi_from_height_and_weight() -> None:
    record = _health_record(bmi=None, height_cm=Decimal("170.0"), weight_kg=Decimal("80.0"))
    score = analysis_service._basic_obesity_score(record, SimpleNamespace(birthday=None))

    assert analysis_service._risk_level_for_analysis_score(AnalysisType.OBESITY, score, record) == RiskLevel.CAUTION


def test_caution_base_score_with_screening_high_keeps_caution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        analysis_service,
        "load_screening_artifact",
        lambda disease_code: SimpleNamespace(feature_columns=["나이", "BMI"]),
    )

    def fake_predict_screening_risk(disease_code: str, features: dict[str, Any]) -> ScreeningPrediction:
        assert disease_code == "HTN"
        assert "나이" in features
        assert "BMI" in features
        return ScreeningPrediction(
            disease_code=disease_code,
            probability=0.9,
            threshold=0.45,
            screening_high=True,
            missing_features=[],
            neutralized_features=[],
            model_count=5,
        )

    monkeypatch.setattr(
        screening_risk_service.screening_predictor,
        "predict_screening_risk",
        fake_predict_screening_risk,
    )

    result = analysis_service._predict_basic_screening_dual_stage(
        user=SimpleNamespace(id=7, birthday=date(1975, 1, 1), gender="MALE"),
        health_record=_health_record(),
        analysis_type=AnalysisType.HYPERTENSION,
        base_risk_level=RiskLevel.CAUTION,
        analysis_mode=AnalysisMode.BASIC,
    )

    assert result is not None
    assert result["status"] == "applied"
    assert result["base_risk_level"] == RiskLevel.CAUTION.value
    assert result["base_high"] is True
    assert result["base_caution_or_above"] is True
    assert result["screening_high"] is True
    assert result["risk_level"] == RiskLevel.CAUTION.value
    assert result["service_band"] == RiskLevel.CAUTION.value
    assert result["service_band_label"] == "주의"
    assert result["service_band_percent"] == 65
    assert "probability" not in result
    assert "screening_probability" not in result


def test_strict_dual_stage_v2_promotes_caution_to_high_caution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis_service.config, "ML_DUAL_STAGE_POLICY_VERSION", "v2")
    monkeypatch.setattr(analysis_service.config, "ENABLE_STRICT_DUAL_STAGE", False)
    monkeypatch.setattr(
        analysis_service,
        "load_screening_artifact",
        lambda disease_code: SimpleNamespace(feature_columns=["나이", "BMI"]),
    )

    def fake_screening_dual_stage(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            disease_code=kwargs["disease_code"],
            base_risk_level=kwargs["base_risk_level"],
            base_high=True,
            base_caution_or_above=True,
            screening_high=False,
            risk_level=RiskLevel.CAUTION.value,
            service_band=SimpleNamespace(value=RiskLevel.CAUTION.value),
            service_band_label="주의",
            service_band_percent=65,
            legacy_risk_level=RiskLevel.CAUTION.value,
            screening_missing_features=[],
            screening_neutralized_features=[],
            screening_model_count=5,
        )

    monkeypatch.setattr(analysis_service, "predict_screening_dual_stage_risk", fake_screening_dual_stage)
    monkeypatch.setattr(
        analysis_service,
        "_predict_basic_strict_signal",
        lambda **kwargs: {
            "status": "applied",
            "strict_high": True,
            "strict_model_count": 5,
            "strict_model_name": "catboost",
            "strict_model_version": "htn_catboost_final",
        },
    )

    result = analysis_service._predict_basic_screening_dual_stage(
        user=SimpleNamespace(id=7, birthday=date(1975, 1, 1), gender="MALE"),
        health_record=_health_record(),
        analysis_type=AnalysisType.HYPERTENSION,
        base_risk_level=RiskLevel.CAUTION,
        analysis_mode=AnalysisMode.BASIC,
    )

    assert result is not None
    assert result["policy_version"] == "v2"
    assert result["screening_high"] is False
    assert result["strict_high"] is True
    assert result["screening_model_count"] == 5
    assert result["strict_model_count"] == 5
    assert result["risk_level"] == RiskLevel.HIGH_CAUTION.value
    assert result["service_band"] == RiskLevel.HIGH_CAUTION.value
    assert result["service_band_label"] == "높은 주의"
    assert result["service_band_percent"] == 80
    assert "probability" not in str(result)


def test_strict_dual_stage_v2_low_with_two_model_signals_is_capped_at_caution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis_service.config, "ML_DUAL_STAGE_POLICY_VERSION", "v2")
    monkeypatch.setattr(
        analysis_service,
        "load_screening_artifact",
        lambda disease_code: SimpleNamespace(feature_columns=["나이", "BMI"]),
    )

    def fake_screening_dual_stage(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            disease_code=kwargs["disease_code"],
            base_risk_level=kwargs["base_risk_level"],
            base_high=False,
            base_caution_or_above=False,
            screening_high=True,
            risk_level=RiskLevel.CAUTION.value,
            service_band=SimpleNamespace(value=RiskLevel.CAUTION.value),
            service_band_label="주의",
            service_band_percent=65,
            legacy_risk_level=RiskLevel.CAUTION.value,
            screening_missing_features=[],
            screening_neutralized_features=[],
            screening_model_count=5,
        )

    monkeypatch.setattr(analysis_service, "predict_screening_dual_stage_risk", fake_screening_dual_stage)
    monkeypatch.setattr(
        analysis_service,
        "_predict_basic_strict_signal",
        lambda **kwargs: {
            "status": "applied",
            "strict_high": True,
            "strict_model_count": 5,
            "strict_model_name": "catboost",
            "strict_model_version": "dm_catboost_final",
        },
    )

    result = analysis_service._predict_basic_screening_dual_stage(
        user=SimpleNamespace(id=7, birthday=date(1975, 1, 1), gender="MALE"),
        health_record=_health_record(),
        analysis_type=AnalysisType.DIABETES,
        base_risk_level=RiskLevel.LOW,
        analysis_mode=AnalysisMode.BASIC,
    )

    assert result is not None
    assert result["policy_version"] == "v2"
    assert result["screening_high"] is True
    assert result["strict_high"] is True
    assert result["risk_level"] == RiskLevel.CAUTION.value
    assert result["service_band"] == RiskLevel.CAUTION.value
    assert result["service_band_percent"] == 65


def test_strict_dual_stage_v2_strict_failure_falls_back_to_screening_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis_service.config, "ML_DUAL_STAGE_POLICY_VERSION", "v2")
    monkeypatch.setattr(
        analysis_service,
        "load_screening_artifact",
        lambda disease_code: SimpleNamespace(feature_columns=["나이", "BMI"]),
    )

    def fake_screening_dual_stage(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            disease_code=kwargs["disease_code"],
            base_risk_level=kwargs["base_risk_level"],
            base_high=True,
            base_caution_or_above=True,
            screening_high=False,
            risk_level=RiskLevel.CAUTION.value,
            service_band=SimpleNamespace(value=RiskLevel.CAUTION.value),
            service_band_label="주의",
            service_band_percent=65,
            legacy_risk_level=RiskLevel.CAUTION.value,
            screening_missing_features=[],
            screening_neutralized_features=[],
            screening_model_count=5,
        )

    monkeypatch.setattr(analysis_service, "predict_screening_dual_stage_risk", fake_screening_dual_stage)
    monkeypatch.setattr(
        analysis_service,
        "_predict_basic_strict_signal",
        lambda **kwargs: {
            "status": "fallback_screening_v1",
            "fallback_reason": "RuntimeError",
            "strict_model_count": 0,
        },
    )

    result = analysis_service._predict_basic_screening_dual_stage(
        user=SimpleNamespace(id=7, birthday=date(1975, 1, 1), gender="MALE"),
        health_record=_health_record(),
        analysis_type=AnalysisType.HYPERTENSION,
        base_risk_level=RiskLevel.CAUTION,
        analysis_mode=AnalysisMode.BASIC,
    )

    assert result is not None
    assert result["policy_version"] == "v2_strict_fallback_v1"
    assert result["strict_status"] == "fallback_screening_v1"
    assert result["strict_fallback_reason"] == "RuntimeError"
    assert result["risk_level"] == RiskLevel.CAUTION.value
    assert result["service_band"] == RiskLevel.CAUTION.value


def test_screening_fallback_snapshot_metadata_does_not_change_risk_level() -> None:
    snapshot = analysis_service._analysis_snapshot_request(
        analysis_type=AnalysisType.HYPERTENSION,
        analysis_mode=AnalysisMode.BASIC,
        health_record=_health_record(),
        score=Decimal("0.52"),
        risk_level=RiskLevel.CAUTION,
        guide_message="간편 분석 참고용입니다.",
        factors=[],
        screening_dual_stage={
            "status": "fallback_basic_only",
            "disease_code": "HTN",
            "base_risk_level": "CAUTION",
            "fallback_reason": "RuntimeError",
        },
    )

    final_outputs = snapshot.output_payload["final_outputs"]
    assert final_outputs["risk_level"] == RiskLevel.CAUTION.value
    assert final_outputs["screening_dual_stage_status"] == "fallback_basic_only"
    assert final_outputs["screening_dual_stage_fallback_reason"] == "RuntimeError"
    assert final_outputs["service_band"] == RiskLevel.CAUTION.value
    assert final_outputs["service_band_label"] == "주의"
    assert final_outputs["service_band_percent"] == 65


def test_strict_dual_stage_snapshot_metadata_excludes_raw_probability() -> None:
    snapshot = analysis_service._analysis_snapshot_request(
        analysis_type=AnalysisType.HYPERTENSION,
        analysis_mode=AnalysisMode.BASIC,
        health_record=_health_record(),
        score=Decimal("0.52"),
        risk_level=RiskLevel.HIGH_CAUTION,
        guide_message="간편 분석 참고용입니다.",
        factors=[],
        screening_dual_stage={
            "status": "applied",
            "disease_code": "HTN",
            "policy_version": "v2",
            "base_risk_level": "CAUTION",
            "screening_high": False,
            "strict_high": True,
            "screening_model_count": 5,
            "strict_model_count": 5,
            "risk_level": "HIGH_CAUTION",
            "service_band": "HIGH_CAUTION",
            "service_band_label": "높은 주의",
            "service_band_percent": 80,
        },
    )

    final_outputs = snapshot.output_payload["final_outputs"]
    assert final_outputs["dual_stage_policy_version"] == "v2"
    assert final_outputs["screening_high"] is False
    assert final_outputs["strict_high"] is True
    assert final_outputs["screening_model_count"] == 5
    assert final_outputs["strict_model_count"] == 5
    assert "probability" not in str(snapshot.output_payload)
    assert "probability" not in str(snapshot.model_payload)


async def _empty_recommendations(user_id: int, result: Any) -> list[int]:
    _ = user_id, result
    return []
