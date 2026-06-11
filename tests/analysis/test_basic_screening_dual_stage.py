from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

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


async def _empty_recommendations(user_id: int, result: Any) -> list[int]:
    _ = user_id, result
    return []
