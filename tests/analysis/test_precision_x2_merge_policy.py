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
        "family_htn": "YES",
        "family_dyslipidemia": "YES",
        "occupation_code": "PROFESSIONAL",
        "bmi": Decimal("26.0"),
        "waist_cm": None,
        "fasting_glucose": 108,
        "hba1c": None,
        "ldl_cholesterol": None,
        "hdl_cholesterol": None,
        "triglyceride": None,
        "total_cholesterol": None,
        "systolic_bp": 145,
        "diastolic_bp": 92,
        "smoking_status": "NEVER",
        "drinking_frequency": "NONE",
        "drinking_amount": "NONE",
        "walking_days_per_week": 5,
        "strength_days_per_week": 2,
        "hemoglobin": Decimal("11.5"),
        "egfr": Decimal("35"),
        "ast": None,
        "alt": None,
        "gamma_gtp": None,
        "creatinine": None,
        "urine_protein": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_basic_mode_keeps_four_diseases_and_does_not_call_x2(monkeypatch: pytest.MonkeyPatch) -> None:
    created, _snapshots = _patch_analysis_writes(monkeypatch)
    monkeypatch.setattr(analysis_service, "_predict_basic_screening_dual_stage", lambda **kwargs: None)

    def fail_if_x2_is_called(*args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        raise AssertionError("BASIC mode must not call X2 mapper")

    monkeypatch.setattr(analysis_service, "map_x2_stage_to_risk_level", fail_if_x2_is_called)

    results = await analysis_service.run_analysis(7, _health_record(), AnalysisMode.BASIC)

    assert [item["analysis_type"] for item in results] == [
        AnalysisType.DIABETES,
        AnalysisType.OBESITY,
        AnalysisType.DYSLIPIDEMIA,
        AnalysisType.HYPERTENSION,
    ]
    assert [item.analysis_type for item in created] == [
        AnalysisType.DIABETES,
        AnalysisType.OBESITY,
        AnalysisType.DYSLIPIDEMIA,
        AnalysisType.HYPERTENSION,
    ]


@pytest.mark.asyncio
async def test_precision_mode_merges_basic_fallback_and_available_x2_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created, snapshots = _patch_analysis_writes(monkeypatch)
    monkeypatch.setattr(analysis_service, "_predict_basic_screening_dual_stage", lambda **kwargs: None)

    results = await analysis_service.run_analysis(7, _health_record(), AnalysisMode.PRECISION)

    by_type = {item.analysis_type: item for item in created}
    assert set(by_type) == {
        AnalysisType.DIABETES,
        AnalysisType.OBESITY,
        AnalysisType.DYSLIPIDEMIA,
        AnalysisType.HYPERTENSION,
        AnalysisType.ANEMIA,
        AnalysisType.KIDNEY_FUNCTION,
        AnalysisType.CHRONIC_KIDNEY_DISEASE,
    }
    assert len(results) == 7
    assert all(item.analysis_mode == AnalysisMode.PRECISION for item in created)

    assert by_type[AnalysisType.HYPERTENSION].risk_level == RiskLevel.CAUTION
    assert by_type[AnalysisType.HYPERTENSION].model_name == "x2_rule"
    assert by_type[AnalysisType.DIABETES].risk_level == RiskLevel.ATTENTION
    assert by_type[AnalysisType.OBESITY].risk_level == RiskLevel.ATTENTION
    assert by_type[AnalysisType.ANEMIA].risk_level == RiskLevel.ATTENTION
    assert by_type[AnalysisType.KIDNEY_FUNCTION].risk_level == RiskLevel.ATTENTION
    assert by_type[AnalysisType.CHRONIC_KIDNEY_DISEASE].risk_level == RiskLevel.CAUTION

    dyslipidemia = by_type[AnalysisType.DYSLIPIDEMIA]
    assert dyslipidemia.model_name == "rule_based"
    assert dyslipidemia.model_version == "web-basic-fallback-v1"
    assert dyslipidemia.risk_level in RiskLevel

    snapshot_by_type = {snapshot.input_payload["analysis_type"]: snapshot for snapshot in snapshots}
    htn_final = snapshot_by_type[AnalysisType.HYPERTENSION.value].output_payload["final_outputs"]
    assert htn_final["result_source"] == "X2_RULE"
    assert htn_final["x2_available"] is True
    assert htn_final["x2_stage_code"] == "HTN_STAGE_1"
    assert htn_final["x2_stage_label"] == "고혈압 1단계 범위"
    assert htn_final["x2_missing_fields"] == []

    dl_final = snapshot_by_type[AnalysisType.DYSLIPIDEMIA.value].output_payload["final_outputs"]
    assert dl_final["result_source"] == "BASIC_FALLBACK"
    assert dl_final["x2_available"] is False
    assert dl_final["x2_missing_fields"] == [
        "total_cholesterol",
        "ldl_cholesterol",
        "hdl_cholesterol",
        "triglyceride",
    ]

    anem_model_payload = snapshot_by_type[AnalysisType.ANEMIA.value].model_payload
    assert anem_model_payload["x2_rule"]["result_source"] == "X2_RULE"
    assert anem_model_payload["x2_rule"]["x2_stage_code"] == "MILD_RANGE"


@pytest.mark.asyncio
async def test_precision_x2_only_unavailable_diseases_are_not_created(monkeypatch: pytest.MonkeyPatch) -> None:
    created, _snapshots = _patch_analysis_writes(monkeypatch)
    monkeypatch.setattr(analysis_service, "_predict_basic_screening_dual_stage", lambda **kwargs: None)

    await analysis_service.run_analysis(
        7,
        _health_record(hemoglobin=None, egfr=None, ast=None, alt=None, gamma_gtp=None, creatinine=None),
        AnalysisMode.PRECISION,
    )

    assert AnalysisType.ANEMIA not in {item.analysis_type for item in created}
    assert AnalysisType.CHRONIC_KIDNEY_DISEASE not in {item.analysis_type for item in created}
    assert AnalysisType.FATTY_LIVER not in {item.analysis_type for item in created}
    assert AnalysisType.LIVER_FUNCTION not in {item.analysis_type for item in created}
    assert AnalysisType.KIDNEY_FUNCTION not in {item.analysis_type for item in created}


@pytest.mark.asyncio
async def test_precision_missing_fields_requires_only_basic_minimum_inputs() -> None:
    missing_fields = await analysis_service.get_missing_fields_for_mode(
        SimpleNamespace(gender="MALE", birthday=date(1990, 1, 1)),
        _health_record(
            systolic_bp=None,
            diastolic_bp=None,
            fasting_glucose=None,
            total_cholesterol=None,
            ldl_cholesterol=None,
            hdl_cholesterol=None,
            triglyceride=None,
            waist_cm=None,
        ),
        AnalysisMode.PRECISION,
    )

    assert missing_fields == []


def _patch_analysis_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[SimpleNamespace], list[Any]]:
    created: list[SimpleNamespace] = []
    snapshots: list[Any] = []

    async def fake_get_user(**kwargs: Any) -> SimpleNamespace:
        assert kwargs == {"id": 7}
        return SimpleNamespace(id=7, birthday=date(1975, 1, 1), gender="MALE")

    async def fake_create_analysis_result(user_id: int, request: Any) -> SimpleNamespace:
        result = SimpleNamespace(
            id=len(created) + 1,
            user_id=user_id,
            health_record_id=request.health_record_id,
            async_job_id=request.async_job_id,
            analysis_type=request.analysis_type,
            analysis_mode=request.analysis_mode,
            risk_score=request.risk_score,
            risk_level=request.risk_level,
            summary=request.summary,
            model_name=request.model_name,
            model_version=request.model_version,
        )
        created.append(result)
        return result

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
        _ = analysis_result_id
        snapshots.append(request)
        return SimpleNamespace(id=len(snapshots))

    monkeypatch.setattr(analysis_service.User, "get_or_none", fake_get_user)
    monkeypatch.setattr(analysis_service, "create_analysis_result", fake_create_analysis_result)
    monkeypatch.setattr(analysis_service, "create_analysis_factors", fake_create_analysis_factors)
    monkeypatch.setattr(analysis_service, "create_analysis_snapshot", fake_create_analysis_snapshot)
    monkeypatch.setattr(analysis_service, "_create_challenge_recommendations", _empty_recommendations)
    monkeypatch.setattr(analysis_service, "_analysis_explanation", lambda result, factors: {})
    return created, snapshots


async def _empty_recommendations(user_id: int, result: Any) -> list[int]:
    _ = user_id, result
    return []
