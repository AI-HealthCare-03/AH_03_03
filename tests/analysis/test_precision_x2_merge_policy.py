from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from app.models.analysis import AnalysisMode, AnalysisType, RiskLevel
from app.services import analysis as analysis_service

_DEFAULT_EXAM_REPORT = object()


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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_exam_measurements_to_x2_payload_normalizes_aliases_and_values() -> None:
    payload = analysis_service.exam_measurements_to_x2_payload(
        [
            _measurement("hb", "13.2 g/dL"),
            _measurement("AST", "30 IU/L"),
            _measurement("ALT", "25 IU/L"),
            _measurement("γ-GTP", "44 IU/L"),
            _measurement("creatinine", "0.9 mg/dL"),
            _measurement("eGFR", "45 mL/min/1.73m2"),
            _measurement("요단백", "+1"),
            _measurement("LDL", "132 mg/dL"),
            _measurement("HbA1c", "5.8 %"),
        ]
    )

    assert payload == {
        "hemoglobin": Decimal("13.2"),
        "ast": 30,
        "alt": 25,
        "gamma_gtp": 44,
        "creatinine": Decimal("0.9"),
        "egfr": Decimal("45"),
        "urine_protein": "plus_1",
        "ldl_cholesterol": 132,
        "hba1c": Decimal("5.8"),
    }


@pytest.mark.asyncio
async def test_precision_input_payload_includes_confirmed_x2_only_exam_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_confirmed_exam_measurements(
        monkeypatch,
        [
            _measurement("hemoglobin", "13.2 g/dL"),
            _measurement("ast", "30 U/L"),
            _measurement("alt", "25 U/L"),
            _measurement("gamma_gtp", "44 U/L"),
            _measurement("creatinine", "0.9 mg/dL"),
            _measurement("egfr", "82 mL/min/1.73m2"),
        ],
    )

    payload = await analysis_service.build_precision_analysis_input_payload(
        user=SimpleNamespace(id=7, gender="MALE"),
        health_record=_health_record(),
    )

    assert payload["selected_exam_report_id"] == 9001
    assert payload["x2_measurement_source"] == "exam_measurements"
    assert payload["x2_input_payload"]["hemoglobin"] == Decimal("13.2")
    assert payload["x2_input_payload"]["ast"] == 30
    assert payload["x2_input_payload"]["alt"] == 25
    assert payload["x2_input_payload"]["gamma_gtp"] == 44
    assert payload["x2_input_payload"]["creatinine"] == Decimal("0.9")
    assert payload["x2_input_payload"]["egfr"] == Decimal("82")
    assert payload["x2_field_sources"]["hemoglobin"] == "exam_measurements"


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
    _patch_confirmed_exam_measurements(
        monkeypatch,
        [
            _measurement("systolic_bp", "145 mmHg"),
            _measurement("diastolic_bp", "92 mmHg"),
            _measurement("fasting_glucose", "108 mg/dL"),
            _measurement("hemoglobin", "11.5 g/dL"),
            _measurement("egfr", "35 mL/min/1.73m2"),
        ],
    )

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
    assert by_type[AnalysisType.OBESITY].risk_level == RiskLevel.CAUTION
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
    assert snapshot_by_type[AnalysisType.HYPERTENSION.value].input_payload["selected_exam_report_id"] == 9001
    assert snapshot_by_type[AnalysisType.HYPERTENSION.value].input_payload["x2_measurement_source"] == (
        "exam_measurements"
    )
    assert snapshot_by_type[AnalysisType.HYPERTENSION.value].input_payload["x2_input_payload"]["hemoglobin"] == 11.5

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
    _patch_confirmed_exam_measurements(monkeypatch, [])

    await analysis_service.run_analysis(
        7,
        _health_record(),
        AnalysisMode.PRECISION,
    )

    assert AnalysisType.ANEMIA not in {item.analysis_type for item in created}
    assert AnalysisType.CHRONIC_KIDNEY_DISEASE not in {item.analysis_type for item in created}
    assert AnalysisType.FATTY_LIVER not in {item.analysis_type for item in created}
    assert AnalysisType.LIVER_FUNCTION not in {item.analysis_type for item in created}
    assert AnalysisType.KIDNEY_FUNCTION not in {item.analysis_type for item in created}


@pytest.mark.asyncio
async def test_precision_uses_exam_measurements_for_x2_only_diseases(monkeypatch: pytest.MonkeyPatch) -> None:
    created, _snapshots = _patch_analysis_writes(monkeypatch)
    monkeypatch.setattr(analysis_service, "_predict_basic_screening_dual_stage", lambda **kwargs: None)
    _patch_confirmed_exam_measurements(
        monkeypatch,
        [
            _measurement("AST", "30 IU/L"),
            _measurement("ALT", "25 IU/L"),
        ],
    )

    await analysis_service.run_analysis(7, _health_record(bmi=Decimal("24.0")), AnalysisMode.PRECISION)

    by_type = {item.analysis_type: item for item in created}
    assert AnalysisType.FATTY_LIVER in by_type
    assert by_type[AnalysisType.FATTY_LIVER].model_name == "x2_rule"


@pytest.mark.asyncio
async def test_precision_prefers_confirmed_exam_measurements_over_health_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created, _snapshots = _patch_analysis_writes(monkeypatch)
    monkeypatch.setattr(analysis_service, "_predict_basic_screening_dual_stage", lambda **kwargs: None)
    _patch_confirmed_exam_measurements(
        monkeypatch,
        [
            _measurement("systolic_bp", "161 mmHg"),
            _measurement("diastolic_bp", "88 mmHg"),
        ],
    )

    await analysis_service.run_analysis(
        7,
        _health_record(systolic_bp=118, diastolic_bp=76),
        AnalysisMode.PRECISION,
    )

    by_type = {item.analysis_type: item for item in created}
    assert by_type[AnalysisType.HYPERTENSION].risk_level == RiskLevel.HIGH_CAUTION


@pytest.mark.asyncio
async def test_precision_obesity_prefers_confirmed_exam_bmi_over_health_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created, _snapshots = _patch_analysis_writes(monkeypatch)
    monkeypatch.setattr(analysis_service, "_predict_basic_screening_dual_stage", lambda **kwargs: None)
    _patch_confirmed_exam_measurements(monkeypatch, [_measurement("bmi", "30.0 kg/m2")])

    await analysis_service.run_analysis(
        7,
        _health_record(bmi=Decimal("22.0"), height_cm=Decimal("172.0"), weight_kg=Decimal("65.0")),
        AnalysisMode.PRECISION,
    )

    by_type = {item.analysis_type: item for item in created}
    assert by_type[AnalysisType.OBESITY].model_name == "x2_rule"
    assert by_type[AnalysisType.OBESITY].risk_level == RiskLevel.HIGH_CAUTION


@pytest.mark.asyncio
async def test_precision_keeps_obesity_and_abdominal_obesity_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created, _snapshots = _patch_analysis_writes(monkeypatch)
    monkeypatch.setattr(analysis_service, "_predict_basic_screening_dual_stage", lambda **kwargs: None)
    _patch_confirmed_exam_measurements(
        monkeypatch,
        [
            _measurement("bmi", "26.0"),
            _measurement("waist_cm", "96 cm"),
        ],
    )

    await analysis_service.run_analysis(7, _health_record(), AnalysisMode.PRECISION)

    by_type = {item.analysis_type: item for item in created}
    assert by_type[AnalysisType.OBESITY].risk_level == RiskLevel.CAUTION
    assert by_type[AnalysisType.ABDOMINAL_OBESITY].risk_level == RiskLevel.CAUTION
    assert by_type[AnalysisType.OBESITY].analysis_type != by_type[AnalysisType.ABDOMINAL_OBESITY].analysis_type


@pytest.mark.asyncio
async def test_precision_without_exam_measurements_uses_health_record_fallback_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created, snapshots = _patch_analysis_writes(monkeypatch)
    monkeypatch.setattr(analysis_service, "_predict_basic_screening_dual_stage", lambda **kwargs: None)
    _patch_confirmed_exam_measurements(monkeypatch, [], report=None)

    await analysis_service.run_analysis(7, _health_record(), AnalysisMode.PRECISION)

    assert AnalysisType.ANEMIA not in {item.analysis_type for item in created}
    snapshot_by_type = {snapshot.input_payload["analysis_type"]: snapshot for snapshot in snapshots}
    assert snapshot_by_type[AnalysisType.HYPERTENSION.value].input_payload["selected_exam_report_id"] is None
    assert snapshot_by_type[AnalysisType.HYPERTENSION.value].input_payload["x2_measurement_source"] == (
        "health_record_fallback"
    )


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


def _patch_confirmed_exam_measurements(
    monkeypatch: pytest.MonkeyPatch,
    measurements: list[SimpleNamespace],
    report: SimpleNamespace | None | object = _DEFAULT_EXAM_REPORT,
) -> None:
    selected_report = SimpleNamespace(id=9001) if report is _DEFAULT_EXAM_REPORT else report

    async def fake_get_latest_confirmed_exam_measurements_for_analysis(user_id: int):
        assert user_id == 7
        return selected_report, measurements

    monkeypatch.setattr(
        analysis_service.exam_service,
        "get_latest_confirmed_exam_measurements_for_analysis",
        fake_get_latest_confirmed_exam_measurements_for_analysis,
    )


def _measurement(key: str, value: object) -> SimpleNamespace:
    return SimpleNamespace(measurement_key=key, measurement_name=key, value=value)


async def _empty_recommendations(user_id: int, result: Any) -> list[int]:
    _ = user_id, result
    return []
