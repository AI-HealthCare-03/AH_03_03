from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from app.models.analysis import AnalysisMode, AnalysisType, RiskLevel
from app.services import analysis as analysis_service


def _user(**overrides: Any) -> SimpleNamespace:
    values = {
        "id": 7,
        "birthday": date(1990, 1, 1),
        "gender": "MALE",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _record(**overrides: Any) -> SimpleNamespace:
    values = {
        "id": 101,
        "height_cm": Decimal("172.0"),
        "weight_kg": Decimal("68.0"),
        "bmi": Decimal("22.9"),
        "waist_cm": Decimal("78.0"),
        "systolic_bp": 115,
        "diastolic_bp": 75,
        "fasting_glucose": 90,
        "hba1c": Decimal("5.4"),
        "total_cholesterol": 170,
        "ldl_cholesterol": 90,
        "hdl_cholesterol": 60,
        "triglyceride": 100,
        "family_htn": "NO",
        "family_dm": "NO",
        "family_dyslipidemia": "NO",
        "smoking_status": "NEVER",
        "drinking_frequency": "NONE",
        "drinking_amount": "NONE",
        "walking_days_per_week": 5,
        "strength_days_per_week": 2,
        "occupation_code": "PROFESSIONAL",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("systolic_bp", "diastolic_bp", "expected"),
    [
        (115, 75, RiskLevel.LOW),
        (130, 85, RiskLevel.CAUTION),
        (139, 89, RiskLevel.CAUTION),
        (140, 90, RiskLevel.HIGH_CAUTION),
        (160, 100, RiskLevel.HIGH_CAUTION),
    ],
)
def test_hypertension_medical_boundary_rule_labels(
    systolic_bp: int,
    diastolic_bp: int,
    expected: RiskLevel,
) -> None:
    record = _record(systolic_bp=systolic_bp, diastolic_bp=diastolic_bp)
    score = analysis_service._hypertension_score(record)

    assert analysis_service._risk_level_for_analysis_score(AnalysisType.HYPERTENSION, score, record) == expected, (
        f"HTN medical boundary mismatch for SBP/DBP={systolic_bp}/{diastolic_bp}; score={score}"
    )


@pytest.mark.parametrize(
    ("fasting_glucose", "hba1c", "expected"),
    [
        (90, Decimal("5.4"), RiskLevel.LOW),
        (105, Decimal("5.7"), RiskLevel.CAUTION),
        (120, Decimal("6.2"), RiskLevel.CAUTION),
        (126, Decimal("6.5"), RiskLevel.HIGH_CAUTION),
        (150, Decimal("7.2"), RiskLevel.HIGH_CAUTION),
    ],
)
def test_diabetes_medical_boundary_rule_labels(
    fasting_glucose: int,
    hba1c: Decimal,
    expected: RiskLevel,
) -> None:
    record = _record(fasting_glucose=fasting_glucose, hba1c=hba1c)
    score = analysis_service._diabetes_score(record)

    assert analysis_service._risk_level_for_analysis_score(AnalysisType.DIABETES, score, record) == expected, (
        f"DM medical boundary mismatch for fasting_glucose={fasting_glucose}, HbA1c={hba1c}; score={score}"
    )


@pytest.mark.parametrize(
    ("total_cholesterol", "ldl_cholesterol", "hdl_cholesterol", "triglyceride", "expected"),
    [
        (170, 90, 60, 100, RiskLevel.LOW),
        (220, 145, 50, 130, RiskLevel.CAUTION),
        (190, 110, 45, 180, RiskLevel.CAUTION),
        (250, 170, 35, 220, RiskLevel.HIGH_CAUTION),
    ],
)
def test_dyslipidemia_medical_boundary_rule_labels(
    total_cholesterol: int,
    ldl_cholesterol: int,
    hdl_cholesterol: int,
    triglyceride: int,
    expected: RiskLevel,
) -> None:
    record = _record(
        total_cholesterol=total_cholesterol,
        ldl_cholesterol=ldl_cholesterol,
        hdl_cholesterol=hdl_cholesterol,
        triglyceride=triglyceride,
    )
    score = analysis_service._dyslipidemia_score(record)

    assert analysis_service._risk_level_for_analysis_score(AnalysisType.DYSLIPIDEMIA, score, record) == expected, (
        "DL medical boundary mismatch for "
        f"TC/LDL/HDL/TG={total_cholesterol}/{ldl_cholesterol}/{hdl_cholesterol}/{triglyceride}; score={score}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Current DL rule does not promote TC 205, LDL 120, HDL 55, TG 140. "
        "If product policy treats this as caution, add a medical guardrail."
    ),
)
def test_dyslipidemia_borderline_values_should_not_remain_low_if_policy_requires_caution() -> None:
    record = _record(total_cholesterol=205, ldl_cholesterol=120, hdl_cholesterol=55, triglyceride=140)
    score = analysis_service._dyslipidemia_score(record)

    assert (
        analysis_service._risk_level_for_analysis_score(AnalysisType.DYSLIPIDEMIA, score, record) == RiskLevel.CAUTION
    ), (
        "DL borderline values remained LOW. If TC>=200 or LDL>=120 should be user-visible caution, "
        "BASIC/PRECISION medical guardrail needs to encode that policy."
    )


def _patch_v2_policy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    screening_high: bool,
    screening_risk_level: RiskLevel,
    strict_high: bool,
) -> None:
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
            base_high=kwargs["base_risk_level"] in {RiskLevel.CAUTION, RiskLevel.HIGH_CAUTION},
            base_caution_or_above=kwargs["base_risk_level"] in {RiskLevel.CAUTION, RiskLevel.HIGH_CAUTION},
            screening_high=screening_high,
            risk_level=screening_risk_level.value,
            service_band=SimpleNamespace(value=screening_risk_level.value),
            service_band_label=analysis_service._risk_level_label(screening_risk_level),
            service_band_percent=analysis_service._risk_level_display_percent(screening_risk_level),
            legacy_risk_level=screening_risk_level.value,
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
            "strict_high": strict_high,
            "strict_model_count": 5,
            "strict_model_name": "catboost",
            "strict_model_version": "medical_alignment_test",
        },
    )


def _predict_v2(
    *,
    base_risk_level: RiskLevel,
    analysis_type: AnalysisType = AnalysisType.HYPERTENSION,
) -> dict[str, Any]:
    result = analysis_service._predict_basic_screening_dual_stage(
        user=_user(),
        health_record=_record(),
        analysis_type=analysis_type,
        base_risk_level=base_risk_level,
        analysis_mode=AnalysisMode.BASIC,
    )
    assert result is not None
    return result


def test_v2_low_with_screening_and_strict_high_is_capped_at_caution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_v2_policy(
        monkeypatch,
        screening_high=True,
        screening_risk_level=RiskLevel.CAUTION,
        strict_high=True,
    )

    result = _predict_v2(base_risk_level=RiskLevel.LOW, analysis_type=AnalysisType.DIABETES)

    assert result["risk_level"] == RiskLevel.CAUTION.value
    assert result["service_band_label"] == "주의"
    assert result["service_band_percent"] == 65
    assert result["screening_high"] is True
    assert result["strict_high"] is True
    assert "probability" not in str(result)


def test_v2_caution_with_strict_high_promotes_to_high_caution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_v2_policy(
        monkeypatch,
        screening_high=False,
        screening_risk_level=RiskLevel.CAUTION,
        strict_high=True,
    )

    result = _predict_v2(base_risk_level=RiskLevel.CAUTION)

    assert result["risk_level"] == RiskLevel.HIGH_CAUTION.value
    assert result["service_band_label"] == "높은 주의"
    assert result["service_band_percent"] == 80
    assert "probability" not in str(result)


def test_v2_high_caution_is_preserved_even_when_model_signals_are_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_v2_policy(
        monkeypatch,
        screening_high=False,
        screening_risk_level=RiskLevel.HIGH_CAUTION,
        strict_high=False,
    )

    result = _predict_v2(base_risk_level=RiskLevel.HIGH_CAUTION)

    assert result["risk_level"] == RiskLevel.HIGH_CAUTION.value
    assert result["service_band_label"] == "높은 주의"
    assert result["service_band_percent"] == 80
    assert "probability" not in str(result)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BASIC _calculate_analysis_scores currently uses X1 lifestyle/family/BMI scores only. "
        "Direct clinical measurements are not merged into BASIC final labels yet."
    ),
)
def test_basic_full_flow_high_blood_pressure_should_not_remain_low_without_medical_guardrail() -> None:
    record = _record(systolic_bp=160, diastolic_bp=100)
    scores = analysis_service._calculate_analysis_scores(record, AnalysisMode.BASIC, _user())
    risk_level = analysis_service._risk_level_for_analysis_score(
        AnalysisType.HYPERTENSION,
        scores[AnalysisType.HYPERTENSION],
        record,
    )

    assert risk_level == RiskLevel.HIGH_CAUTION, (
        "BASIC final HTN label stayed below HIGH_CAUTION despite SBP/DBP=160/100. "
        "A medical guardrail should merge direct BP thresholds before user-visible labels are finalized."
    )
