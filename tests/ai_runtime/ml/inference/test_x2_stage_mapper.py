from __future__ import annotations

import pytest

from ai_runtime.ml.inference.x2_stage_mapper import (
    X2ResultSource,
    map_x2_stage_to_risk_level,
)


@pytest.mark.parametrize(
    ("analysis_type", "features", "expected_analysis_type", "expected_risk_level", "expected_stage_code"),
    [
        ("HTN", {"systolic_bp": 118, "diastolic_bp": 76}, "HYPERTENSION", "LOW", "NORMAL"),
        ("HTN", {"systolic_bp": 124, "diastolic_bp": 78}, "HYPERTENSION", "ATTENTION", "ELEVATED"),
        ("HTN", {"systolic_bp": 132, "diastolic_bp": 78}, "HYPERTENSION", "ATTENTION", "HTN_PRE_STAGE"),
        ("HTN", {"systolic_bp": 145, "diastolic_bp": 88}, "HYPERTENSION", "CAUTION", "HTN_STAGE_1"),
        ("HTN", {"systolic_bp": 161, "diastolic_bp": 88}, "HYPERTENSION", "HIGH_CAUTION", "HTN_STAGE_2"),
        ("DM", {"fasting_glucose": 91}, "DIABETES", "LOW", "NORMAL"),
        ("DM", {"fasting_glucose": 102}, "DIABETES", "ATTENTION", "PRE_DIABETES_RANGE"),
        ("DM", {"fasting_glucose": 126}, "DIABETES", "CAUTION", "DIABETES_RANGE"),
        ("DL", {"total_cholesterol": 185}, "DYSLIPIDEMIA", "LOW", "NORMAL"),
        ("DL", {"total_cholesterol": 201}, "DYSLIPIDEMIA", "ATTENTION", "BORDERLINE_RANGE"),
        ("DL", {"ldl_cholesterol": 190}, "DYSLIPIDEMIA", "HIGH_CAUTION", "HIGH_RISK_RANGE"),
        ("OBE", {"bmi": 20.1}, "OBESITY", "LOW", "NORMAL"),
        ("OBE", {"bmi": 23.5}, "OBESITY", "ATTENTION", "PRE_OBESITY"),
        ("OBE", {"bmi": 30.1}, "OBESITY", "HIGH_CAUTION", "OBESITY_STAGE_2"),
        ("OBE", {"bmi": 35}, "OBESITY", "HIGH_CAUTION", "OBESITY_STAGE_3"),
        ("ABO", {"waist_cm": 82, "sex": "FEMALE"}, "ABDOMINAL_OBESITY", "LOW", "LOW_RISK_RANGE"),
        ("ABO", {"waist_cm": 88, "sex": "FEMALE"}, "ABDOMINAL_OBESITY", "ATTENTION", "RISK_RANGE"),
        ("ABO", {"waist_cm": 96, "sex": "MALE"}, "ABDOMINAL_OBESITY", "CAUTION", "HIGH_RISK_RANGE"),
        ("FL", {"ast": 30, "alt": 20, "bmi": 22, "sex": "MALE"}, "FATTY_LIVER", "LOW", "HSI_LOW_RANGE"),
        ("FL", {"ast": 30, "alt": 25, "bmi": 24, "sex": "MALE"}, "FATTY_LIVER", "ATTENTION", "HSI_RISK_RANGE"),
        ("FL", {"ast": 20, "alt": 45, "bmi": 21, "sex": "FEMALE"}, "FATTY_LIVER", "CAUTION", "HSI_HIGH_RANGE"),
        ("ANEM", {"hemoglobin": 13.1, "sex": "MALE"}, "ANEMIA", "LOW", "NORMAL"),
        ("ANEM", {"hemoglobin": 11.3, "sex": "FEMALE"}, "ANEMIA", "ATTENTION", "MILD_RANGE"),
        ("ANEM", {"hemoglobin": 8.1, "sex": "MALE"}, "ANEMIA", "CAUTION", "MODERATE_RANGE"),
        ("ANEM", {"hemoglobin": 7.9, "sex": "FEMALE"}, "ANEMIA", "HIGH_CAUTION", "SEVERE_RANGE"),
        ("LF", {"ast": 30}, "LIVER_FUNCTION", "LOW", "NORMAL"),
        ("LF", {"alt": 41}, "LIVER_FUNCTION", "ATTENTION", "ABNORMAL_SUSPECTED"),
        ("KF", {"creatinine": 1.0}, "KIDNEY_FUNCTION", "LOW", "NORMAL"),
        ("KF", {"urine_protein": "+1"}, "KIDNEY_FUNCTION", "ATTENTION", "DAMAGE_SUSPECTED"),
        ("CKD", {"egfr": 92}, "CHRONIC_KIDNEY_DISEASE", "LOW", "EGFR_90_PLUS"),
        ("CKD", {"egfr": 58}, "CHRONIC_KIDNEY_DISEASE", "ATTENTION", "EGFR_45_59"),
        ("CKD", {"egfr": 35}, "CHRONIC_KIDNEY_DISEASE", "CAUTION", "EGFR_30_44"),
        ("CKD", {"egfr": 29}, "CHRONIC_KIDNEY_DISEASE", "HIGH_CAUTION", "EGFR_UNDER_30"),
    ],
)
def test_map_x2_stage_to_risk_level_supported_diseases(
    analysis_type: str,
    features: dict[str, object],
    expected_analysis_type: str,
    expected_risk_level: str,
    expected_stage_code: str,
) -> None:
    result = map_x2_stage_to_risk_level(analysis_type, features)

    assert result.analysis_type == expected_analysis_type
    assert result.risk_level == expected_risk_level
    assert result.result_source == X2ResultSource.X2_RULE
    assert result.x2_stage_code == expected_stage_code
    assert result.x2_available is True
    assert result.x2_missing_fields == []


@pytest.mark.parametrize(
    ("analysis_type", "features", "missing_fields"),
    [
        ("HTN", {}, ["systolic_bp", "diastolic_bp"]),
        ("DM", {}, ["fasting_glucose", "hba1c"]),
        ("DL", {}, ["total_cholesterol", "ldl_cholesterol", "hdl_cholesterol", "triglyceride"]),
        ("OBE", {}, ["bmi", "height_cm", "weight_kg"]),
        ("ABO", {"waist_cm": 91}, ["sex"]),
        ("FL", {"ast": 0, "alt": 40, "bmi": 27, "sex": "MALE"}, ["ast"]),
        ("ANEM", {"hemoglobin": 12}, ["sex"]),
        ("LF", {}, ["ast", "alt", "gamma_gtp"]),
        ("LF", {"gamma_gtp": 40}, ["sex"]),
        ("KF", {}, ["urine_protein", "creatinine", "egfr"]),
        ("CKD", {}, ["egfr"]),
    ],
)
def test_missing_values_return_x2_unavailable(
    analysis_type: str,
    features: dict[str, object],
    missing_fields: list[str],
) -> None:
    result = map_x2_stage_to_risk_level(analysis_type, features)

    assert result.analysis_type is not None
    assert result.risk_level is None
    assert result.result_source == X2ResultSource.X2_UNAVAILABLE
    assert result.x2_stage_code is None
    assert result.x2_stage_label is None
    assert result.x2_available is False
    assert result.x2_missing_fields == missing_fields


def test_dl_accepts_one_lipid_value_without_forcing_missing_fields() -> None:
    result = map_x2_stage_to_risk_level("DYSLIPIDEMIA", {"triglyceride": 151})

    assert result.risk_level == "ATTENTION"
    assert result.result_source == X2ResultSource.X2_RULE
    assert result.x2_missing_fields == []


def test_dl_hdl_only_requires_sex_because_hdl_cutoff_is_sex_specific() -> None:
    result = map_x2_stage_to_risk_level("DL", {"hdl_cholesterol": 39})

    assert result.risk_level is None
    assert result.result_source == X2ResultSource.X2_UNAVAILABLE
    assert result.x2_missing_fields == ["sex"]


def test_dl_hdl_is_ignored_without_sex_when_other_lipid_values_are_available() -> None:
    result = map_x2_stage_to_risk_level("DL", {"total_cholesterol": 180, "hdl_cholesterol": 39})

    assert result.risk_level == "LOW"
    assert result.result_source == X2ResultSource.X2_RULE
    assert result.x2_missing_fields == []


def test_dm_accepts_hba1c_without_fasting_glucose() -> None:
    result = map_x2_stage_to_risk_level("DM", {"hba1c": 5.8})

    assert result.risk_level == "ATTENTION"
    assert result.x2_stage_code == "PRE_DIABETES_RANGE"


@pytest.mark.parametrize(
    ("analysis_type", "features"),
    [
        ("DM", {"fasting_glucose": 180}),
        ("DL", {"ldl_cholesterol": 190}),
        ("ABO", {"waist_cm": 120, "sex": "MALE"}),
        ("FL", {"ast": 20, "alt": 50, "bmi": 25, "sex": "FEMALE"}),
        ("LF", {"ast": 100}),
        ("KF", {"egfr": 40}),
    ],
)
def test_diseases_without_highest_levels_do_not_force_missing_stage(
    analysis_type: str,
    features: dict[str, object],
) -> None:
    result = map_x2_stage_to_risk_level(analysis_type, features)

    assert result.x2_available is True
    assert result.risk_level is not None


def test_hypertension_normal_requires_sbp_and_dbp_to_be_normal() -> None:
    result = map_x2_stage_to_risk_level("HYPERTENSION", {"systolic_bp": 118, "diastolic_bp": 82})

    assert result.risk_level == "ATTENTION"
    assert result.x2_stage_code == "HTN_PRE_STAGE"


def test_obesity_can_calculate_bmi_from_height_and_weight() -> None:
    result = map_x2_stage_to_risk_level("OBE", {"height_cm": 170, "weight_kg": 80})

    assert result.risk_level == "CAUTION"
    assert result.x2_stage_code == "OBESITY_STAGE_1"


@pytest.mark.parametrize(
    ("bmi", "expected_risk_level", "expected_stage_code"),
    [
        (18.4, "LOW", "UNDERWEIGHT"),
        (22.9, "LOW", "NORMAL"),
        (23.0, "ATTENTION", "PRE_OBESITY"),
        (24.9, "ATTENTION", "PRE_OBESITY"),
        (25.0, "CAUTION", "OBESITY_STAGE_1"),
        (29.9, "CAUTION", "OBESITY_STAGE_1"),
        (30.0, "HIGH_CAUTION", "OBESITY_STAGE_2"),
        (34.9, "HIGH_CAUTION", "OBESITY_STAGE_2"),
        (35.0, "HIGH_CAUTION", "OBESITY_STAGE_3"),
    ],
)
def test_obesity_bmi_boundaries_follow_final_rule(
    bmi: float,
    expected_risk_level: str,
    expected_stage_code: str,
) -> None:
    result = map_x2_stage_to_risk_level("OBESITY", {"bmi": bmi})

    assert result.risk_level == expected_risk_level
    assert result.x2_stage_code == expected_stage_code


def test_source_variable_names_are_supported() -> None:
    result = map_x2_stage_to_risk_level("HTN", {"HE_sbp": 160, "HE_dbp": 75})

    assert result.analysis_type == "HYPERTENSION"
    assert result.risk_level == "HIGH_CAUTION"
    assert result.x2_stage_code == "HTN_STAGE_2"


def test_gender_alias_is_supported_as_sex() -> None:
    result = map_x2_stage_to_risk_level("ABO", {"waist_cm": 90, "gender": "FEMALE"})

    assert result.risk_level == "CAUTION"


def test_to_dict_matches_payload_shape() -> None:
    result = map_x2_stage_to_risk_level("HTN", {"systolic_bp": 145, "diastolic_bp": 92}).to_dict()

    assert result == {
        "analysis_type": "HYPERTENSION",
        "risk_level": "CAUTION",
        "result_source": "X2_RULE",
        "x2_stage_code": "HTN_STAGE_1",
        "x2_stage_label": "고혈압 1단계 범위",
        "x2_available": True,
        "x2_missing_fields": [],
    }


def test_unknown_analysis_type_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unsupported X2 analysis_type"):
        map_x2_stage_to_risk_level("UNKNOWN", {})
