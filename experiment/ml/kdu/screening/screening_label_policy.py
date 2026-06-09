"""Screening-local screening label definitions for screening_catboost.

The labels in this module are management/check-needed screening labels. They
are not confirmed diagnosis labels and must not be described to users as a
diagnosis or disease probability.

Clinical exam columns are intentionally allowed for target creation here.
Those same columns are forbidden as model input features; see
screening_feature_set.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


SOURCE_FILES_HN13_24: tuple[str, ...] = (
    "HN13_all.sav",
    "HN14_all.sav",
    "HN15_all.sav",
    "HN16_all.sav",
    "HN17_all.sav",
    "HN18_all.sav",
    "HN19_all.sav",
    "HN20_all.sav",
    "HN21_all.sav",
    "HN22_all.sav",
    "HN23_all.sav",
    "HN24_ALL.sav",
)


@dataclass(frozen=True)
class LabelPolicyDefinition:
    disease: str
    policy_suffix: str
    label_name: str
    positive_condition: str
    source_columns: tuple[str, ...]
    note: str


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _flag(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _positive_any(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    positive = pd.Series(False, index=frame.index)
    for column in columns:
        if column in frame.columns:
            positive = positive | _num(frame, column).eq(1)
    return positive


def _first_available_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    for column in columns:
        if column in frame.columns:
            result = result.combine_first(_num(frame, column))
    return result


def _has_any_input(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    present = pd.Series(False, index=frame.index)
    for column in columns:
        if column in frame.columns:
            present = present | _num(frame, column).notna()
    return present


HTN_SOURCE_COLUMNS = ("DI1_dg", "DI1_pr", "DI1_pt", "HE_sbp", "HE_dbp")
DM_SOURCE_COLUMNS = ("DE1_dg", "DE1_pr", "DE1_pt", "HE_glu", "HE_HbA1c")
DL_SOURCE_COLUMNS = (
    "DI2_dg",
    "DI2_pr",
    "DI2_pt",
    "HE_chol",
    "HE_TG",
    "HE_LDL_drct",
    "HE_LDL",
    "HE_HDL_st2",
    "HE_HDL",
)


def hypertension_strict_diagnosis(frame: pd.DataFrame) -> pd.Series:
    """Positive if DI1_dg/pr/pt == 1 or SBP >= 140 or DBP >= 90."""
    measured = (_num(frame, "HE_sbp") >= 140) | (_num(frame, "HE_dbp") >= 90)
    label = _positive_any(frame, ("DI1_dg", "DI1_pr", "DI1_pt")) | measured
    return label.where(_has_any_input(frame, HTN_SOURCE_COLUMNS))


def hypertension_screening_risk(frame: pd.DataFrame) -> pd.Series:
    """Positive if strict positive or elevated BP: SBP >= 130 or DBP >= 80."""
    elevated = (_num(frame, "HE_sbp") >= 130) | (_num(frame, "HE_dbp") >= 80)
    label = hypertension_strict_diagnosis(frame).fillna(False) | elevated
    return label.where(_has_any_input(frame, HTN_SOURCE_COLUMNS))


def diabetes_strict_diagnosis(frame: pd.DataFrame) -> pd.Series:
    """Positive if DE1_dg/pr/pt == 1 or glucose >= 126 or HbA1c >= 6.5."""
    measured = (_num(frame, "HE_glu") >= 126) | (_num(frame, "HE_HbA1c") >= 6.5)
    label = _positive_any(frame, ("DE1_dg", "DE1_pr", "DE1_pt")) | measured
    return label.where(_has_any_input(frame, DM_SOURCE_COLUMNS))


def diabetes_screening_risk(frame: pd.DataFrame) -> pd.Series:
    """Positive if strict positive or prediabetes range glucose/HbA1c."""
    borderline = _num(frame, "HE_glu").between(100, 125.999) | _num(frame, "HE_HbA1c").between(5.7, 6.499)
    label = diabetes_strict_diagnosis(frame).fillna(False) | borderline
    return label.where(_has_any_input(frame, DM_SOURCE_COLUMNS))


def dyslipidemia_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Return screening-local dyslipidemia strict and borderline components."""
    ldl = _first_available_numeric(frame, ("HE_LDL_drct", "HE_LDL"))
    hdl = _first_available_numeric(frame, ("HE_HDL_st2", "HE_HDL"))
    components = pd.DataFrame(index=frame.index)
    components["source_available"] = _has_any_input(frame, DL_SOURCE_COLUMNS)
    components["diagnosis_or_treatment"] = _positive_any(frame, ("DI2_dg", "DI2_pr", "DI2_pt"))
    components["tc_high"] = _flag(_num(frame, "HE_chol") >= 240)
    components["ldl_high"] = _flag(ldl >= 160)
    components["tg_high"] = _flag(_num(frame, "HE_TG") >= 200)
    components["hdl_low"] = _flag(hdl < 40)
    components["tc_borderline"] = _flag(_num(frame, "HE_chol").between(200, 239.999))
    components["ldl_borderline"] = _flag(ldl.between(130, 159.999))
    components["tg_borderline"] = _flag(_num(frame, "HE_TG").between(150, 199.999))
    components["hdl_borderline"] = _flag(hdl.between(40, 49.999))
    high_columns = ["diagnosis_or_treatment", "tc_high", "ldl_high", "tg_high", "hdl_low"]
    borderline_columns = ["tc_borderline", "ldl_borderline", "tg_borderline", "hdl_borderline"]
    components["strict_v2_positive"] = components[high_columns].any(axis=1)
    components["borderline_component_count"] = components[borderline_columns].sum(axis=1)
    components["screening_v2_positive"] = components["strict_v2_positive"] | (
        components["borderline_component_count"] >= 2
    )
    components["diagnosed_treated_only_positive"] = components["diagnosis_or_treatment"]
    return components


def dyslipidemia_strict_v2(frame: pd.DataFrame) -> pd.Series:
    """Positive if DI2_dg/pr/pt == 1, TC >= 240, LDL >= 160, TG >= 200, or HDL < 40."""
    components = dyslipidemia_components(frame)
    label = components["strict_v2_positive"].astype("boolean")
    return label.where(components["source_available"])


def dyslipidemia_screening_v2(frame: pd.DataFrame) -> pd.Series:
    """Positive if strict_v2 positive or at least two borderline lipid components exist."""
    components = dyslipidemia_components(frame)
    label = components["screening_v2_positive"].astype("boolean")
    return label.where(components["source_available"])


def dyslipidemia_diagnosed_treated_only(frame: pd.DataFrame) -> pd.Series:
    """Positive only if DI2_dg/pr/pt == 1; kept as an auxiliary reference label."""
    components = dyslipidemia_components(frame)
    label = components["diagnosed_treated_only_positive"].astype("boolean")
    return label.where(components["source_available"])


LABEL_BUILDERS: dict[tuple[str, str], Callable[[pd.DataFrame], pd.Series]] = {
    ("hypertension", "strict"): hypertension_strict_diagnosis,
    ("hypertension", "screening"): hypertension_screening_risk,
    ("diabetes", "strict"): diabetes_strict_diagnosis,
    ("diabetes", "screening"): diabetes_screening_risk,
    ("dyslipidemia", "strict_v2"): dyslipidemia_strict_v2,
    ("dyslipidemia", "screening_v2"): dyslipidemia_screening_v2,
    ("dyslipidemia", "diagnosed_treated_only"): dyslipidemia_diagnosed_treated_only,
}

POLICY_DEFINITIONS: dict[tuple[str, str], LabelPolicyDefinition] = {
    ("hypertension", "screening"): LabelPolicyDefinition(
        disease="hypertension",
        policy_suffix="screening",
        label_name="screening_risk",
        positive_condition="DI1_dg/pr/pt == 1 OR HE_sbp >= 130 OR HE_dbp >= 80",
        source_columns=HTN_SOURCE_COLUMNS,
        note="Blood pressure management/check-needed screening label, not confirmed hypertension diagnosis.",
    ),
    ("hypertension", "strict"): LabelPolicyDefinition(
        disease="hypertension",
        policy_suffix="strict",
        label_name="strict_diagnosis",
        positive_condition="DI1_dg/pr/pt == 1 OR HE_sbp >= 140 OR HE_dbp >= 90",
        source_columns=HTN_SOURCE_COLUMNS,
        note="Auxiliary high-attention signal only.",
    ),
    ("diabetes", "screening"): LabelPolicyDefinition(
        disease="diabetes",
        policy_suffix="screening",
        label_name="screening_risk",
        positive_condition="DE1_dg/pr/pt == 1 OR HE_glu >= 100 OR HE_HbA1c >= 5.7",
        source_columns=DM_SOURCE_COLUMNS,
        note="Glucose management/check-needed screening label, not confirmed diabetes diagnosis.",
    ),
    ("diabetes", "strict"): LabelPolicyDefinition(
        disease="diabetes",
        policy_suffix="strict",
        label_name="strict_diagnosis",
        positive_condition="DE1_dg/pr/pt == 1 OR HE_glu >= 126 OR HE_HbA1c >= 6.5",
        source_columns=DM_SOURCE_COLUMNS,
        note="Auxiliary high-attention signal only.",
    ),
    ("dyslipidemia", "screening_v2"): LabelPolicyDefinition(
        disease="dyslipidemia",
        policy_suffix="screening_v2",
        label_name="dyslipidemia_screening_v2",
        positive_condition="strict_v2 positive OR borderline lipid component count >= 2",
        source_columns=DL_SOURCE_COLUMNS,
        note="Lipid management/check-needed screening label, not confirmed dyslipidemia diagnosis.",
    ),
    ("dyslipidemia", "strict_v2"): LabelPolicyDefinition(
        disease="dyslipidemia",
        policy_suffix="strict_v2",
        label_name="dyslipidemia_strict_v2",
        positive_condition="DI2_dg/pr/pt == 1 OR TC >= 240 OR LDL >= 160 OR TG >= 200 OR HDL < 40",
        source_columns=DL_SOURCE_COLUMNS,
        note="Auxiliary high-attention signal only.",
    ),
    ("dyslipidemia", "diagnosed_treated_only"): LabelPolicyDefinition(
        disease="dyslipidemia",
        policy_suffix="diagnosed_treated_only",
        label_name="dyslipidemia_diagnosed_treated_only",
        positive_condition="DI2_dg/pr/pt == 1",
        source_columns=("DI2_dg", "DI2_pr", "DI2_pt"),
        note="Reference-only diagnosed/treated response label.",
    ),
}


def get_label_definition(disease: str, policy_suffix: str) -> LabelPolicyDefinition:
    return POLICY_DEFINITIONS[(disease, policy_suffix)]


def build_label(frame: pd.DataFrame, disease: str, policy_suffix: str) -> pd.Series:
    return LABEL_BUILDERS[(disease, policy_suffix)](frame)
