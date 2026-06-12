"""KDU X1 service screening candidate configuration."""

from __future__ import annotations

from dataclasses import dataclass


FEATURE_SET = "x1_service_runtime"
TRAINING_MODEL_NAME = "catboost"
ARTIFACT_MODEL_NAME = "screening_catboost"
SERVICE_POLICY_ALLOWED = False
RANDOM_STATE = 42


@dataclass(frozen=True)
class ScreeningTarget:
    disease: str
    policy_suffix: str
    label_policy: str
    data_version: str
    file_name: str
    role: str
    recommended_threshold: float | None
    recommended_gray_zone: tuple[float, float] | None
    service_expression: str


MAIN_SCREENING_TARGETS: tuple[ScreeningTarget, ...] = (
    ScreeningTarget(
        disease="hypertension",
        policy_suffix="screening",
        label_policy="screening_risk",
        data_version="x1_service_runtime_hypertension_hn13_24_screening",
        file_name="hypertension_hn13_24_screening.csv",
        role="main_screening",
        recommended_threshold=0.45,
        recommended_gray_zone=(0.35, 0.65),
        service_expression="혈압 관리 필요 가능성",
    ),
    ScreeningTarget(
        disease="diabetes",
        policy_suffix="screening",
        label_policy="screening_risk",
        data_version="x1_service_runtime_diabetes_hn13_24_screening",
        file_name="diabetes_hn13_24_screening.csv",
        role="main_screening",
        recommended_threshold=0.40,
        recommended_gray_zone=(0.35, 0.65),
        service_expression="혈당 관리 필요 가능성",
    ),
    ScreeningTarget(
        disease="dyslipidemia",
        policy_suffix="screening_v2",
        label_policy="dyslipidemia_screening_v2",
        data_version="x1_service_runtime_dyslipidemia_hn13_24_screening_v2",
        file_name="dyslipidemia_hn13_24_screening_v2.csv",
        role="main_screening",
        recommended_threshold=0.45,
        recommended_gray_zone=(0.35, 0.65),
        service_expression="지질 관리 필요 가능성",
    ),
)


AUXILIARY_TARGETS: tuple[ScreeningTarget, ...] = (
    ScreeningTarget(
        disease="hypertension",
        policy_suffix="strict",
        label_policy="strict_diagnosis",
        data_version="x1_service_runtime_hypertension_hn13_24_strict",
        file_name="hypertension_hn13_24_strict.csv",
        role="auxiliary_high_attention",
        recommended_threshold=None,
        recommended_gray_zone=None,
        service_expression="혈압 검진 확인 권장 보조 신호",
    ),
    ScreeningTarget(
        disease="diabetes",
        policy_suffix="strict",
        label_policy="strict_diagnosis",
        data_version="x1_service_runtime_diabetes_hn13_24_strict",
        file_name="diabetes_hn13_24_strict.csv",
        role="auxiliary_high_attention",
        recommended_threshold=None,
        recommended_gray_zone=None,
        service_expression="혈당 검진 확인 권장 보조 신호",
    ),
    ScreeningTarget(
        disease="dyslipidemia",
        policy_suffix="strict_v2",
        label_policy="dyslipidemia_strict_v2",
        data_version="x1_service_runtime_dyslipidemia_hn13_24_strict_v2",
        file_name="dyslipidemia_hn13_24_strict_v2.csv",
        role="auxiliary_high_attention",
        recommended_threshold=None,
        recommended_gray_zone=None,
        service_expression="지질 검진 확인 권장 보조 신호",
    ),
    ScreeningTarget(
        disease="dyslipidemia",
        policy_suffix="diagnosed_treated_only",
        label_policy="dyslipidemia_diagnosed_treated_only",
        data_version="x1_service_runtime_dyslipidemia_hn13_24_diagnosed_treated_only",
        file_name="dyslipidemia_hn13_24_diagnosed_treated_only.csv",
        role="auxiliary_reference",
        recommended_threshold=None,
        recommended_gray_zone=None,
        service_expression="이상지질 진단/치료 응답 참고 신호",
    ),
)


SCREENING_AND_AUXILIARY_TARGETS: tuple[ScreeningTarget, ...] = MAIN_SCREENING_TARGETS + AUXILIARY_TARGETS


DISEASE_RUNTIME_KEYS = {
    "hypertension": "htn",
    "diabetes": "dm",
    "dyslipidemia": "dl",
}


FORBIDDEN_USER_EXPRESSIONS = (
    "확정 진단",
    "질병입니다",
    "발병 확률",
    "약물 치료 필요",
    "의사 진단 대체",
)
