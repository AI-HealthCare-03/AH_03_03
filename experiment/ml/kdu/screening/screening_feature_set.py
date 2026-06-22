"""Screening-local X1 service runtime feature definition.

These are the only model input features used by the KDU screening_catboost
artifacts. Clinical exam columns may be used to create screening labels, but
they must not be used as model features.
"""

from __future__ import annotations

import json
from pathlib import Path


FEATURE_SET_NAME = "x1_service_runtime"

X1_SERVICE_RUNTIME_FEATURES: list[str] = [
    "성별",
    "나이",
    "음주빈도",
    "음주량",
    "현재흡연",
    "걷기일수",
    "근력운동일수",
    "고혈압가족력_부",
    "고혈압가족력_모",
    "고혈압가족력_형제",
    "고지혈증가족력_부",
    "고지혈증가족력_모",
    "고지혈증가족력_형제",
    "당뇨가족력_부",
    "당뇨가족력_모",
    "당뇨가족력_형제",
    "키",
    "체중",
    "BMI",
    "직업_관리전문",
    "직업_사무",
    "직업_서비스판매",
    "직업_농림어업",
    "직업_기능노무",
    "직업_주부학생",
    "직업_무직",
    "직업_작업미상",
]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "성별": "사용자 성별. KNHANES 원자료 sex를 서비스 입력 형태로 변환한 값.",
    "나이": "만 나이. 성인 대상 screening 모델의 핵심 인구학 변수.",
    "음주빈도": "설문 기반 음주 빈도.",
    "음주량": "설문 기반 1회 음주량.",
    "현재흡연": "현재 흡연 여부/상태 설문 값.",
    "걷기일수": "최근 1주 걷기 일수.",
    "근력운동일수": "최근 1주 근력운동 일수.",
    "고혈압가족력_부": "부의 고혈압 가족력.",
    "고혈압가족력_모": "모의 고혈압 가족력.",
    "고혈압가족력_형제": "형제자매의 고혈압 가족력.",
    "고지혈증가족력_부": "부의 고지혈증/이상지질혈증 가족력.",
    "고지혈증가족력_모": "모의 고지혈증/이상지질혈증 가족력.",
    "고지혈증가족력_형제": "형제자매의 고지혈증/이상지질혈증 가족력.",
    "당뇨가족력_부": "부의 당뇨 가족력.",
    "당뇨가족력_모": "모의 당뇨 가족력.",
    "당뇨가족력_형제": "형제자매의 당뇨 가족력.",
    "키": "신장. BMI 계산에 사용되는 서비스 입력값.",
    "체중": "체중. BMI 계산에 사용되는 서비스 입력값.",
    "BMI": "키와 체중으로 서비스에서 deterministic하게 계산 가능한 체질량지수.",
    "직업_관리전문": "직업군 one-hot: 관리자/전문가 계열.",
    "직업_사무": "직업군 one-hot: 사무직.",
    "직업_서비스판매": "직업군 one-hot: 서비스/판매직.",
    "직업_농림어업": "직업군 one-hot: 농림어업.",
    "직업_기능노무": "직업군 one-hot: 기능/장치/노무 계열.",
    "직업_주부학생": "직업군 one-hot: 주부/학생.",
    "직업_무직": "직업군 one-hot: 무직.",
    "직업_작업미상": "직업군 one-hot: 미상/분류 불가.",
}

FORBIDDEN_LEAKAGE_COLUMNS: list[str] = [
    "DI1_dg",
    "DI1_pr",
    "DI1_pt",
    "DE1_dg",
    "DE1_pr",
    "DE1_pt",
    "DI2_dg",
    "DI2_pr",
    "DI2_pt",
    "HE_sbp",
    "HE_dbp",
    "HE_glu",
    "HE_HbA1c",
    "HE_chol",
    "HE_TG",
    "HE_LDL",
    "HE_LDL_drct",
    "HE_HDL",
    "HE_HDL_st2",
    "HE_ast",
    "HE_alt",
    "HE_rGTP",
    "HE_GGT",
    "HE_Upro",
    "HE_eGFR",
    "eGFR",
    "HE_crea",
    "target",
    "survey_year",
]


def validate_feature_columns(feature_columns: list[str]) -> tuple[bool, dict[str, list[str]]]:
    """Compare an artifact feature list with the screening-local feature set."""
    expected = list(X1_SERVICE_RUNTIME_FEATURES)
    observed = list(feature_columns)
    return expected == observed, {
        "missing_from_artifact": [column for column in expected if column not in observed],
        "unexpected_in_artifact": [column for column in observed if column not in expected],
        "order_mismatch": [] if expected == observed else observed,
    }


def load_artifact_features(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))
