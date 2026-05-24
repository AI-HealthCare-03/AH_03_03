from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from ai_worker.ml.inference.feature_mapper import FeatureMappingError, map_service_features

BASE_FEATURE_COLUMNS = [
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

DM_DL_DERIVED_FEATURE_COLUMNS = [
    "나이_19_39",
    "나이_40대",
    "나이_50대",
    "나이_60대",
    "나이_70대",
    "나이_80이상",
    "BMI_구간",
    "고혈압가족력_합산",
    "당뇨가족력_합산",
    "고지혈증가족력_합산",
    "BMI_X_나이",
]

HTN_DERIVED_FEATURE_COLUMNS = [
    "나이_19_39",
    "나이_40대",
    "나이_50대",
    "나이_60대",
    "나이_70대",
    "나이_80이상",
    "음주위험군",
    "걷기활동량",
    "고혈압가족력_합산",
    "당뇨가족력_합산",
    "고지혈증가족력_합산",
    "BMI_X_나이",
]


def test_mapper_orders_features_by_feature_columns() -> None:
    columns = [
        "성별",
        "나이",
        "키",
        "체중",
        "BMI",
        "현재흡연",
        "직업_사무",
        "나이_40대",
        "BMI_구간",
        "고혈압가족력_합산",
        "BMI_X_나이",
    ]

    result = map_service_features(_user(), _health_record(), columns)

    assert list(result.features) == columns
    assert result.features["성별"] == 1.0
    assert result.features["키"] == 175.0
    assert result.features["체중"] == 80.0
    assert result.features["BMI"] == pytest.approx(26.12, abs=0.01)
    assert result.features["현재흡연"] == 1.0
    assert result.features["직업_사무"] == 1.0
    assert result.features["고혈압가족력_합산"] == 1.0
    assert result.is_valid


def test_mapper_validates_required_source_values() -> None:
    user = _user(gender=None)
    health_record = _health_record(height_cm=None, bmi=None)

    with pytest.raises(FeatureMappingError) as error:
        map_service_features(user, health_record, ["성별", "키", "BMI"], strict=True)

    assert "성별" in error.value.missing_sources
    assert "키" in error.value.missing_sources
    assert "BMI" in error.value.missing_sources


def test_mapper_can_default_derived_features_without_hiding_missing_sources() -> None:
    result = map_service_features(
        _user(birthday=None),
        _health_record(height_cm=None, weight_kg=None, bmi=None),
        ["나이_40대", "BMI_구간", "BMI_X_나이"],
        strict=False,
    )

    assert result.features == {"나이_40대": 0.0, "BMI_구간": 0.0, "BMI_X_나이": 0.0}
    assert "나이" in result.missing_required_sources
    assert "BMI" in result.missing_required_sources
    assert sorted(result.defaulted_features) == ["BMI_X_나이", "BMI_구간"]


def test_mapper_supports_final_feature_counts() -> None:
    dm_columns = BASE_FEATURE_COLUMNS + DM_DL_DERIVED_FEATURE_COLUMNS
    htn_columns = BASE_FEATURE_COLUMNS + HTN_DERIVED_FEATURE_COLUMNS
    dl_columns = BASE_FEATURE_COLUMNS + DM_DL_DERIVED_FEATURE_COLUMNS

    assert len(map_service_features(_user(), _health_record(), dm_columns).features) == 38
    assert len(map_service_features(_user(), _health_record(), htn_columns).features) == 39
    assert len(map_service_features(_user(), _health_record(), dl_columns).features) == 38


def test_mapper_accepts_service_drinking_frequency_rare() -> None:
    result = map_service_features(_user(), _health_record(drinking_frequency="RARE"), ["음주빈도"])

    assert result.features["음주빈도"] == 0.0


def _user(**overrides):
    today = date.today()
    values = {
        "gender": "MALE",
        "birthday": date(today.year - 46, 1, 1),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _health_record(**overrides):
    values = {
        "height_cm": 175,
        "weight_kg": 80,
        "bmi": None,
        "drinking_frequency": "WEEKLY_2_3",
        "drinking_amount": "THREE_TO_FOUR",
        "smoking_status": "CURRENT_SMOKER",
        "walking_days_per_week": 4,
        "strength_days_per_week": 2,
        "family_htn": "YES",
        "family_dm": "NO",
        "family_dyslipidemia": "NO",
        "occupation_code": "OFFICE",
    }
    values.update(overrides)
    return SimpleNamespace(**values)
