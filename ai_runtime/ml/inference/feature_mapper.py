from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any


class FeatureMappingError(ValueError):
    def __init__(self, missing_sources: list[str], warnings: list[str] | None = None):
        self.missing_sources = missing_sources
        self.warnings = warnings or []
        message = "ML feature mapping에 필요한 원천값이 누락되었습니다: " + ", ".join(missing_sources)
        super().__init__(message)


@dataclass(frozen=True)
class FeatureMappingResult:
    features: dict[str, Any]
    feature_columns: list[str]
    missing_required_sources: list[str] = field(default_factory=list)
    defaulted_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.missing_required_sources


DERIVED_FEATURES = {
    "나이_19_39",
    "나이_40대",
    "나이_50대",
    "나이_60대",
    "나이_70대",
    "나이_80이상",
    "BMI_구간",
    "음주위험군",
    "걷기활동량",
    "근력활동량",
    "고혈압가족력_합산",
    "당뇨가족력_합산",
    "고지혈증가족력_합산",
    "BMI_X_나이",
    "비만여부",
}

OCCUPATION_ONE_HOT_FEATURES = {
    "직업_관리전문",
    "직업_사무",
    "직업_서비스판매",
    "직업_농림어업",
    "직업_기능노무",
    "직업_주부학생",
    "직업_무직",
    "직업_작업미상",
}

UNAVAILABLE_GRANULAR_FAMILY_FEATURES = {
    "고혈압가족력_모",
    "고혈압가족력_형제",
    "고지혈증가족력_모",
    "고지혈증가족력_형제",
    "당뇨가족력_모",
    "당뇨가족력_형제",
}


def map_service_features(
    user: Any,
    health_record: Any,
    feature_columns: list[str],
    *,
    strict: bool = True,
) -> FeatureMappingResult:
    base_row, missing_sources, warnings = _base_feature_row(user, health_record)
    row = base_row | _feature_engineering(base_row)

    ordered: dict[str, Any] = {}
    defaulted_features: list[str] = []
    for column in feature_columns:
        value = row.get(column)
        if value is None and _can_default_zero(column):
            value = 0.0
            defaulted_features.append(column)
        ordered[column] = value

    result = FeatureMappingResult(
        features=ordered,
        feature_columns=list(feature_columns),
        missing_required_sources=missing_sources,
        defaulted_features=defaulted_features,
        warnings=warnings,
    )
    if strict and missing_sources:
        raise FeatureMappingError(missing_sources, warnings)
    return result


def build_service_feature_row(user: Any, health_record: Any, feature_columns: list[str]) -> dict[str, Any]:
    return map_service_features(user, health_record, feature_columns, strict=False).features


def _base_feature_row(user: Any, health_record: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    missing: list[str] = []
    warnings: list[str] = []

    gender = _gender_value(_get(user, "gender"))
    birthday = _get(user, "birthday")
    age = _age(birthday)
    height_cm = _number(_get(health_record, "height_cm"))
    weight_kg = _number(_get(health_record, "weight_kg"))
    bmi = _bmi(health_record)
    drinking_frequency = _drinking_frequency(_get(health_record, "drinking_frequency"))
    drinking_amount = _drinking_amount(_get(health_record, "drinking_amount"))
    smoking_status = _get(health_record, "smoking_status")
    current_smoking = _current_smoking(smoking_status)
    walking_days = _number(_get(health_record, "walking_days_per_week"))
    strength_days = _number(_get(health_record, "strength_days_per_week"))
    occupation = str(_get(health_record, "occupation_code") or "").upper()

    _mark_missing(missing, gender, "성별")
    _mark_missing(missing, age, "나이")
    _mark_missing(missing, height_cm, "키")
    _mark_missing(missing, weight_kg, "체중")
    _mark_missing(missing, bmi, "BMI")
    _mark_missing(missing, drinking_frequency, "음주빈도")
    _mark_missing(missing, drinking_amount, "음주량")
    _mark_missing(missing, current_smoking, "현재흡연")
    _mark_missing(missing, walking_days, "걷기일수")
    _mark_missing(missing, strength_days, "근력운동일수")
    if not occupation:
        warnings.append("직업군이 없어 직업_작업미상으로 매핑합니다.")

    family_htn = _family_flag(_get(health_record, "family_htn"))
    family_dm = _family_flag(_get(health_record, "family_dm"))
    family_dyslipidemia = _family_flag(_get(health_record, "family_dyslipidemia"))
    _mark_missing(missing, family_htn, "고혈압 가족력")
    _mark_missing(missing, family_dm, "당뇨 가족력")
    _mark_missing(missing, family_dyslipidemia, "이상지질혈증 가족력")

    row = {
        "성별": gender,
        "나이": age,
        "음주빈도": drinking_frequency,
        "음주량": drinking_amount,
        "현재흡연": current_smoking,
        "걷기일수": walking_days,
        "근력운동일수": strength_days,
        "고혈압가족력_부": family_htn,
        "고혈압가족력_모": 0.0,
        "고혈압가족력_형제": 0.0,
        "고지혈증가족력_부": family_dyslipidemia,
        "고지혈증가족력_모": 0.0,
        "고지혈증가족력_형제": 0.0,
        "당뇨가족력_부": family_dm,
        "당뇨가족력_모": 0.0,
        "당뇨가족력_형제": 0.0,
        "키": height_cm,
        "체중": weight_kg,
        "BMI": bmi,
        "직업_관리전문": 1.0 if occupation in {"MANAGER", "PROFESSIONAL", "ADMIN_PROFESSIONAL"} else 0.0,
        "직업_사무": 1.0 if occupation in {"OFFICE", "CLERK"} else 0.0,
        "직업_서비스판매": 1.0 if occupation in {"SERVICE", "SALES", "SERVICE_SALES"} else 0.0,
        "직업_농림어업": 1.0 if occupation in {"AGRICULTURE", "FISHERY", "FARMING"} else 0.0,
        "직업_기능노무": 1.0 if occupation in {"LABOR", "TECHNICAL", "MANUAL"} else 0.0,
        "직업_주부학생": 1.0 if occupation in {"HOMEMAKER", "STUDENT"} else 0.0,
        "직업_무직": 1.0 if occupation in {"UNEMPLOYED", "NONE"} else 0.0,
        "직업_작업미상": 1.0 if not occupation or occupation in {"UNKNOWN", "OTHER"} else 0.0,
    }
    return row, missing, warnings


def _feature_engineering(row: dict[str, Any]) -> dict[str, Any]:
    age = _number(row.get("나이"))
    bmi = _number(row.get("BMI"))
    drinking_frequency = _number(row.get("음주빈도"))
    walking_days = _number(row.get("걷기일수"))
    strength_days = _number(row.get("근력운동일수"))
    return {
        "나이_19_39": _range_flag(age, 0, 40),
        "나이_40대": _range_flag(age, 40, 50),
        "나이_50대": _range_flag(age, 50, 60),
        "나이_60대": _range_flag(age, 60, 70),
        "나이_70대": _range_flag(age, 70, 80),
        "나이_80이상": 1.0 if age is not None and age >= 80 else 0.0,
        "BMI_구간": _bmi_bin(bmi),
        "음주위험군": _alcohol_risk(drinking_frequency),
        "걷기활동량": _walk_level(walking_days),
        "근력활동량": _strength_level(strength_days),
        "고혈압가족력_합산": _family_sum(row, "고혈압가족력"),
        "당뇨가족력_합산": _family_sum(row, "당뇨가족력"),
        "고지혈증가족력_합산": _family_sum(row, "고지혈증가족력"),
        "BMI_X_나이": bmi * age if bmi is not None and age is not None else None,
        "비만여부": 1.0 if bmi is not None and bmi >= 25 else 0.0 if bmi is not None else None,
    }


def _get(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gender_value(value: Any) -> float | None:
    gender = str(value or "").upper()
    if gender.endswith("FEMALE"):
        return 2.0
    if gender.endswith("MALE"):
        return 1.0
    return None


def _age(birthday: Any) -> float | None:
    if birthday is None:
        return None
    if isinstance(birthday, datetime):
        birthday = birthday.date()
    if isinstance(birthday, date):
        today = date.today()
        return float(today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day)))
    return None


def _bmi(health_record: Any) -> float | None:
    bmi = _number(_get(health_record, "bmi"))
    if bmi is not None:
        return bmi
    height_cm = _number(_get(health_record, "height_cm"))
    weight_kg = _number(_get(health_record, "weight_kg"))
    if height_cm is None or weight_kg is None or height_cm <= 0:
        return None
    return weight_kg / ((height_cm / 100) ** 2)


def _family_flag(value: Any) -> float | None:
    if value is None or value == "":
        return None
    normalized = str(value).strip().upper()
    if normalized == "YES":
        return 1.0
    if normalized == "NO":
        return 0.0
    return float("nan")


def _current_smoking(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return 1.0 if str(value).upper() == "CURRENT_SMOKER" else 0.0


def _family_sum(row: dict[str, Any], prefix: str) -> float:
    return min(
        sum(_number(row.get(column)) or 0.0 for column in (f"{prefix}_부", f"{prefix}_모", f"{prefix}_형제")),
        3.0,
    )


def _drinking_frequency(value: Any) -> float | None:
    mapping = {
        "NONE": 0.0,          # 마시지 않음
        "RARE": 1.0,          # 월 1회 미만
        "MONTHLY_1": 2.0,     # 월 1회
        "MONTHLY_2_4": 3.0,   # 월 2-4회
        "WEEKLY_2_3": 4.0,    # 주 2-3회
        "WEEKLY_4_PLUS": 5.0, # 주 4회 이상
        "DAILY": 5.0,
    }
    return mapping.get(str(value or "").upper())



def _drinking_amount(value: Any) -> float | None:
    mapping = {
        "NONE": 0.0,
        "ONE_TO_TWO": 1.0,
        "THREE_TO_FOUR": 2.0,
        "FIVE_TO_SIX": 3.0,
        "SEVEN_TO_NINE": 4.0,  # 추가
        "TEN_PLUS": 5.0,       # 추가
        "SEVEN_PLUS": 4.0,     # 기존 호환
        "HEAVY": 4.0,          # 기존 호환
        "LIGHT": 1.0,          # 기존 호환
        "THREE_TO_SIX": 3.0,   # 기존 호환
    }
    return mapping.get(str(value or "").upper())


def _range_flag(value: float | None, start: float, end: float) -> float:
    if value is None:
        return float("nan")
    return 1.0 if start <= value < end else 0.0


def _bmi_bin(value: float | None) -> float | None:
    if value is None:
        return None
    if value < 23:
        return 0.0
    if value < 25:
        return 1.0
    if value < 30:
        return 2.0
    return 3.0


def _alcohol_risk(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return 0.0
    if value <= 2:
        return 1.0
    return 2.0


def _walk_level(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return 0.0
    if value <= 3:
        return 1.0
    return 2.0


def _strength_level(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return 0.0
    if value <= 2:
        return 1.0
    return 2.0


def _mark_missing(missing: list[str], value: Any, label: str) -> None:
    if value is None:
        missing.append(label)


def _can_default_zero(column: str) -> bool:
    return (
        column in DERIVED_FEATURES
        or column in OCCUPATION_ONE_HOT_FEATURES
        or column in UNAVAILABLE_GRANULAR_FAMILY_FEATURES
    )
