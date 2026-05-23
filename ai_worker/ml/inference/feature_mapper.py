from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any


def build_service_feature_row(user: Any, health_record: Any, feature_columns: list[str]) -> dict[str, Any]:
    row = _base_feature_row(user, health_record)
    row.update(_feature_engineering(row))
    return {column: row.get(column) for column in feature_columns}


def _base_feature_row(user: Any, health_record: Any) -> dict[str, Any]:
    occupation = str(getattr(health_record, "occupation_code", "") or "").upper()
    return {
        "성별": _gender_value(getattr(user, "gender", None)),
        "나이": _age(getattr(user, "birthday", None)),
        "음주빈도": _drinking_frequency(getattr(health_record, "drinking_frequency", None)),
        "음주량": _drinking_amount(getattr(health_record, "drinking_amount", None)),
        "현재흡연": 1.0 if getattr(health_record, "smoking_status", None) == "CURRENT_SMOKER" else 0.0,
        "걷기일수": _number(getattr(health_record, "walking_days_per_week", None)),
        "근력운동일수": _number(getattr(health_record, "strength_days_per_week", None)),
        "고혈압가족력_부": _family_flag(getattr(health_record, "family_htn", None)),
        "고혈압가족력_모": 0.0,
        "고혈압가족력_형제": 0.0,
        "고지혈증가족력_부": _family_flag(getattr(health_record, "family_dyslipidemia", None)),
        "고지혈증가족력_모": 0.0,
        "고지혈증가족력_형제": 0.0,
        "당뇨가족력_부": _family_flag(getattr(health_record, "family_dm", None)),
        "당뇨가족력_모": 0.0,
        "당뇨가족력_형제": 0.0,
        "키": _number(getattr(health_record, "height_cm", None)),
        "체중": _number(getattr(health_record, "weight_kg", None)),
        "BMI": _bmi(health_record),
        "직업_관리전문": 1 if occupation in {"MANAGER", "PROFESSIONAL", "ADMIN_PROFESSIONAL"} else 0,
        "직업_사무": 1 if occupation in {"OFFICE", "CLERK"} else 0,
        "직업_서비스판매": 1 if occupation in {"SERVICE", "SALES", "SERVICE_SALES"} else 0,
        "직업_농림어업": 1 if occupation in {"AGRICULTURE", "FISHERY", "FARMING"} else 0,
        "직업_기능노무": 1 if occupation in {"LABOR", "TECHNICAL", "MANUAL"} else 0,
        "직업_주부학생": 1 if occupation in {"HOMEMAKER", "STUDENT"} else 0,
        "직업_무직": 1 if occupation in {"UNEMPLOYED", "NONE"} else 0,
        "직업_작업미상": 1 if not occupation or occupation in {"UNKNOWN", "OTHER"} else 0,
    }


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
        "나이_80이상": 1 if age is not None and age >= 80 else 0,
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
    if gender.endswith("MALE") and not gender.endswith("FEMALE"):
        return 1.0
    if gender.endswith("FEMALE"):
        return 2.0
    return None


def _age(birthday: Any) -> float | None:
    if birthday is None:
        return None
    if isinstance(birthday, date):
        today = date.today()
        return float(today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day)))
    return None


def _bmi(health_record: Any) -> float | None:
    bmi = _number(getattr(health_record, "bmi", None))
    if bmi is not None:
        return bmi
    height_cm = _number(getattr(health_record, "height_cm", None))
    weight_kg = _number(getattr(health_record, "weight_kg", None))
    if height_cm is None or weight_kg is None or height_cm <= 0:
        return None
    return weight_kg / ((height_cm / 100) ** 2)


def _family_flag(value: Any) -> float:
    return 1.0 if str(value or "").upper() == "YES" else 0.0


def _family_sum(row: dict[str, Any], prefix: str) -> float:
    return min(
        sum(_number(row.get(column)) or 0.0 for column in (f"{prefix}_부", f"{prefix}_모", f"{prefix}_형제")),
        3.0,
    )


def _drinking_frequency(value: Any) -> float | None:
    mapping = {
        "NONE": 0.0,
        "MONTHLY_1": 1.0,
        "MONTHLY_2_4": 2.0,
        "WEEKLY_2_3": 3.0,
        "WEEKLY_4_PLUS": 4.0,
        "DAILY": 5.0,
    }
    return mapping.get(str(value or "").upper())


def _drinking_amount(value: Any) -> float | None:
    mapping = {
        "NONE": 0.0,
        "LIGHT": 1.0,
        "ONE_TO_TWO": 1.0,
        "THREE_TO_FOUR": 2.0,
        "FIVE_TO_SIX": 3.0,
        "SEVEN_PLUS": 4.0,
        "HEAVY": 4.0,
    }
    return mapping.get(str(value or "").upper())


def _range_flag(value: float | None, start: float, end: float) -> int:
    return 1 if value is not None and start <= value < end else 0


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
