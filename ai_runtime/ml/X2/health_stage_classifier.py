"""Rule-based X2 health stage classifier.

This module is a pure Python reference implementation for health exam values.
It does not call the database, OCR, queues, or ML models. Outputs are
reference-only stage labels and must not be presented as medical decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

SOURCE_VARIABLE_MAP = {
    "HE_sbp": "systolic_bp",
    "HE_dbp": "diastolic_bp",
    "HE_glu": "fasting_glucose",
    "HE_HbA1c": "hba1c",
    "HE_chol": "total_cholesterol",
    "HE_TG": "triglyceride",
    "HE_HDL_st2": "hdl_cholesterol",
    "HE_LDL_drct": "ldl_cholesterol",
    "HE_BMI": "bmi",
    "HE_ht": "height_cm",
    "HE_wt": "weight_kg",
    "HE_HB": "hemoglobin",
    "sex": "gender",
}

SUPPORTED_DISEASES = ("HTN", "DM", "DL", "OBE", "ANEM")


@dataclass(frozen=True)
class StageResult:
    disease: str
    stage: str
    label: str
    detail: str
    missing: list[str]

    def is_normal(self) -> bool:
        return self.stage == "NORMAL"

    def is_classifiable(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _missing_result(disease: str, missing: list[str]) -> StageResult:
    return StageResult(
        disease=disease,
        stage="UNCLASSIFIABLE",
        label="판정 불가",
        detail="필수 수치가 부족해 참고용 판정을 만들 수 없습니다.",
        missing=missing,
    )


def classify_htn(systolic_bp: Any = None, diastolic_bp: Any = None) -> StageResult:
    sbp = _to_decimal(systolic_bp)
    dbp = _to_decimal(diastolic_bp)
    missing = []
    if sbp is None:
        missing.append("systolic_bp")
    if dbp is None:
        missing.append("diastolic_bp")
    if missing:
        return _missing_result("HTN", missing)

    if sbp >= 160 or dbp >= 100:
        stage, label = "HTN_STAGE_2", "고혈압 2단계 범위"
    elif sbp >= 140 or dbp >= 90:
        stage, label = "HTN_STAGE_1", "고혈압 1단계 범위"
    elif sbp >= 130 or dbp >= 80:
        stage, label = "HTN_PRE_STAGE", "고혈압 전단계 범위"
    elif 120 <= sbp <= 129 and dbp < 80:
        stage, label = "ELEVATED", "주의혈압 범위"
    else:
        stage, label = "NORMAL", "정상 범위"

    return StageResult(
        disease="HTN",
        stage=stage,
        label=label,
        detail=f"수축기 {sbp:g}mmHg, 이완기 {dbp:g}mmHg 기준의 참고용 단계입니다. 의료기관 상담을 권장할 수 있습니다.",
        missing=[],
    )


def classify_dm(fasting_glucose: Any = None, hba1c: Any = None) -> StageResult:
    glucose = _to_decimal(fasting_glucose)
    a1c = _to_decimal(hba1c)
    if glucose is None and a1c is None:
        return _missing_result("DM", ["fasting_glucose", "hba1c"])

    if (glucose is not None and glucose >= 126) or (a1c is not None and a1c >= Decimal("6.5")):
        stage, label = "DIABETES_RANGE", "당뇨병 범위"
    elif (glucose is not None and glucose >= 100) or (a1c is not None and a1c >= Decimal("5.7")):
        stage, label = "PRE_DIABETES_RANGE", "공복혈당장애/전단계 범위"
    elif glucose is None:
        return _missing_result("DM", ["fasting_glucose"])
    elif a1c is None:
        return _missing_result("DM", ["hba1c"])
    else:
        stage, label = "NORMAL", "정상 범위"

    glucose_text = "-" if glucose is None else f"{glucose:g}mg/dL"
    a1c_text = "-" if a1c is None else f"{a1c:g}%"
    return StageResult(
        disease="DM",
        stage=stage,
        label=label,
        detail=f"공복혈당 {glucose_text}, 당화혈색소 {a1c_text} 기준의 참고용 단계입니다. 의료 진단이 아닙니다.",
        missing=[],
    )


def _collect_lipid_reasons(
    total: Decimal, ldl_value: Decimal, tg: Decimal, hdl_value: Decimal
) -> tuple[int, list[str]]:
    severity = 0
    reasons: list[str] = []
    if total >= 240:
        severity = max(severity, 2)
        reasons.append("총콜레스테롤 관리 필요 범위")
    elif total >= 200:
        severity = max(severity, 1)
        reasons.append("총콜레스테롤 경계 범위")

    if ldl_value >= 190:
        severity = max(severity, 3)
        reasons.append("LDL 고위험 수치 범위")
    elif ldl_value >= 160:
        severity = max(severity, 2)
        reasons.append("LDL 관리 필요 범위")
    elif ldl_value >= 130:
        severity = max(severity, 1)
        reasons.append("LDL 경계 범위")

    if tg >= 500:
        severity = max(severity, 3)
        reasons.append("중성지방 고위험 수치 범위")
    elif tg >= 200:
        severity = max(severity, 2)
        reasons.append("중성지방 관리 필요 범위")
    elif tg >= 150:
        severity = max(severity, 1)
        reasons.append("중성지방 경계 범위")

    if hdl_value < 40:
        severity = max(severity, 2)
        reasons.append("HDL 낮은 범위")

    return severity, reasons


def _lipid_stage(severity: int) -> tuple[str, str]:
    if severity >= 3:
        return "HIGH_RISK_RANGE", "고위험 수치 범위"
    if severity == 2:
        return "MANAGEMENT_RANGE", "관리 필요 범위"
    if severity == 1:
        return "BORDERLINE_RANGE", "경계 범위"
    return "NORMAL", "정상 범위"


def classify_dl(
    total_cholesterol: Any = None,
    ldl_cholesterol: Any = None,
    triglyceride: Any = None,
    hdl_cholesterol: Any = None,
    *,
    ldl: Any = None,
    hdl: Any = None,
) -> StageResult:
    ldl_value = ldl_cholesterol if ldl_cholesterol is not None else ldl
    hdl_value = hdl_cholesterol if hdl_cholesterol is not None else hdl
    total = _to_decimal(total_cholesterol)
    ldl_dec = _to_decimal(ldl_value)
    tg = _to_decimal(triglyceride)
    hdl_dec = _to_decimal(hdl_value)

    missing = []
    if total is None:
        missing.append("total_cholesterol")
    if ldl_dec is None:
        missing.append("ldl_cholesterol")
    if tg is None:
        missing.append("triglyceride")
    if hdl_dec is None:
        missing.append("hdl_cholesterol")
    if missing:
        return _missing_result("DL", missing)

    severity, reasons = _collect_lipid_reasons(total, ldl_dec, tg, hdl_dec)
    stage, label = _lipid_stage(severity)
    if stage == "NORMAL":
        reasons.append("주요 지질 수치가 정상 범위")

    return StageResult(
        disease="DL",
        stage=stage,
        label=label,
        detail=", ".join(reasons) + "입니다. 참고용 판정이며 의료기관 상담을 권장할 수 있습니다.",
        missing=[],
    )


def _calculate_bmi(height_cm: Any, weight_kg: Any) -> Decimal | None:
    height = _to_decimal(height_cm)
    weight = _to_decimal(weight_kg)
    if height is None or weight is None or height <= 0 or weight <= 0:
        return None
    height_m = height / Decimal("100")
    return weight / (height_m * height_m)


def classify_obe(bmi: Any = None, height_cm: Any = None, weight_kg: Any = None) -> StageResult:
    bmi_value = _to_decimal(bmi)
    if bmi_value is None:
        bmi_value = _calculate_bmi(height_cm, weight_kg)
    if bmi_value is None:
        missing = ["bmi"]
        if height_cm is None:
            missing.append("height_cm")
        if weight_kg is None:
            missing.append("weight_kg")
        return _missing_result("OBE", missing)

    if bmi_value >= 35:
        stage, label = "OBESITY_STAGE_3", "비만 3단계 범위"
    elif bmi_value >= 30:
        stage, label = "OBESITY_STAGE_2", "비만 2단계 범위"
    elif bmi_value >= 25:
        stage, label = "OBESITY_STAGE_1", "비만 1단계 범위"
    elif bmi_value >= 23:
        stage, label = "PRE_OBESITY", "비만 전단계 범위"
    elif bmi_value >= Decimal("18.5"):
        stage, label = "NORMAL", "정상 범위"
    else:
        stage, label = "UNDERWEIGHT", "저체중 범위"

    return StageResult(
        disease="OBE",
        stage=stage,
        label=label,
        detail=f"BMI {bmi_value.quantize(Decimal('0.1'))} 기준의 참고용 단계입니다. 의료 진단이 아닙니다.",
        missing=[],
    )


def _normalize_gender(gender: Any) -> str | None:
    if gender is None:
        return None
    normalized = str(gender).strip().upper()
    if normalized in {"MALE", "M", "MAN", "남", "남성"}:
        return "MALE"
    if normalized in {"FEMALE", "F", "WOMAN", "여", "여성"}:
        return "FEMALE"
    return None


def classify_anem(hemoglobin: Any = None, gender: Any = None) -> StageResult:
    hb = _to_decimal(hemoglobin)
    normalized_gender = _normalize_gender(gender)
    missing = []
    if hb is None:
        missing.append("hemoglobin")
    if normalized_gender is None:
        missing.append("gender")
    if missing:
        return _missing_result("ANEM", missing)

    normal_cutoff = Decimal("13.0") if normalized_gender == "MALE" else Decimal("12.0")
    if hb >= normal_cutoff:
        stage, label = "NORMAL", "정상 범위"
    elif hb >= Decimal("11.0"):
        stage, label = "MILD_RANGE", "경증 빈혈 범위"
    elif hb >= Decimal("8.0"):
        stage, label = "MODERATE_RANGE", "중등도 빈혈 범위"
    else:
        stage, label = "SEVERE_RANGE", "중증 빈혈 범위"

    gender_label = "남성" if normalized_gender == "MALE" else "여성"
    return StageResult(
        disease="ANEM",
        stage=stage,
        label=label,
        detail=f"{gender_label} 혈색소 {hb:g}g/dL 기준의 참고용 단계입니다. 의료기관 상담을 권장할 수 있습니다.",
        missing=[],
    )


def classify_all(
    *,
    systolic_bp: Any = None,
    diastolic_bp: Any = None,
    fasting_glucose: Any = None,
    hba1c: Any = None,
    total_cholesterol: Any = None,
    ldl_cholesterol: Any = None,
    triglyceride: Any = None,
    hdl_cholesterol: Any = None,
    bmi: Any = None,
    height_cm: Any = None,
    weight_kg: Any = None,
    hemoglobin: Any = None,
    gender: Any = None,
) -> dict[str, StageResult]:
    return {
        "HTN": classify_htn(systolic_bp=systolic_bp, diastolic_bp=diastolic_bp),
        "DM": classify_dm(fasting_glucose=fasting_glucose, hba1c=hba1c),
        "DL": classify_dl(
            total_cholesterol=total_cholesterol,
            ldl_cholesterol=ldl_cholesterol,
            triglyceride=triglyceride,
            hdl_cholesterol=hdl_cholesterol,
        ),
        "OBE": classify_obe(bmi=bmi, height_cm=height_cm, weight_kg=weight_kg),
        "ANEM": classify_anem(hemoglobin=hemoglobin, gender=gender),
    }


def map_source_variables(values: dict[str, Any]) -> dict[str, Any]:
    """Map source health exam variable names to current service field names."""

    mapped: dict[str, Any] = {}
    for key, value in values.items():
        mapped[SOURCE_VARIABLE_MAP.get(key, key)] = value
    return mapped


def _demo() -> None:
    sample = {
        "systolic_bp": 132,
        "diastolic_bp": 84,
        "fasting_glucose": 108,
        "hba1c": 5.8,
        "total_cholesterol": 205,
        "ldl_cholesterol": 136,
        "triglyceride": 155,
        "hdl_cholesterol": 52,
        "height_cm": 172,
        "weight_kg": 79.2,
        "hemoglobin": 14.2,
        "gender": "MALE",
    }
    for result in classify_all(**sample).values():
        print(result.to_dict())


if __name__ == "__main__":
    _demo()
