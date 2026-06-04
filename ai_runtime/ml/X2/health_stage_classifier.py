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
    "HE_ast": "ast",
    "HE_alt": "alt",
    "HE_wc": "waist_cm",
    "HE_gamma_GT": "gamma_gtp",
    "HE_upro": "urine_protein",
    "HE_crea": "creatinine",
    "HE_GFR": "egfr",
}

SUPPORTED_DISEASES = (
    "HTN",
    "DM",
    "DL",
    "OBE",
    "ANEM",
    "FL",
    "ABO",
    "LF",
    "KF",
    "CKD",
)


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
        detail=(
            f"수축기 {sbp:g}mmHg, 이완기 {dbp:g}mmHg 기준의 참고용 단계입니다. 의료기관 상담을 권장할 수 있습니다."
        ),
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
        detail=(f"공복혈당 {glucose_text}, 당화혈색소 {a1c_text} 기준의 참고용 단계입니다. 의료 진단이 아닙니다."),
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
        detail=(f"BMI {bmi_value.quantize(Decimal('0.1'))} 기준의 참고용 단계입니다. 의료 진단이 아닙니다."),
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
        detail=(f"{gender_label} 혈색소 {hb:g}g/dL 기준의 참고용 단계입니다. 의료기관 상담을 권장할 수 있습니다."),
        missing=[],
    )


def classify_fl(
    ast: Any = None,
    alt: Any = None,
    bmi: Any = None,
    height_cm: Any = None,
    weight_kg: Any = None,
    gender: Any = None,
) -> StageResult:
    """지방간 위험도 분류 (HSI, Hepatic Steatosis Index).

    HSI = 8 × (AST / ALT) + BMI + 2 × (여성이면 1, 아니면 0)
    - HSI < 30  : 저위험군
    - 30 ≤ HSI ≤ 36 : 위험군
    - HSI > 36  : 고위험군
    """
    ast_val = _to_decimal(ast)
    alt_val = _to_decimal(alt)
    bmi_val = _to_decimal(bmi)
    if bmi_val is None:
        bmi_val = _calculate_bmi(height_cm, weight_kg)
    normalized_gender = _normalize_gender(gender)

    missing = []
    if ast_val is None:
        missing.append("ast")
    if alt_val is None:
        missing.append("alt")
    if alt_val is not None and alt_val == 0:
        missing.append("alt (0은 허용되지 않습니다)")
    if bmi_val is None:
        missing.append("bmi (또는 height_cm, weight_kg)")
    if normalized_gender is None:
        missing.append("gender")
    if missing:
        return _missing_result("FL", missing)

    female_flag = Decimal("1") if normalized_gender == "FEMALE" else Decimal("0")
    hsi = Decimal("8") * (ast_val / alt_val) + bmi_val + Decimal("2") * female_flag

    if hsi < 30:
        stage, label = "LOW_RISK", "저위험군"
        detail_risk = "지방간 가능성이 낮은 저위험군"
    elif hsi <= 36:
        stage, label = "RISK", "위험군"
        detail_risk = "지방간 가능성이 있는 위험군"
    else:
        stage, label = "HIGH_RISK", "고위험군"
        detail_risk = "지방간 가능성이 높은 고위험군"

    gender_label = "여성" if normalized_gender == "FEMALE" else "남성"
    hsi_rounded = hsi.quantize(Decimal("0.1"))
    return StageResult(
        disease="FL",
        stage=stage,
        label=label,
        detail=(
            f"HSI {hsi_rounded} ({gender_label}, "
            f"AST {ast_val:g}U/L, ALT {alt_val:g}U/L, "
            f"BMI {bmi_val.quantize(Decimal('0.1'))}) 기준 {detail_risk}입니다. "
            "참고용 판정이며 의료 진단이 아닙니다."
        ),
        missing=[],
    )


def classify_abo(waist_cm: Any = None, gender: Any = None) -> StageResult:
    """복부비만 위험도 분류 (대한비만학회 기준).

    남성 허리둘레:
      < 90cm       → 저위험군
      90cm ~ 94cm  → 위험군
      ≥ 95cm       → 고위험군

    여성 허리둘레:
      < 85cm       → 저위험군
      85cm ~ 89cm  → 위험군
      ≥ 90cm       → 고위험군
    """
    waist = _to_decimal(waist_cm)
    normalized_gender = _normalize_gender(gender)

    missing = []
    if waist is None:
        missing.append("waist_cm")
    if normalized_gender is None:
        missing.append("gender")
    if missing:
        return _missing_result("ABO", missing)

    if normalized_gender == "MALE":
        if waist < 90:
            stage, label = "LOW_RISK", "저위험군"
        elif waist <= 94:
            stage, label = "RISK", "위험군"
        else:
            stage, label = "HIGH_RISK", "고위험군"
    else:  # FEMALE
        if waist < 85:
            stage, label = "LOW_RISK", "저위험군"
        elif waist <= 89:
            stage, label = "RISK", "위험군"
        else:
            stage, label = "HIGH_RISK", "고위험군"

    gender_label = "남성" if normalized_gender == "MALE" else "여성"
    cutoff_text = "90/95cm" if normalized_gender == "MALE" else "85/90cm"
    return StageResult(
        disease="ABO",
        stage=stage,
        label=label,
        detail=(
            f"{gender_label} 허리둘레 {waist:g}cm 기준 {label}입니다 "
            f"(대한비만학회 기준: {gender_label} 위험군 {cutoff_text}). "
            "참고용 판정이며 의료 진단이 아닙니다."
        ),
        missing=[],
    )


def classify_lf(
    ast: Any = None,
    alt: Any = None,
    gamma_gtp: Any = None,
    gender: Any = None,
) -> StageResult:
    """간기능 위험도 분류 (국가건강검진 기준).

    아래 조건 중 하나라도 해당하면 '간기능 이상 의심':
      - AST > 40 IU/L
      - ALT > 40 IU/L
      - γ-GTP > 63 IU/L (남성) 또는 > 35 IU/L (여성)

    ast, alt, gamma_gtp 중 적어도 하나는 필수.
    gamma_gtp를 입력한 경우 gender도 필수.
    """
    ast_val = _to_decimal(ast)
    alt_val = _to_decimal(alt)
    gtp_val = _to_decimal(gamma_gtp)
    normalized_gender = _normalize_gender(gender)

    # 최소 하나의 수치 필요
    if ast_val is None and alt_val is None and gtp_val is None:
        return _missing_result("LF", ["ast", "alt", "gamma_gtp"])

    # γ-GTP를 입력했는데 성별이 없으면 판정 불가
    if gtp_val is not None and normalized_gender is None:
        return _missing_result("LF", ["gender"])

    abnormal_reasons: list[str] = []

    if ast_val is not None and ast_val > 40:
        abnormal_reasons.append(f"AST {ast_val:g}IU/L (기준 초과)")
    if alt_val is not None and alt_val > 40:
        abnormal_reasons.append(f"ALT {alt_val:g}IU/L (기준 초과)")
    if gtp_val is not None and normalized_gender is not None:
        gtp_cutoff = Decimal("63") if normalized_gender == "MALE" else Decimal("35")
        if gtp_val > gtp_cutoff:
            gender_label = "남성" if normalized_gender == "MALE" else "여성"
            abnormal_reasons.append(f"γ-GTP {gtp_val:g}IU/L (기준 초과, {gender_label} 기준 {gtp_cutoff:g}IU/L)")

    # 수치가 있으나 모두 정상 범위인 경우
    if abnormal_reasons:
        stage, label = "ABNORMAL_SUSPECTED", "간기능 이상 의심"
        detail = (
            ", ".join(abnormal_reasons) + "으로 간기능 이상이 의심됩니다. 참고용 판정이며 의료기관 상담을 권장합니다."
        )
    else:
        stage, label = "NORMAL", "정상 범위"
        values_text = []
        if ast_val is not None:
            values_text.append(f"AST {ast_val:g}IU/L")
        if alt_val is not None:
            values_text.append(f"ALT {alt_val:g}IU/L")
        if gtp_val is not None:
            values_text.append(f"γ-GTP {gtp_val:g}IU/L")
        detail = ", ".join(values_text) + " 기준 정상 범위입니다. 참고용 판정이며 의료 진단이 아닙니다."

    return StageResult(
        disease="LF",
        stage=stage,
        label=label,
        detail=detail,
        missing=[],
    )


_URINE_PROTEIN_POSITIVE = {
    # 숫자 코드 (국가건강검진 DB 코드값)
    "1",
    "2",
    "3",
    "4",
    # 텍스트 표기
    "+1",
    "+2",
    "+3",
    "+4",
    "1+",
    "2+",
    "3+",
    "4+",
    "양성",
    "양성(+1)",
    "양성(+2)",
    "양성(+3)",
    "양성(+4)",
    "POSITIVE",
    "POS",
    "+",
}
_URINE_PROTEIN_NEGATIVE = {
    "0",
    "-1",
    "음성",
    "음성(-)",
    "NEGATIVE",
    "NEG",
    "-",
    # 미량(±)은 양성 기준 미달이므로 정상 처리
    "미량",
    "±",
    "TRACE",
}


def _parse_urine_protein(value: Any) -> bool | None:
    """요단백 값을 양성(True) / 음성(False) / 판정불가(None)로 변환."""
    if value is None or value == "":
        return None
    token = str(value).strip().upper().replace(" ", "")
    if token in {v.upper() for v in _URINE_PROTEIN_POSITIVE}:
        return True
    if token in {v.upper() for v in _URINE_PROTEIN_NEGATIVE}:
        return False
    return None


def classify_kf(
    urine_protein: Any = None,
    creatinine: Any = None,
    egfr: Any = None,
) -> StageResult:
    """신장기능 위험도 분류 (국가건강검진 기준).

    아래 조건 중 하나라도 해당하면 '신기능 손상 의심':
      - 요단백 양성(+1) 이상
      - 혈청 크레아티닌 > 1.5 mg/dL
      - eGFR ≤ 60 mL/min/1.73m²

    세 항목 모두 미입력 시 판정 불가.
    요단백이 인식 불가 값이면 판정 불가.
    """
    protein_positive = _parse_urine_protein(urine_protein)
    crea_val = _to_decimal(creatinine)
    egfr_val = _to_decimal(egfr)

    # 모두 미입력
    if urine_protein is None and creatinine is None and egfr is None:
        return _missing_result("KF", ["urine_protein", "creatinine", "egfr"])

    # 요단백 입력했으나 인식 불가
    if urine_protein is not None and protein_positive is None:
        return _missing_result("KF", [f"urine_protein (인식 불가 값: {urine_protein!r})"])

    abnormal_reasons: list[str] = []

    if protein_positive is True:
        abnormal_reasons.append(f"요단백 양성 ({urine_protein})")
    if crea_val is not None and crea_val > Decimal("1.5"):
        abnormal_reasons.append(f"혈청 크레아티닌 {crea_val:g}mg/dL (기준 초과)")
    if egfr_val is not None and egfr_val <= 60:
        abnormal_reasons.append(f"eGFR {egfr_val:g}mL/min/1.73m² (기준 이하)")

    if abnormal_reasons:
        stage, label = "DAMAGE_SUSPECTED", "신기능 손상 의심"
        detail = (
            ", ".join(abnormal_reasons) + "으로 신기능 손상이 의심됩니다. 참고용 판정이며 의료기관 상담을 권장합니다."
        )
    else:
        stage, label = "NORMAL", "정상 범위"
        values_text = []
        if protein_positive is not None:
            values_text.append("요단백 음성")
        if crea_val is not None:
            values_text.append(f"크레아티닌 {crea_val:g}mg/dL")
        if egfr_val is not None:
            values_text.append(f"eGFR {egfr_val:g}mL/min/1.73m²")
        detail = ", ".join(values_text) + " 기준 정상 범위입니다. 참고용 판정이며 의료 진단이 아닙니다."

    return StageResult(
        disease="KF",
        stage=stage,
        label=label,
        detail=detail,
        missing=[],
    )


def classify_ckd(egfr: Any = None) -> StageResult:
    """만성콩팥병(CKD) 위험도 분류 (KDIGO 기준).

    중증도가 높은 단계가 우선 적용됩니다 (겹치는 구간은 고위험 우선):
      eGFR < 30               → 즉시 병원 방문 권유
      30 ≤ eGFR < 45          → 초고위험군
      45 ≤ eGFR < 65          → 고위험군   (※ 60-64 구간은 위험군 범위와 겹치나 고위험 우선)
      65 ≤ eGFR < 90          → 위험군
      eGFR ≥ 90               → 저위험군
    """
    egfr_val = _to_decimal(egfr)
    if egfr_val is None:
        return _missing_result("CKD", ["egfr"])

    if egfr_val < 30:
        stage = "URGENT"
        label = "즉시 병원 방문 권유"
        detail = (
            f"eGFR {egfr_val:g}mL/min/1.73m² — 신기능이 심각하게 저하된 상태입니다. "
            "즉시 신장내과 전문의 진료를 받으시기 바랍니다. 참고용 판정이며 의료 진단이 아닙니다."
        )
    elif egfr_val < 45:
        stage = "VERY_HIGH_RISK"
        label = "초고위험군"
        detail = (
            f"eGFR {egfr_val:g}mL/min/1.73m² — 신기능이 현저히 저하된 초고위험군입니다. "
            "신장내과 전문의 상담을 강력히 권고합니다. 참고용 판정이며 의료 진단이 아닙니다."
        )
    elif egfr_val < 65:
        stage = "HIGH_RISK"
        label = "고위험군"
        detail = (
            f"eGFR {egfr_val:g}mL/min/1.73m² — 신기능이 중등도로 저하된 고위험군입니다. "
            "의료기관 상담을 권장합니다. 참고용 판정이며 의료 진단이 아닙니다."
        )
    elif egfr_val < 90:
        stage = "RISK"
        label = "위험군"
        detail = (
            f"eGFR {egfr_val:g}mL/min/1.73m² — 신기능이 경미하게 저하된 위험군입니다. "
            "정기적인 추적 관찰을 권장합니다. 참고용 판정이며 의료 진단이 아닙니다."
        )
    else:
        stage = "LOW_RISK"
        label = "저위험군"
        detail = (
            f"eGFR {egfr_val:g}mL/min/1.73m² — 신기능이 정상 범위의 저위험군입니다. "
            "참고용 판정이며 의료 진단이 아닙니다."
        )

    return StageResult(
        disease="CKD",
        stage=stage,
        label=label,
        detail=detail,
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
    ast: Any = None,
    alt: Any = None,
    waist_cm: Any = None,
    gamma_gtp: Any = None,
    urine_protein: Any = None,
    creatinine: Any = None,
    egfr: Any = None,
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
        "FL": classify_fl(
            ast=ast,
            alt=alt,
            bmi=bmi,
            height_cm=height_cm,
            weight_kg=weight_kg,
            gender=gender,
        ),
        "ABO": classify_abo(waist_cm=waist_cm, gender=gender),
        "LF": classify_lf(ast=ast, alt=alt, gamma_gtp=gamma_gtp, gender=gender),
        "KF": classify_kf(urine_protein=urine_protein, creatinine=creatinine, egfr=egfr),
        "CKD": classify_ckd(egfr=egfr),
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
