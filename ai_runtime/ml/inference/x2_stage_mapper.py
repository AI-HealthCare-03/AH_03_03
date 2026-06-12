from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from ai_runtime.ml.inference.dual_stage_policy import ServiceBand
from ai_runtime.ml.X2.health_stage_classifier import SOURCE_VARIABLE_MAP


class X2ResultSource(StrEnum):
    X2_RULE = "X2_RULE"
    X2_UNAVAILABLE = "X2_UNAVAILABLE"


@dataclass(frozen=True)
class X2StageMappingResult:
    analysis_type: str
    risk_level: str | None
    result_source: str
    x2_stage_code: str | None
    x2_stage_label: str | None
    x2_available: bool
    x2_missing_fields: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DISEASE_ALIASES = {
    "HTN": "HYPERTENSION",
    "HYPERTENSION": "HYPERTENSION",
    "DM": "DIABETES",
    "DIABETES": "DIABETES",
    "DL": "DYSLIPIDEMIA",
    "DYSLIPIDEMIA": "DYSLIPIDEMIA",
    "OBE": "OBESITY",
    "OBESITY": "OBESITY",
    "ABO": "ABDOMINAL_OBESITY",
    "ABDOMINAL_OBESITY": "ABDOMINAL_OBESITY",
    "ANEM": "ANEMIA",
    "ANEMIA": "ANEMIA",
    "FL": "FATTY_LIVER",
    "FATTY_LIVER": "FATTY_LIVER",
    "LF": "LIVER_FUNCTION",
    "LIVER_FUNCTION": "LIVER_FUNCTION",
    "KF": "KIDNEY_FUNCTION",
    "KIDNEY_FUNCTION": "KIDNEY_FUNCTION",
    "CKD": "CHRONIC_KIDNEY_DISEASE",
    "CHRONIC_KIDNEY_DISEASE": "CHRONIC_KIDNEY_DISEASE",
}

_URINE_PROTEIN_POSITIVE = {
    "1",
    "2",
    "3",
    "4",
    "+",
    "+1",
    "+2",
    "+3",
    "+4",
    "plus_1",
    "plus_2",
    "plus_3",
    "plus_4",
    "1+",
    "2+",
    "3+",
    "4+",
    "양성",
    "POS",
    "POSITIVE",
}
_URINE_PROTEIN_NEGATIVE = {
    "0",
    "-1",
    "-",
    "음성",
    "NEG",
    "NEGATIVE",
    "미량",
    "±",
    "TRACE",
}


def map_x2_stage_to_risk_level(analysis_type: str, features: Mapping[str, Any]) -> X2StageMappingResult:
    """Map X2 health-exam values to the service 4-step RiskLevel.

    This mapper is intentionally not connected to the BASIC analysis flow yet.
    Missing disease-specific exam values return X2_UNAVAILABLE, never LOW.
    """
    normalized_analysis_type = _normalize_analysis_type(analysis_type)
    normalized_features = _normalize_features(features)

    if normalized_analysis_type == "HYPERTENSION":
        return _map_htn(normalized_features)
    if normalized_analysis_type == "DIABETES":
        return _map_dm(normalized_features)
    if normalized_analysis_type == "DYSLIPIDEMIA":
        return _map_dl(normalized_features)
    if normalized_analysis_type == "OBESITY":
        return _map_obe(normalized_features)
    if normalized_analysis_type == "ABDOMINAL_OBESITY":
        return _map_abo(normalized_features)
    if normalized_analysis_type == "FATTY_LIVER":
        return _map_fl(normalized_features)
    if normalized_analysis_type == "ANEMIA":
        return _map_anem(normalized_features)
    if normalized_analysis_type == "LIVER_FUNCTION":
        return _map_lf(normalized_features)
    if normalized_analysis_type == "KIDNEY_FUNCTION":
        return _map_kf(normalized_features)
    if normalized_analysis_type == "CHRONIC_KIDNEY_DISEASE":
        return _map_ckd(normalized_features)

    raise ValueError(f"unsupported X2 analysis_type: {analysis_type}")


def _map_htn(features: Mapping[str, Any]) -> X2StageMappingResult:
    sbp = _to_decimal(features.get("systolic_bp"))
    dbp = _to_decimal(features.get("diastolic_bp"))
    missing = _missing_numeric_fields(
        {"systolic_bp": sbp, "diastolic_bp": dbp},
    )
    if missing:
        return _unavailable("HYPERTENSION", missing)

    if sbp >= 160 or dbp >= 100:
        return _available("HYPERTENSION", ServiceBand.HIGH_CAUTION, "HTN_STAGE_2", "고혈압 2단계 범위")
    if sbp >= 140 or dbp >= 90:
        return _available("HYPERTENSION", ServiceBand.CAUTION, "HTN_STAGE_1", "고혈압 1단계 범위")
    if sbp >= 130 or dbp >= 80:
        return _available("HYPERTENSION", ServiceBand.ATTENTION, "HTN_PRE_STAGE", "고혈압 전단계 범위")
    if sbp >= 120 and dbp < 80:
        return _available("HYPERTENSION", ServiceBand.ATTENTION, "ELEVATED", "주의혈압 범위")
    return _available("HYPERTENSION", ServiceBand.LOW, "NORMAL", "정상 범위")


def _map_dm(features: Mapping[str, Any]) -> X2StageMappingResult:
    glucose = _to_decimal(features.get("fasting_glucose"))
    a1c = _to_decimal(features.get("hba1c"))
    if glucose is None and a1c is None:
        return _unavailable("DIABETES", ["fasting_glucose", "hba1c"])

    if _gte(glucose, "126") or _gte(a1c, "6.5"):
        return _available("DIABETES", ServiceBand.CAUTION, "DIABETES_RANGE", "당뇨병 범위")
    if _gte(glucose, "100") or _gte(a1c, "5.7"):
        return _available("DIABETES", ServiceBand.ATTENTION, "PRE_DIABETES_RANGE", "공복혈당장애/전당뇨 범위")
    return _available("DIABETES", ServiceBand.LOW, "NORMAL", "정상 범위")


def _map_dl(features: Mapping[str, Any]) -> X2StageMappingResult:
    total = _to_decimal(features.get("total_cholesterol"))
    ldl = _to_decimal(features.get("ldl_cholesterol"))
    hdl = _to_decimal(features.get("hdl_cholesterol"))
    tg = _to_decimal(features.get("triglyceride"))
    sex = _normalize_sex(features.get("sex"))
    has_non_hdl_lipid = any(value is not None for value in (total, ldl, tg))

    if total is None and ldl is None and hdl is None and tg is None:
        return _unavailable(
            "DYSLIPIDEMIA",
            ["total_cholesterol", "ldl_cholesterol", "hdl_cholesterol", "triglyceride"],
        )
    if hdl is not None and not has_non_hdl_lipid and sex is None:
        return _unavailable("DYSLIPIDEMIA", ["sex"])

    if _gte(ldl, "190") or _gte(tg, "500"):
        return _available("DYSLIPIDEMIA", ServiceBand.HIGH_CAUTION, "HIGH_RISK_RANGE", "고위험 수치 범위")

    hdl_low = False
    if hdl is not None and sex is not None:
        hdl_low = hdl < (Decimal("40") if sex == "MALE" else Decimal("50"))

    if _gte(total, "240") or _gte(ldl, "160") or _gte(tg, "200") or hdl_low:
        return _available("DYSLIPIDEMIA", ServiceBand.ATTENTION, "MANAGEMENT_RANGE", "관리 필요 범위")
    if _gte(total, "200") or _gte(ldl, "130") or _gte(tg, "150"):
        return _available("DYSLIPIDEMIA", ServiceBand.ATTENTION, "BORDERLINE_RANGE", "경계 범위")
    return _available("DYSLIPIDEMIA", ServiceBand.LOW, "NORMAL", "정상 범위")


def _map_obe(features: Mapping[str, Any]) -> X2StageMappingResult:
    bmi = _to_decimal(features.get("bmi")) or _calculate_bmi(features.get("height_cm"), features.get("weight_kg"))
    if bmi is None:
        return _unavailable("OBESITY", ["bmi", "height_cm", "weight_kg"])

    if bmi >= 35:
        return _available("OBESITY", ServiceBand.HIGH_CAUTION, "OBESITY_STAGE_3", "비만 3단계 범위")
    if bmi >= 30:
        return _available("OBESITY", ServiceBand.HIGH_CAUTION, "OBESITY_STAGE_2", "비만 2단계 범위")
    if bmi >= 25:
        return _available("OBESITY", ServiceBand.CAUTION, "OBESITY_STAGE_1", "비만 1단계 범위")
    if bmi >= 23:
        return _available("OBESITY", ServiceBand.ATTENTION, "PRE_OBESITY", "비만 전단계 범위")
    if bmi < Decimal("18.5"):
        return _available("OBESITY", ServiceBand.LOW, "UNDERWEIGHT", "저체중 범위")
    return _available("OBESITY", ServiceBand.LOW, "NORMAL", "정상 범위")


def _map_abo(features: Mapping[str, Any]) -> X2StageMappingResult:
    waist = _to_decimal(features.get("waist_cm"))
    sex = _normalize_sex(features.get("sex"))
    missing: list[str] = []
    if waist is None:
        missing.append("waist_cm")
    if sex is None:
        missing.append("sex")
    if missing:
        return _unavailable("ABDOMINAL_OBESITY", missing)

    if sex == "MALE":
        if waist >= 95:
            return _available("ABDOMINAL_OBESITY", ServiceBand.CAUTION, "HIGH_RISK_RANGE", "고위험군")
        if waist >= 90:
            return _available("ABDOMINAL_OBESITY", ServiceBand.ATTENTION, "RISK_RANGE", "위험군")
        return _available("ABDOMINAL_OBESITY", ServiceBand.LOW, "LOW_RISK_RANGE", "저위험군")

    if waist >= 90:
        return _available("ABDOMINAL_OBESITY", ServiceBand.CAUTION, "HIGH_RISK_RANGE", "고위험군")
    if waist >= 85:
        return _available("ABDOMINAL_OBESITY", ServiceBand.ATTENTION, "RISK_RANGE", "위험군")
    return _available("ABDOMINAL_OBESITY", ServiceBand.LOW, "LOW_RISK_RANGE", "저위험군")


def _map_fl(features: Mapping[str, Any]) -> X2StageMappingResult:
    ast = _to_decimal(features.get("ast"))
    alt = _to_decimal(features.get("alt"))
    bmi = _to_decimal(features.get("bmi")) or _calculate_bmi(features.get("height_cm"), features.get("weight_kg"))
    sex = _normalize_sex(features.get("sex"))
    missing: list[str] = []
    if ast is None or ast == 0:
        missing.append("ast")
    if alt is None:
        missing.append("alt")
    if bmi is None:
        missing.append("bmi")
    if sex is None:
        missing.append("sex")
    if missing:
        return _unavailable("FATTY_LIVER", missing)

    female_bonus = Decimal("2") if sex == "FEMALE" else Decimal("0")
    hsi = Decimal("8") * (alt / ast) + bmi + female_bonus
    if hsi > 36:
        return _available("FATTY_LIVER", ServiceBand.CAUTION, "HSI_HIGH_RANGE", "HSI 36 초과 범위")
    if hsi >= 30:
        return _available("FATTY_LIVER", ServiceBand.ATTENTION, "HSI_RISK_RANGE", "HSI 30~36 범위")
    return _available("FATTY_LIVER", ServiceBand.LOW, "HSI_LOW_RANGE", "HSI 30 미만 범위")


def _map_anem(features: Mapping[str, Any]) -> X2StageMappingResult:
    hb = _to_decimal(features.get("hemoglobin"))
    sex = _normalize_sex(features.get("sex"))
    missing: list[str] = []
    if hb is None:
        missing.append("hemoglobin")
    if sex is None:
        missing.append("sex")
    if missing:
        return _unavailable("ANEMIA", missing)

    normal_cutoff = Decimal("13.0") if sex == "MALE" else Decimal("12.0")
    if hb >= normal_cutoff:
        return _available("ANEMIA", ServiceBand.LOW, "NORMAL", "정상 범위")
    if hb >= 11:
        return _available("ANEMIA", ServiceBand.ATTENTION, "MILD_RANGE", "경증 빈혈 범위")
    if hb >= 8:
        return _available("ANEMIA", ServiceBand.CAUTION, "MODERATE_RANGE", "중등도 빈혈 범위")
    return _available("ANEMIA", ServiceBand.HIGH_CAUTION, "SEVERE_RANGE", "중증 빈혈 범위")


def _map_lf(features: Mapping[str, Any]) -> X2StageMappingResult:
    ast = _to_decimal(features.get("ast"))
    alt = _to_decimal(features.get("alt"))
    gtp = _to_decimal(features.get("gamma_gtp"))
    sex = _normalize_sex(features.get("sex"))
    has_ast_or_alt = ast is not None or alt is not None

    if ast is None and alt is None and gtp is None:
        return _unavailable("LIVER_FUNCTION", ["ast", "alt", "gamma_gtp"])
    if gtp is not None and not has_ast_or_alt and sex is None:
        return _unavailable("LIVER_FUNCTION", ["sex"])

    abnormal = _gt(ast, "40") or _gt(alt, "40")
    if gtp is not None and sex is not None:
        abnormal = abnormal or gtp > (Decimal("63") if sex == "MALE" else Decimal("35"))

    if abnormal:
        return _available("LIVER_FUNCTION", ServiceBand.ATTENTION, "ABNORMAL_SUSPECTED", "간기능 이상 의심")
    return _available("LIVER_FUNCTION", ServiceBand.LOW, "NORMAL", "정상 범위")


def _map_kf(features: Mapping[str, Any]) -> X2StageMappingResult:
    protein_positive = _parse_urine_protein(features.get("urine_protein"))
    creatinine = _to_decimal(features.get("creatinine"))
    egfr = _to_decimal(features.get("egfr"))
    if features.get("urine_protein") is None and creatinine is None and egfr is None:
        return _unavailable("KIDNEY_FUNCTION", ["urine_protein", "creatinine", "egfr"])
    if features.get("urine_protein") is not None and protein_positive is None:
        return _unavailable("KIDNEY_FUNCTION", ["urine_protein"])

    if protein_positive is True or _gt(creatinine, "1.5") or (egfr is not None and egfr <= 60):
        return _available("KIDNEY_FUNCTION", ServiceBand.ATTENTION, "DAMAGE_SUSPECTED", "신기능 손상 의심")
    return _available("KIDNEY_FUNCTION", ServiceBand.LOW, "NORMAL", "정상 범위")


def _map_ckd(features: Mapping[str, Any]) -> X2StageMappingResult:
    egfr = _to_decimal(features.get("egfr"))
    if egfr is None:
        return _unavailable("CHRONIC_KIDNEY_DISEASE", ["egfr"])
    if egfr < 30:
        return _available(
            "CHRONIC_KIDNEY_DISEASE", ServiceBand.HIGH_CAUTION, "EGFR_UNDER_30", "eGFR 30 미만, 상담 권장"
        )
    if egfr < 45:
        return _available("CHRONIC_KIDNEY_DISEASE", ServiceBand.CAUTION, "EGFR_30_44", "eGFR 30~44, 추적 관찰 필요")
    if egfr < 60:
        return _available("CHRONIC_KIDNEY_DISEASE", ServiceBand.ATTENTION, "EGFR_45_59", "eGFR 45~59, 추적 관찰 필요")
    if egfr < 90:
        return _available("CHRONIC_KIDNEY_DISEASE", ServiceBand.ATTENTION, "EGFR_60_89", "eGFR 60~89, 추적 관찰 필요")
    return _available("CHRONIC_KIDNEY_DISEASE", ServiceBand.LOW, "EGFR_90_PLUS", "eGFR 90 이상")


def _available(
    analysis_type: str,
    risk_level: ServiceBand,
    x2_stage_code: str,
    x2_stage_label: str,
) -> X2StageMappingResult:
    return X2StageMappingResult(
        analysis_type=analysis_type,
        risk_level=risk_level.value,
        result_source=X2ResultSource.X2_RULE.value,
        x2_stage_code=x2_stage_code,
        x2_stage_label=x2_stage_label,
        x2_available=True,
        x2_missing_fields=[],
    )


def _unavailable(analysis_type: str, missing_fields: list[str]) -> X2StageMappingResult:
    return X2StageMappingResult(
        analysis_type=analysis_type,
        risk_level=None,
        result_source=X2ResultSource.X2_UNAVAILABLE.value,
        x2_stage_code=None,
        x2_stage_label=None,
        x2_available=False,
        x2_missing_fields=missing_fields,
    )


def _normalize_analysis_type(analysis_type: str) -> str:
    normalized = str(analysis_type).strip().upper()
    return _DISEASE_ALIASES.get(normalized, normalized)


def _normalize_features(features: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in features.items():
        mapped_key = SOURCE_VARIABLE_MAP.get(str(key), str(key))
        if mapped_key == "gender":
            mapped_key = "sex"
        elif mapped_key == "ldl":
            mapped_key = "ldl_cholesterol"
        elif mapped_key == "hdl":
            mapped_key = "hdl_cholesterol"
        normalized[mapped_key] = value
    if "gender" in normalized and "sex" not in normalized:
        normalized["sex"] = normalized["gender"]
    return normalized


def _normalize_sex(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    if normalized in {"MALE", "M", "MAN", "남", "남성", "1"}:
        return "MALE"
    if normalized in {"FEMALE", "F", "WOMAN", "여", "여성", "2"}:
        return "FEMALE"
    return None


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _calculate_bmi(height_cm: Any, weight_kg: Any) -> Decimal | None:
    height = _to_decimal(height_cm)
    weight = _to_decimal(weight_kg)
    if height is None or weight is None or height <= 0 or weight <= 0:
        return None
    height_m = height / Decimal("100")
    return weight / (height_m * height_m)


def _parse_urine_protein(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    token = str(value).strip().upper().replace(" ", "")
    if token in {item.upper() for item in _URINE_PROTEIN_POSITIVE}:
        return True
    if token in {item.upper() for item in _URINE_PROTEIN_NEGATIVE}:
        return False
    return None


def _missing_numeric_fields(values: Mapping[str, Decimal | None]) -> list[str]:
    return [key for key, value in values.items() if value is None]


def _gte(value: Decimal | None, threshold: str) -> bool:
    return value is not None and value >= Decimal(threshold)


def _gt(value: Decimal | None, threshold: str) -> bool:
    return value is not None and value > Decimal(threshold)
