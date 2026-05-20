import re
from typing import Any

VALUE_FIELDS = [
    "exam_date",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "systolic_bp",
    "diastolic_bp",
    "hemoglobin",
    "fasting_glucose",
    "total_cholesterol",
    "triglyceride",
    "hdl",
    "ldl",
    "creatinine",
    "egfr",
    "ast",
    "alt",
    "gamma_gtp",
    "urine_protein",
    "chest_xray",
    "suspected_diseases",
    "lifestyle_smoking",
    "lifestyle_drinking",
    "lifestyle_physical_activity",
    "lifestyle_strength_training",
]

NUMERIC_FIELDS = {
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "systolic_bp",
    "diastolic_bp",
    "hemoglobin",
    "fasting_glucose",
    "total_cholesterol",
    "triglyceride",
    "hdl",
    "ldl",
    "creatinine",
    "egfr",
    "ast",
    "alt",
    "gamma_gtp",
}

NOT_APPLICABLE = "비해당"


def empty_result() -> dict[str, Any]:
    return {field: None for field in VALUE_FIELDS}


def parse_health_exam_result(text: str) -> dict[str, Any]:
    """Parse key health exam values from OCR text.

    This is a rule-based parser for CLOVA OCR PoC output. It is intentionally
    conservative: value-level accuracy must be measured against human-written
    ground truth JSON, not inferred from OCR confidence.
    """

    lines = _clean_lines(text)
    compact = _compact_text(lines)
    result = empty_result()

    result["exam_date"] = _parse_exam_date(compact)

    height_weight = _parse_height_weight(compact)
    result.update(height_weight)

    result["bmi"] = _find_number_near_alias(lines, ["체질량지수", "bmi"], min_value=10, max_value=80)
    result["waist_cm"] = _find_number_near_alias(lines, ["허리둘레", "허리 둘레"], min_value=40, max_value=200)

    bp = _parse_blood_pressure(lines, compact)
    result["systolic_bp"] = bp[0]
    result["diastolic_bp"] = bp[1]

    result["hemoglobin"] = _find_number_near_alias(lines, ["혈색소", "헤모글로빈"], min_value=5, max_value=25)
    result["fasting_glucose"] = _find_number_near_alias(lines, ["공복혈당", "공복 혈당"], min_value=40, max_value=500)
    result["total_cholesterol"] = _find_lipid_value(
        lines, ["총콜레스테롤", "총 콜레스테롤"], min_value=50, max_value=500
    )
    result["triglyceride"] = _find_lipid_value(
        lines, ["중성지방", "트리글리세라이드", "triglyceride"], min_value=20, max_value=1000
    )
    result["hdl"] = _find_lipid_value(lines, ["hdl", "고밀도", "고밀도 콜레스테롤"], min_value=10, max_value=200)
    result["ldl"] = _find_lipid_value(
        lines, ["ldl", "저밀도", "저밀도 콜레스테롤", "저밀도 콜레스트롤"], min_value=10, max_value=300
    )
    result["creatinine"] = _find_number_near_alias(
        lines, ["혈청크레아티닌", "크레아티닌", "creatinine"], min_value=0.1, max_value=20
    )
    result["egfr"] = _find_number_near_alias(
        lines, ["egfr", "e-gfr", "신사구체여과율", "사구체여과율"], min_value=1, max_value=200
    )
    result["ast"] = _find_number_near_alias(lines, ["ast", "sgot"], min_value=1, max_value=1000)
    result["alt"] = _find_number_near_alias(lines, ["alt", "sgpt"], min_value=1, max_value=1000)
    result["gamma_gtp"] = _find_number_near_alias(
        lines, ["감마지티피", "g-gtp", "γ-gtp", "&-gtp", "감마gtp"], min_value=1, max_value=2000
    )

    result["urine_protein"] = _find_status_after_alias(lines, ["요단백", "요 단백"], ["정상", "경계", "단백뇨 의심"])
    result["chest_xray"] = _find_status_after_alias(
        lines, ["흉부촬영", "흉부 촬영", "흉부x선"], ["정상", "비활동성 폐결핵", "질환의심"]
    )
    result["suspected_diseases"] = _parse_suspected_diseases(compact)
    result["lifestyle_smoking"] = _parse_lifestyle_need(
        compact,
        need_aliases=["금연필요", "금연 필요"],
        normal_aliases=["비흡연", "금연유지"],
    )
    result["lifestyle_drinking"] = _parse_lifestyle_need(
        compact,
        need_aliases=["절주필요", "절주 필요", "금주필요", "금주 필요"],
        normal_aliases=["음주정상", "정상음주"],
    )
    result["lifestyle_physical_activity"] = _parse_lifestyle_need(
        compact,
        need_aliases=["신체활동필요", "신체 활동 필요", "운동필요", "운동 필요"],
        normal_aliases=["신체활동정상", "운동정상"],
    )
    result["lifestyle_strength_training"] = _parse_lifestyle_need(
        compact,
        need_aliases=["근력운동필요", "근력 운동 필요"],
        normal_aliases=["근력운동정상"],
    )

    return _coerce_numeric_fields(result)


def _clean_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def _compact_text(lines: list[str]) -> str:
    return "\n".join(lines)


def _parse_exam_date(text: str) -> str | None:
    match = re.search(r"검진일\s*[:\n ]*\s*(\d{4})[.\-/년 ]+(\d{1,2})[.\-/월 ]+(\d{1,2})", text)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _parse_height_weight(text: str) -> dict[str, float | None]:
    result = {"height_cm": None, "weight_kg": None}
    pattern = re.compile(r"(?P<height>1[3-9]\d(?:\.\d+)?)\s*/\s*(?P<weight>\d{2,3}(?:\.\d+)?)")
    for match in pattern.finditer(text):
        height = float(match.group("height"))
        weight = float(match.group("weight"))
        if 130 <= height <= 220 and 20 <= weight <= 250:
            result["height_cm"] = height
            result["weight_kg"] = weight
            return result
    return result


def _parse_blood_pressure(lines: list[str], compact: str) -> tuple[int | None, int | None]:
    pattern = re.compile(r"(?P<sys>\d{2,3})\s*/\s*(?P<dia>\d{2,3})")
    for match in pattern.finditer(compact):
        sys = int(match.group("sys"))
        dia = int(match.group("dia"))
        if 70 <= sys <= 250 and 40 <= dia <= 150:
            window = compact[max(0, match.start() - 120) : match.end() + 120]
            if any(alias in _normalize(window) for alias in ["혈압", "수축기", "이완기", "mmhg"]):
                return sys, dia

    value = _find_number_near_alias(lines, ["수축기", "혈압"], min_value=70, max_value=250)
    next_value = _find_number_after_number(lines, value, min_value=40, max_value=150) if value else None
    return _to_int(value), _to_int(next_value)


def _find_number_near_alias(
    lines: list[str],
    aliases: list[str],
    *,
    min_value: float,
    max_value: float,
    window: int = 12,
) -> int | float | None:
    normalized_aliases = [_normalize(alias) for alias in aliases]
    for index, line in enumerate(lines):
        if not any(alias in _normalize(line) for alias in normalized_aliases):
            continue
        candidates = lines[index : index + window]
        for candidate in candidates:
            number = _first_number(candidate)
            if number is not None and min_value <= number <= max_value:
                return _normalize_number(number)
    return None


def _find_lipid_value(
    lines: list[str],
    aliases: list[str],
    *,
    min_value: float,
    max_value: float,
    window: int = 12,
) -> int | float | str | None:
    normalized_aliases = [_normalize(alias) for alias in aliases]
    for index, line in enumerate(lines):
        if not any(alias in _normalize(line) for alias in normalized_aliases):
            continue

        for candidate in lines[index : index + window]:
            if NOT_APPLICABLE in _normalize(candidate):
                return NOT_APPLICABLE

            for match in _number_matches(candidate):
                number = float(match.group(1))
                if not min_value <= number <= max_value:
                    continue
                if _is_reference_number(candidate, match):
                    continue
                return _normalize_number(number)

    return None


def _find_number_after_number(
    lines: list[str], first_value: int | float | None, *, min_value: float, max_value: float
) -> int | float | None:
    if first_value is None:
        return None
    seen_first = False
    for line in lines:
        number = _first_number(line)
        if number is None:
            continue
        if not seen_first and abs(number - float(first_value)) < 0.01:
            seen_first = True
            continue
        if seen_first and min_value <= number <= max_value:
            return _normalize_number(number)
    return None


def _find_status_after_alias(lines: list[str], aliases: list[str], statuses: list[str], window: int = 10) -> str | None:
    normalized_aliases = [_normalize(alias) for alias in aliases]
    for index, line in enumerate(lines):
        if not any(alias in _normalize(line) for alias in normalized_aliases):
            continue
        haystack = _normalize(" ".join(lines[index : index + window]))
        for status in statuses:
            if _normalize(status) in haystack:
                return status
    return None


def _parse_suspected_diseases(text: str) -> list[str] | None:
    match = re.search(r"의심\s*질환\s*[:：]\s*([^\n〉>]+)", text)
    if not match:
        return None
    diseases = [item.strip(" ,./") for item in re.split(r"[,，/]", match.group(1)) if item.strip(" ,./")]
    return diseases or None


def _parse_lifestyle_need(text: str, need_aliases: list[str], normal_aliases: list[str]) -> bool | None:
    normalized_text = _normalize(text)
    if any(_normalize(alias) in normalized_text for alias in need_aliases):
        return True
    if any(_normalize(alias) in normalized_text for alias in normal_aliases):
        return False
    return None


def _first_number(text: str) -> float | None:
    match = re.search(r"(?<!\d)(\d{1,4}(?:\.\d+)?)(?!\d)", text)
    if not match:
        return None
    return float(match.group(1))


def _number_matches(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"(?<!\d)(\d{1,4}(?:\.\d+)?)(?!\d)", text))


def _is_reference_number(text: str, match: re.Match[str]) -> bool:
    tail = text[match.end() : match.end() + 8]
    return bool(re.match(r"\s*(미만|이상|이하|초과)", tail))


def _normalize_number(value: float) -> int | float:
    if abs(value - round(value)) < 0.000001:
        return int(round(value))
    return value


def _coerce_numeric_fields(result: dict[str, Any]) -> dict[str, Any]:
    for field in NUMERIC_FIELDS:
        value = result.get(field)
        if value is None:
            continue
        if not isinstance(value, int | float):
            continue
        if field in {
            "systolic_bp",
            "diastolic_bp",
            "fasting_glucose",
            "total_cholesterol",
            "triglyceride",
            "hdl",
            "ldl",
            "ast",
            "alt",
            "gamma_gtp",
        }:
            result[field] = _to_int(value)
        else:
            result[field] = float(value)
    return result


def _to_int(value: int | float | None) -> int | None:
    if value is None:
        return None
    return int(round(float(value)))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text).lower())
