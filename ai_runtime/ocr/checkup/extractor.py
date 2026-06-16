"""
ai_runtime/ocr/checkup/extractor.py
PaddleOCR 2.7.3 기반 건강검진표 수치 추출 모듈.
이미지 및 PDF(텍스트/스캔) 모두 지원.
"""

import logging
import re

try:
    from paddleocr import PaddleOCR as _PaddleOCR
except ModuleNotFoundError:
    _PaddleOCR = None

from .pdf_handler import PdfType, detect_pdf_type, extract_text_from_pdf, parse_text_lines, pdf_to_images
from .preprocessor import preprocess_for_ocr
from .schemas import CheckupOcrData, OcrStatus

logger = logging.getLogger(__name__)

# ── PaddleOCR 싱글톤 ──────────────────────────────────────────────────────────
PaddleOCR = _PaddleOCR
_ocr_engine = None


class CheckupOcrDependencyError(RuntimeError):
    """Raised when optional OCR dependencies are not installed."""


def _get_paddle_ocr_class():
    if PaddleOCR is None:
        msg = "paddleocr is required for checkup OCR. Install OCR dependencies before running OCR."
        raise CheckupOcrDependencyError(msg)
    return PaddleOCR


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("PaddleOCR 모델 로딩 중...")
        ocr_cls = _get_paddle_ocr_class()
        try:
            _ocr_engine = ocr_cls(
                lang="korean",
                use_textline_orientation=True,
            )
        except ValueError:
            _ocr_engine = ocr_cls(
                use_angle_cls=True,
                lang="korean",
            )
        logger.info("PaddleOCR 모델 로딩 완료")
    return _ocr_engine


# ── 키워드 매핑 ───────────────────────────────────────────────────────────────

FIELD_KEYWORDS = {
    "systolic_bp": ["수축기", "고혈압", "혈압", "mmHg", "SBP"],
    "diastolic_bp": ["이완기", "DBP"],
    "fasting_glucose": ["공복혈당", "혈당", "공복", "GLU", "Glucose"],
    "hba1c": ["당화혈색소", "HbA1c", "A1c"],
    "hb": ["혈색소", "헤모글로빈"],  # Hb 단독 키워드는 HbA1c와 혼동 위험으로 제외
    "hemoglobin": ["혈색소", "헤모글로빈"],
    "total_cholesterol": ["총콜레스테롤", "콜레스테롤", "TC", "T-CHO", "CHOL"],
    "triglyceride": ["중성지방", "TG", "Triglyceride"],
    "hdl": ["고밀도", "HDL"],
    "ldl": ["저밀도", "LDL"],
    "ast": ["AST", "GOT", "SGOT"],
    "alt": ["ALT", "GPT", "SGPT"],
    "gamma_gtp": ["감마GTP", "감마지티피", "γ-GTP", "r-GTP", "GGT", "GGTP"],
    "creatinine": ["크레아티닌", "Creatinine", "Cr", "혈청크레아티닌"],
    "egfr": ["eGFR", "사구체여과율", "신사구체여과율"],
    "height_cm": ["신장", "키(cm)", "키", "Height", "HT"],
    "weight_kg": ["체중", "몸무게(kg)", "몸무게", "Weight", "WT"],
    "bmi": ["BMI", "체질량", "비만도", "체질량지수"],
    "waist_cm": ["허리둘레(cm)", "허리둘레", "허리", "복부둘레", "Waist"],
}

# hb 유효 범위 (g/dL): 남 13~17, 여 12~16. 넉넉하게 설정
HB_RANGE = (5.0, 25.0)

CONFIDENCE_THRESHOLD = 0.7
CHECKBOX_PATTERN = re.compile(r"[■□▣▪●○◆◇]")
NOT_APPLICABLE_KEYWORDS = ["비해당", "해당없음"]
MEASUREMENT_PAGE_KEYWORDS = {
    "검사항목": 3,
    "참고치": 3,
    "결과": 1,
    "공복혈당": 4,
    "총콜레스테롤": 4,
    "중성지방": 4,
    "HDL": 3,
    "LDL": 3,
    "AST": 2,
    "ALT": 2,
    "혈색소": 3,
    "수축기": 2,
    "이완기": 2,
    "혈압": 2,
    "신장": 2,
    "체중": 2,
    "허리둘레": 2,
    "BMI": 2,
}
MEASUREMENT_PAGE_MIN_SCORE = 6


def clean_text(text):
    return CHECKBOX_PATTERN.sub("", text).strip()


def is_not_applicable(text):
    return any(kw in text for kw in NOT_APPLICABLE_KEYWORDS)


def extract_numbers(text):
    text = clean_text(text).replace(",", "")
    matches = re.findall(r"\d+\.?\d*", text)
    results = []
    for m in matches:
        try:
            results.append(float(m))
        except ValueError:
            continue
    return results


def extract_first_number(text):
    numbers = extract_numbers(text)
    return numbers[0] if numbers else None


def is_keyword_match(text, keywords):
    text_upper = text.upper()
    return any(kw.upper() in text_upper for kw in keywords)


def validate_value(field, value):
    ranges = {
        "systolic_bp": (60, 250),
        "diastolic_bp": (40, 150),
        "fasting_glucose": (40, 600),
        "hba1c": (3.0, 20.0),
        "hb": (5.0, 25.0),
        "hemoglobin": (5.0, 25.0),
        "total_cholesterol": (50, 600),
        "triglyceride": (20, 2000),
        "hdl": (10, 200),
        "ldl": (20, 500),
        "ast": (1, 1000),
        "alt": (1, 1000),
        "gamma_gtp": (1, 2000),
        "creatinine": (0.1, 20),
        "egfr": (1, 200),
        "height_cm": (100, 250),
        "weight_kg": (20, 300),
        "bmi": (10, 70),
        "waist_cm": (40, 200),
    }
    if field not in ranges:
        return True
    low, high = ranges[field]
    return low <= value <= high


def calculate_bmi(height_cm, weight_kg) -> float | None:
    """키(cm)와 몸무게(kg)로 BMI를 계산합니다."""
    try:
        h = float(height_cm)
        w = float(weight_kg)
        if h > 0 and w > 0:
            return round(w / ((h / 100) ** 2), 1)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


def parse_blood_pressure(text_lines):
    for text, _ in text_lines:
        if is_keyword_match(text, FIELD_KEYWORDS["systolic_bp"]):
            bp_candidates = [n for n in extract_numbers(text) if 40 <= n <= 250]
            if len(bp_candidates) >= 2:
                first, second = bp_candidates[:2]
                return max(first, second), min(first, second)
            if len(bp_candidates) == 1:
                return bp_candidates[0], None
    return None, None


def parse_height_weight(text_lines):
    """
    키와 몸무게를 파싱합니다.
    같은 줄 또는 이후 줄에서 순서대로 탐색합니다.
    """
    for i, (text, _) in enumerate(text_lines):
        if is_keyword_match(text, FIELD_KEYWORDS["height_cm"]):
            # 현재 줄 + 이후 6줄 안에서 키/몸무게 탐색
            context = text_lines[i : i + 7]
            all_nums = []
            for ctx_text, _ in context:
                all_nums.extend(extract_numbers(ctx_text))
            candidates_h = [n for n in all_nums if 100 <= n <= 250]
            if not candidates_h:
                continue
            height = candidates_h[0]
            height_index = all_nums.index(height)
            weight_pool = all_nums[height_index + 1 :] or all_nums
            candidates_w = [n for n in weight_pool if 20 <= n <= 150 and n != height and n < height]
            if not candidates_w:
                candidates_w = [n for n in weight_pool if 20 <= n <= 150 and n != height]
            if candidates_w:
                return height, candidates_w[0]
    return None, None


def parse_hb(text_lines) -> tuple[float | None, float | None]:
    """
    혈색소(Hb) 수치를 파싱합니다.
    당화혈색소(HbA1c)와 혼동하지 않도록 "당화" 포함 줄은 제외합니다.
    """
    for i, (text, conf) in enumerate(text_lines):
        # 당화혈색소 줄은 스킵
        if "당화" in text:
            continue
        if not is_keyword_match(text, FIELD_KEYWORDS["hb"]):
            continue
        # 같은 줄에서 유효한 숫자 추출
        nums = [n for n in extract_numbers(text) if HB_RANGE[0] <= n <= HB_RANGE[1]]
        if nums:
            return nums[0], conf
        # 다음 2줄 탐색
        for j in range(i + 1, min(i + 3, len(text_lines))):
            next_text, next_conf = text_lines[j]
            if "당화" in next_text:
                break
            nums = [n for n in extract_numbers(next_text) if HB_RANGE[0] <= n <= HB_RANGE[1]]
            if nums:
                return nums[0], next_conf
    return None, None


def parse_bmi(text_lines) -> float | None:
    """
    BMI 실측 수치만 추출합니다.
    범위 표현(18.5미만, 18.5~24.9 등)은 제거 후 독립 수치만 추출합니다.
    없으면 None 반환 → 키·몸무게로 자체 계산.
    """
    range_pattern = re.compile(r"\d+\.?\d*\s*[-~]\s*\d+\.?\d*")
    bound_pattern = re.compile(r"\d+\.?\d*\s*(미만|이상|이하|초과)")

    for i, (text, _) in enumerate(text_lines):
        if not is_keyword_match(text, FIELD_KEYWORDS["bmi"]):
            continue
        cleaned = range_pattern.sub("", text)
        cleaned = bound_pattern.sub("", cleaned)
        nums = [n for n in extract_numbers(cleaned) if 10 <= n <= 70]
        if nums:
            return nums[0]
        if i + 1 < len(text_lines):
            next_text = text_lines[i + 1][0]
            next_cleaned = range_pattern.sub("", next_text)
            next_cleaned = bound_pattern.sub("", next_cleaned)
            nums = [n for n in extract_numbers(next_cleaned) if 10 <= n <= 70]
            if nums:
                return nums[0]
    return None


def _extract_value_from_context(field, text_lines, i, text):
    value = next((number for number in extract_numbers(text) if validate_value(field, number)), None)
    confidence = text_lines[i][1]
    if value is None:
        for j in range(i + 1, min(i + 3, len(text_lines))):
            value = next(
                (number for number in extract_numbers(text_lines[j][0]) if validate_value(field, number)), None
            )
            if value is not None:
                confidence = text_lines[j][1]
                break
    return value, confidence


DYSLIPIDEMIA_FIELDS = {"total_cholesterol", "triglyceride", "hdl", "ldl"}


def _parse_general_fields(text_lines, extracted, skip_fields, low_conf):
    for i, (text, _) in enumerate(text_lines):
        for field, keywords in FIELD_KEYWORDS.items():
            if field in skip_fields or extracted.get(field) is not None:
                continue
            if not is_keyword_match(text, keywords):
                continue

            # 이상지질혈증 4개 필드는 비해당 먼저 체크
            if field in DYSLIPIDEMIA_FIELDS:
                # 현재 줄 + 다음 2줄 안에 비해당 있으면 비해당으로 처리
                check_lines = [text] + [text_lines[j][0] for j in range(i + 1, min(i + 3, len(text_lines)))]
                if any(is_not_applicable(line) for line in check_lines):
                    continue

            value, confidence = _extract_value_from_context(field, text_lines, i, text)
            if value is not None and validate_value(field, value):
                extracted[field] = value
                if confidence < CONFIDENCE_THRESHOLD:
                    low_conf.append(field)
                logger.info("필드 추출 | %s = %s (신뢰도: %.2f)", field, value, confidence)


def parse_from_text_lines(text_lines):
    raw_texts = [t for t, _ in text_lines]
    extracted: dict = {f: None for f in FIELD_KEYWORDS}
    low_conf = []

    # 혈압
    systolic, diastolic = parse_blood_pressure(text_lines)
    if systolic and validate_value("systolic_bp", systolic):
        extracted["systolic_bp"] = systolic
    if diastolic and validate_value("diastolic_bp", diastolic):
        extracted["diastolic_bp"] = diastolic

    # 키/몸무게
    height, weight = parse_height_weight(text_lines)
    if height and validate_value("height_cm", height):
        extracted["height_cm"] = height
    if weight and validate_value("weight_kg", weight):
        extracted["weight_kg"] = weight

    # 혈색소 (hb) — 전용 파서 사용
    hb, hb_confidence = parse_hb(text_lines)
    if hb is not None:
        extracted["hb"] = hb
        if hb_confidence is not None and hb_confidence < CONFIDENCE_THRESHOLD:
            low_conf.append("hb")

    skip_fields = {"systolic_bp", "diastolic_bp", "height_cm", "weight_kg", "hb", "hemoglobin", "bmi"}
    _parse_general_fields(text_lines, extracted, skip_fields, low_conf)

    # BMI 전용 파서 — 범위 표현 제거 후 실측값만 추출
    bmi = parse_bmi(text_lines)
    if bmi is not None:
        extracted["bmi"] = bmi
        # BMI 교차검증: 키/몸무게로 계산한 값과 비교
        if extracted.get("height_cm") and extracted.get("weight_kg"):
            calculated = calculate_bmi(extracted["height_cm"], extracted["weight_kg"])
            if calculated is not None:
                diff = abs(bmi - calculated)
                if diff > 1.0:  # 1.0 이상 차이나면 계산값으로 대체
                    logger.warning("BMI 교차검증 불일치 | 추출=%.1f 계산=%.1f → 계산값 사용", bmi, calculated)
                    extracted["bmi"] = calculated

    # ── BMI 처리 ──────────────────────────────────────────────────────────────
    # 검진표에 BMI 수치 있으면 그대로, 없으면 키·몸무게로 자체 계산
    bmi_calculated = False
    if extracted.get("bmi") is None and extracted.get("height_cm") and extracted.get("weight_kg"):
        calculated = calculate_bmi(extracted["height_cm"], extracted["weight_kg"])
        if calculated is not None:
            extracted["bmi"] = calculated
            bmi_calculated = True
            logger.info(
                "BMI 자체 계산 | height=%.1f weight=%.1f bmi=%.1f",
                extracted["height_cm"],
                extracted["weight_kg"],
                calculated,
            )

    data = CheckupOcrData(**extracted, bmi_calculated=bmi_calculated)
    return data, low_conf, raw_texts


def _float_or_default(value, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _append_ocr_line(text_lines: list[tuple[str, float]], text, confidence=1.0) -> None:
    if not isinstance(text, str):
        return
    cleaned = text.strip()
    if cleaned:
        text_lines.append((cleaned, _float_or_default(confidence)))


def _as_mapping(value):
    if isinstance(value, dict):
        return value

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            mapped = to_dict()
        except TypeError:
            mapped = None
        if isinstance(mapped, dict):
            return mapped

    json_value = getattr(value, "json", None)
    if callable(json_value):
        try:
            json_value = json_value()
        except TypeError:
            json_value = None
    if isinstance(json_value, dict):
        return json_value

    return None


def _collect_ocr_lines(result, text_lines: list[tuple[str, float]]) -> None:
    mapping = _as_mapping(result)
    if mapping is not None:
        rec_texts = mapping.get("rec_texts") or mapping.get("texts")
        if isinstance(rec_texts, list):
            rec_scores = mapping.get("rec_scores") or mapping.get("scores") or []
            for index, text in enumerate(rec_texts):
                confidence = rec_scores[index] if index < len(rec_scores) else 1.0
                _append_ocr_line(text_lines, text, confidence)
            return

        text = mapping.get("text") or mapping.get("rec_text")
        if text is not None:
            confidence = mapping.get("confidence") or mapping.get("score") or mapping.get("rec_score") or 1.0
            _append_ocr_line(text_lines, text, confidence)
            return

        for value in mapping.values():
            if isinstance(value, (dict, list, tuple)) or _as_mapping(value) is not None:
                _collect_ocr_lines(value, text_lines)
        return

    if isinstance(result, (list, tuple)):
        if len(result) >= 2:
            candidate = result[1]
            if isinstance(candidate, (list, tuple)) and candidate and isinstance(candidate[0], str):
                confidence = candidate[1] if len(candidate) > 1 else 1.0
                _append_ocr_line(text_lines, candidate[0], confidence)
                return
            if isinstance(result[0], str):
                _append_ocr_line(text_lines, result[0], result[1])
                return

        for item in result:
            _collect_ocr_lines(item, text_lines)


def _normalize_paddle_ocr_result(results) -> list[tuple[str, float]]:
    text_lines: list[tuple[str, float]] = []
    _collect_ocr_lines(results, text_lines)
    return text_lines


def _run_paddle_ocr(engine, processed):
    attempts = [
        ("ocr", {}),
        ("predict", {}),
        ("ocr", {"cls": True}),
    ]
    last_type_error: TypeError | None = None
    for method_name, kwargs in attempts:
        method = getattr(engine, method_name, None)
        if not callable(method):
            continue
        try:
            return method(processed, **kwargs)
        except TypeError as exc:
            last_type_error = exc
            logger.debug("PaddleOCR 호출 방식 불일치 | method=%s kwargs=%s error=%s", method_name, kwargs, exc)
            continue

    if last_type_error is not None:
        raise last_type_error
    msg = "PaddleOCR engine does not provide an ocr or predict method."
    raise RuntimeError(msg)


def run_ocr_on_image(image_bytes):
    processed = preprocess_for_ocr(image_bytes)
    engine = get_ocr_engine()
    results = _run_paddle_ocr(engine, processed)
    text_lines = _normalize_paddle_ocr_result(results)

    logger.info("OCR 인식 %d줄", len(text_lines))
    return text_lines


def determine_status(data, low_conf):
    extracted_count = sum(1 for v in data.model_dump().values() if v is not None and v is not False)
    if extracted_count == 0:
        return OcrStatus.FAILED
    if low_conf or extracted_count < 4:
        return OcrStatus.PARTIAL
    return OcrStatus.SUCCESS


async def run_ocr(image_bytes):
    text_lines = run_ocr_on_image(image_bytes)
    data, low_conf, raw = parse_from_text_lines(text_lines)
    status = determine_status(data, low_conf)
    return data, low_conf, raw, status


async def run_ocr_on_pdf(pdf_bytes):
    pdf_type = detect_pdf_type(pdf_bytes)
    page_text_lines: list[list[tuple[str, float]]] = []

    if pdf_type == PdfType.TEXT:
        logger.info("텍스트 PDF 처리 중...")
        texts = extract_text_from_pdf(pdf_bytes)
        page_text_lines = [[(line, 1.0) for line in parse_text_lines([text])] for text in texts]
    else:
        logger.info("스캔 PDF 처리 중...")
        image_bytes_list = pdf_to_images(pdf_bytes)
        if not image_bytes_list:
            logger.error("PDF 이미지 변환 실패")
            return CheckupOcrData(), [], [], OcrStatus.FAILED
        for i, img_bytes in enumerate(image_bytes_list):
            logger.info("페이지 %d OCR 실행 중...", i + 1)
            page_lines = run_ocr_on_image(img_bytes)
            page_text_lines.append(page_lines)

    measurement_lines = select_measurement_page_lines(page_text_lines)
    data, low_conf, raw = parse_from_text_lines(measurement_lines)
    status = determine_status(data, low_conf)
    if status == OcrStatus.FAILED and measurement_lines != flatten_page_lines(page_text_lines):
        logger.info("측정값 페이지 우선 추출 실패, 전체 PDF 텍스트 fallback")
        data, low_conf, raw = parse_from_text_lines(flatten_page_lines(page_text_lines))
        status = determine_status(data, low_conf)
    return data, low_conf, raw, status


def flatten_page_lines(page_text_lines: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    return [line for page_lines in page_text_lines for line in page_lines]


def score_measurement_page(text_lines: list[tuple[str, float]]) -> int:
    text = "\n".join(line for line, _ in text_lines).upper()
    return sum(weight * text.count(keyword.upper()) for keyword, weight in MEASUREMENT_PAGE_KEYWORDS.items())


def select_measurement_page_lines(page_text_lines: list[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    if not page_text_lines:
        return []
    scores = [score_measurement_page(page_lines) for page_lines in page_text_lines]
    max_score = max(scores, default=0)
    if max_score < MEASUREMENT_PAGE_MIN_SCORE:
        return flatten_page_lines(page_text_lines)
    selected_pages = [
        page_lines for page_lines, score in zip(page_text_lines, scores, strict=True) if score == max_score
    ]
    return flatten_page_lines(selected_pages)
