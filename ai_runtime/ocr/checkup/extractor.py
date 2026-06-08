"""
ai_runtime/ocr/checkup/extractor.py
PaddleOCR 2.7.3 기반 건강검진표 수치 추출 모듈.
이미지 및 PDF(텍스트/스캔) 모두 지원.
"""

import logging
import re

from paddleocr import PaddleOCR

from .pdf_handler import PdfType, detect_pdf_type, extract_text_from_pdf, extract_text_from_health_check_pdf, parse_text_lines, pdf_to_images
from .preprocessor import preprocess_for_ocr
from .schemas import CheckupOcrData, OcrStatus

logger = logging.getLogger(__name__)

# ── PaddleOCR 싱글톤 ──────────────────────────────────────────────────────────

_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        logger.info("PaddleOCR 모델 로딩 중...")
        _ocr_engine = PaddleOCR(
            use_angle_cls=False,
            lang="korean",
            enable_mkldnn=False
        )
        logger.info("PaddleOCR 모델 로딩 완료")
    return _ocr_engine


# ── 키워드 매핑 ───────────────────────────────────────────────────────────────

FIELD_KEYWORDS = {
    "systolic_bp": ["수축기", "고혈압", "혈압", "mmHg", "SBP"],
    "diastolic_bp": ["이완기", "DBP"],
    "fasting_glucose": ["공복혈당", "혈당", "공복", "GLU", "Glucose"],
    "hb": ["혈색소", "헤모글로빈"],  # Hb 단독 키워드는 HbA1c와 혼동 위험으로 제외
    "total_cholesterol": ["총콜레스테롤", "콜레스테롤", "TC", "T-CHO", "CHOL"],
    "triglyceride": ["중성지방", "TG", "Triglyceride"],
    "hdl": ["고밀도", "HDL"],
    "ldl": ["저밀도", "LDL"],
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
        "hb": (5.0, 25.0),
        "total_cholesterol": (50, 600),
        "triglyceride": (20, 2000),
        "hdl": (10, 200),
        "ldl": (20, 500),
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


def parse_blood_pressure(text_lines, is_health_check=False):
    for i, (text, _) in enumerate(text_lines):
        if is_keyword_match(text, FIELD_KEYWORDS["systolic_bp"]):
            if is_health_check:
                for k in range(0, len(text_lines)):
                    next_text, _ = text_lines[k]
                    if 'mmHg' in next_text:
                        if '/' in next_text:
                            parts = next_text.split('/')
                            systolic_nums = [n for n in extract_numbers(parts[0]) if 60 <= n <= 250]
                            diastolic_nums = [n for n in extract_numbers(parts[1]) if 40 <= n <= 150]
                            if systolic_nums and diastolic_nums:
                                # 공단 원본: 슬래시 앞이 수축기
                                s, d = systolic_nums[-1], diastolic_nums[0]
                                return (s, d) if s > d else (d, s)
                            if diastolic_nums and not systolic_nums:
                                # 합성 구조: 슬래시 뒤 첫 번째가 이완기, 줄 맨 끝이 수축기
                                all_nums = extract_numbers(next_text)
                                bp_nums = [n for n in all_nums if 60 <= n <= 250]
                                if bp_nums:
                                    s = bp_nums[-1]
                                    d = diastolic_nums[0]
                                    if validate_value("systolic_bp", s) and validate_value("diastolic_bp", d) and s != d:
                                        return (s, d) if s > d else (d, s)
                        # mmHg 줄 앞뒤 3줄에서 슬래시 수치 탐색
                        for offset in [-3, -2, -1, 1, 2, 3]:
                            idx = k + offset
                            if 0 <= idx < len(text_lines):
                                nearby_text, _ = text_lines[idx]
                                if '/' in nearby_text and 'mmHg' not in nearby_text:
                                    parts = nearby_text.split('/')
                                    if len(parts) == 2:
                                        systolic_nums = [n for n in extract_numbers(parts[0]) if 60 <= n <= 250]
                                        diastolic_nums = [n for n in extract_numbers(parts[1]) if 40 <= n <= 150]
                                        if systolic_nums and len(diastolic_nums) == 1:
                                            s, d = systolic_nums[-1], diastolic_nums[0]
                                            return (s, d) if s > d else (d, s)
            else:
                if '/' in text:
                    parts = text.split('/')
                    systolic_nums = [n for n in extract_numbers(parts[0]) if 60 <= n <= 250]
                    diastolic_nums = [n for n in extract_numbers(parts[1]) if 40 <= n <= 150] if len(parts) > 1 else []
                    if systolic_nums and diastolic_nums:
                        return systolic_nums[-1], diastolic_nums[0]
                bp_candidates = [n for n in extract_numbers(text) if 40 <= n <= 250]
                if len(bp_candidates) >= 2:
                    first, second = bp_candidates[:2]
                    return max(first, second), min(first, second)
                if len(bp_candidates) == 1:
                    return bp_candidates[0], None
    return None, None


def parse_height_weight(text_lines):
    for i, (text, _) in enumerate(text_lines):
        if is_keyword_match(text, FIELD_KEYWORDS["height_cm"]):
            context = text_lines[i : i + 15]
            all_nums = []
            for ctx_text, _ in context:
                # BMI 줄의 수치는 몸무게 후보로 잘못 집힐 수 있으므로 제외
                if is_keyword_match(ctx_text, FIELD_KEYWORDS["bmi"]):
                    continue
                all_nums.extend(extract_numbers(ctx_text))
            # 키 범위: 100~250, 단 혈압 범위(60~180)와 겹치므로 140 이상만 허용
            candidates_h = [n for n in all_nums if 140 <= n <= 220]
            if not candidates_h:
                continue
            height = candidates_h[0]
            height_index = all_nums.index(height)
            weight_pool = all_nums[height_index + 1:]
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
    키워드와 수치가 멀리 떨어진 공단 PDF 양식도 처리합니다.
    """
    hb_keyword_idx = None
    for i, (text, conf) in enumerate(text_lines):
        if "당화" in text:
            continue
        if not is_keyword_match(text, FIELD_KEYWORDS["hb"]):
            continue
        # 같은 줄에서 유효한 숫자 추출
        nums = [n for n in extract_numbers(text) if HB_RANGE[0] <= n <= HB_RANGE[1]]
        if nums:
            return nums[0], conf
        hb_keyword_idx = i
        break

    if hb_keyword_idx is None:
        return None, None

    # 키워드 이후 전체 줄에서 탐색 (공단 PDF는 수치가 멀리 있음)
    for j in range(hb_keyword_idx + 1, len(text_lines)):
        next_text, next_conf = text_lines[j]
        if "당화" in next_text:
            continue
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


def _extract_value_from_context(text_lines, i, text):
    # 참고치 패턴 제거 후 첫 번째 유효 숫자 추출
    ref_pattern = re.compile(r"\d+\.?\d*\s*(미만|이상|이하|초과)")
    cleaned = ref_pattern.sub("", text)
    value = extract_first_number(cleaned)
    confidence = text_lines[i][1]
    if value is None:
        for j in range(i + 1, min(i + 15, len(text_lines))):
            next_text = text_lines[j][0]
            next_cleaned = ref_pattern.sub("", next_text)
            value = extract_first_number(next_cleaned)
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
            # extractor.py의 _parse_general_fields 함수 내부
            if field in DYSLIPIDEMIA_FIELDS:
                check_lines = [text] + [text_lines[j][0] for j in range(i + 1, min(i + 3, len(text_lines)))]
                if any(is_not_applicable(line) for line in check_lines):
                    extracted[field] = "비해당"  # ← None 대신 "비해당"으로 채움
                    continue

            value, confidence = _extract_value_from_context(text_lines, i, text)
            if value is not None and validate_value(field, value):
                extracted[field] = value
                if confidence < CONFIDENCE_THRESHOLD:
                    low_conf.append(field)
                logger.info("필드 추출 | %s = %s (신뢰도: %.2f)", field, value, confidence)


def parse_from_text_lines(text_lines, is_health_check=False):
    raw_texts = [t for t, _ in text_lines]
    extracted: dict = {f: None for f in FIELD_KEYWORDS}
    low_conf = []

    # 혈압
    systolic, diastolic = parse_blood_pressure(text_lines, is_health_check=is_health_check)
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

    skip_fields = {"systolic_bp", "diastolic_bp", "height_cm", "weight_kg", "hb", "bmi"}
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


def run_ocr_on_image(image_bytes):
    processed = preprocess_for_ocr(image_bytes)
    engine = get_ocr_engine()
    results = engine.ocr(processed, cls=True)

    text_lines = []
    if results and results[0]:
        for line in results[0]:
            if line and len(line) >= 2:
                text = line[1][0].strip()
                confidence = float(line[1][1])
                text_lines.append((text, confidence))

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


async def run_ocr_on_pdf(pdf_bytes, filename: str = ""):
    is_health_check = filename.startswith("health_check")
    # 기존
    if is_health_check:
        texts = extract_text_from_health_check_pdf(pdf_bytes)
        page_text_lines = [[(line, 1.0) for line in parse_text_lines([text])] for text in texts]

    # 변경
    if is_health_check:
        texts = extract_text_from_health_check_pdf(pdf_bytes)
        if texts:
            page_text_lines = [[(line, 1.0) for line in parse_text_lines([text])] for text in texts]
        else:
            image_bytes_list = pdf_to_images(pdf_bytes)
            if not image_bytes_list:
                return CheckupOcrData(), [], [], OcrStatus.FAILED
            page_text_lines = []
            for img_bytes in image_bytes_list:
                page_lines = run_ocr_on_image(img_bytes)
                page_text_lines.append(page_lines)
    else:
        pdf_type = detect_pdf_type(pdf_bytes)
        page_text_lines = []
        if pdf_type == PdfType.TEXT:
            texts = extract_text_from_pdf(pdf_bytes)
            page_text_lines = [[(line, 1.0) for line in parse_text_lines([text])] for text in texts]
        else:
            image_bytes_list = pdf_to_images(pdf_bytes)
            if not image_bytes_list:
                return CheckupOcrData(), [], [], OcrStatus.FAILED
            for i, img_bytes in enumerate(image_bytes_list):
                page_lines = run_ocr_on_image(img_bytes)
                page_text_lines.append(page_lines)

    measurement_lines = select_measurement_page_lines(page_text_lines)
    data, low_conf, raw = parse_from_text_lines(measurement_lines, is_health_check=is_health_check)
    status = determine_status(data, low_conf)
    if status == OcrStatus.FAILED and measurement_lines != flatten_page_lines(page_text_lines):
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
