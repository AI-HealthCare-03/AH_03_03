"""
ai_worker/vision/ocr/extractor.py
PaddleOCR 2.7.3 기반 건강검진표 수치 추출 모듈.
이미지 및 PDF(텍스트/스캔) 모두 지원.
"""

import logging
import re

from paddleocr import PaddleOCR

from .pdf_handler import PdfType, detect_pdf_type, extract_text_from_pdf, parse_text_lines, pdf_to_images
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
            use_angle_cls=True,
            lang="korean",
            show_log=False,
        )
        logger.info("PaddleOCR 모델 로딩 완료")
    return _ocr_engine


# ── 키워드 매핑 ───────────────────────────────────────────────────────────────

FIELD_KEYWORDS = {
    "systolic_bp":       ["수축기", "고혈압", "혈압", "mmHg", "SBP"],
    "diastolic_bp":      ["이완기", "DBP"],
    "fasting_glucose":   ["공복혈당", "혈당", "공복", "GLU", "Glucose"],
    "hba1c":             ["당화혈색소", "HbA1c", "A1C", "HBA1C"],
    "total_cholesterol": ["총콜레스테롤", "콜레스테롤", "TC", "T-CHO", "CHOL"],
    "triglyceride":      ["중성지방", "TG", "Triglyceride"],
    "hdl":               ["고밀도", "HDL"],
    "ldl":               ["저밀도", "LDL"],
    "height_cm":         ["신장", "키", "Height", "HT"],
    "weight_kg":         ["체중", "몸무게", "Weight", "WT"],
    "bmi":               ["BMI", "체질량", "비만도", "체질량지수"],
    "waist_cm":          ["허리둘레", "허리", "복부둘레", "Waist"],
}

CONFIDENCE_THRESHOLD = 0.7
CHECKBOX_PATTERN     = re.compile(r"[■□▣▪●○◆◇]")


def clean_text(text):
    return CHECKBOX_PATTERN.sub("", text).strip()


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
        "systolic_bp":       (60, 250),
        "diastolic_bp":      (40, 150),
        "fasting_glucose":   (40, 600),
        "hba1c":             (3, 20),
        "total_cholesterol": (50, 600),
        "triglyceride":      (20, 2000),
        "hdl":               (10, 200),
        "ldl":               (20, 500),
        "height_cm":         (100, 250),
        "weight_kg":         (20, 300),
        "bmi":               (10, 70),
        "waist_cm":          (40, 200),
    }
    if field not in ranges:
        return True
    low, high = ranges[field]
    return low <= value <= high


def parse_blood_pressure(text_lines):
    for text, _ in text_lines:
        if is_keyword_match(text, FIELD_KEYWORDS["systolic_bp"]):
            numbers = extract_numbers(text)
            if len(numbers) >= 2:
                return numbers[0], numbers[1]
            elif len(numbers) == 1:
                return numbers[0], None
    return None, None


def parse_height_weight(text_lines):
    for text, _ in text_lines:
        if is_keyword_match(text, ["키", "신장", "몸무게", "체중", "Height"]):
            numbers = extract_numbers(text)
            if len(numbers) >= 2:
                h, w = numbers[0], numbers[1]
                if 100 <= h <= 250 and 20 <= w <= 300:
                    return h, w
    return None, None


def parse_from_text_lines(text_lines):
    raw_texts  = [t for t, _ in text_lines]
    extracted  = {f: None for f in FIELD_KEYWORDS}
    low_conf   = []

    systolic, diastolic = parse_blood_pressure(text_lines)
    if systolic and validate_value("systolic_bp", systolic):
        extracted["systolic_bp"] = systolic
    if diastolic and validate_value("diastolic_bp", diastolic):
        extracted["diastolic_bp"] = diastolic

    height, weight = parse_height_weight(text_lines)
    if height and validate_value("height_cm", height):
        extracted["height_cm"] = height
    if weight and validate_value("weight_kg", weight):
        extracted["weight_kg"] = weight

    skip_fields = {"systolic_bp", "diastolic_bp", "height_cm", "weight_kg"}

    for i, (text, confidence) in enumerate(text_lines):
        for field, keywords in FIELD_KEYWORDS.items():
            if field in skip_fields or extracted[field] is not None:
                continue
            if not is_keyword_match(text, keywords):
                continue

            value = extract_first_number(text)
            if value is None:
                for j in range(i + 1, min(i + 3, len(text_lines))):
                    value = extract_first_number(text_lines[j][0])
                    if value is not None:
                        confidence = text_lines[j][1]
                        break

            if value is not None and validate_value(field, value):
                extracted[field] = value
                if confidence < CONFIDENCE_THRESHOLD:
                    low_conf.append(field)
                logger.info("필드 추출 | %s = %s (신뢰도: %.2f)", field, value, confidence)

    return CheckupOcrData(**extracted), low_conf, raw_texts


def run_ocr_on_image(image_bytes):
    processed = preprocess_for_ocr(image_bytes)
    engine    = get_ocr_engine()
    results   = engine.ocr(processed, cls=True)

    text_lines = []
    if results and results[0]:
        for line in results[0]:
            if line and len(line) >= 2:
                text       = line[1][0].strip()
                confidence = float(line[1][1])
                text_lines.append((text, confidence))

    logger.info("OCR 인식 %d줄", len(text_lines))
    return text_lines


def determine_status(data, low_conf):
    extracted_count = sum(1 for v in data.model_dump().values() if v is not None)
    if extracted_count == 0:
        return OcrStatus.FAILED
    if low_conf or extracted_count < 4:
        return OcrStatus.PARTIAL
    return OcrStatus.SUCCESS


async def run_ocr(image_bytes):
    text_lines          = run_ocr_on_image(image_bytes)
    data, low_conf, raw = parse_from_text_lines(text_lines)
    status              = determine_status(data, low_conf)
    return data, low_conf, raw, status


async def run_ocr_on_pdf(pdf_bytes):
    pdf_type       = detect_pdf_type(pdf_bytes)
    all_text_lines = []

    if pdf_type == PdfType.TEXT:
        logger.info("텍스트 PDF 처리 중...")
        texts = extract_text_from_pdf(pdf_bytes)
        lines = parse_text_lines(texts)
        all_text_lines = [(line, 1.0) for line in lines]
    else:
        logger.info("스캔 PDF 처리 중...")
        image_bytes_list = pdf_to_images(pdf_bytes)
        if not image_bytes_list:
            logger.error("PDF 이미지 변환 실패")
            return CheckupOcrData(), [], [], OcrStatus.FAILED
        for i, img_bytes in enumerate(image_bytes_list):
            logger.info("페이지 %d OCR 실행 중...", i + 1)
            page_lines = run_ocr_on_image(img_bytes)
            all_text_lines.extend(page_lines)

    data, low_conf, raw = parse_from_text_lines(all_text_lines)
    status              = determine_status(data, low_conf)
    return data, low_conf, raw, status
