"""
ai_runtime/ocr/checkup/pdf_handler.py
PDF 파일 처리 모듈.
텍스트 PDF → pdfplumber 직접 추출
스캔 PDF   → pdf2image 변환 후 OCR
"""

import io
import logging
from enum import StrEnum

import pdf2image
import pdfplumber

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 50
MAX_PAGES: int | None = None


class PdfType(StrEnum):
    TEXT = "text"
    SCAN = "scan"


def detect_pdf_type(pdf_bytes):
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:2]:
                text = page.extract_text() or ""
                if len(text.strip()) >= MIN_TEXT_LENGTH:
                    logger.info("텍스트 PDF 감지")
                    return PdfType.TEXT
    except Exception as e:
        logger.warning("PDF 타입 감지 실패: %s", e)
    logger.info("스캔 PDF 감지")
    return PdfType.SCAN


def extract_text_from_pdf(pdf_bytes):
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = pdf.pages if MAX_PAGES is None else pdf.pages[:MAX_PAGES]
            for i, page in enumerate(pages):
                text = page.extract_text() or ""
                if text.strip():
                    texts.append(text)
                    logger.info("페이지 %d 텍스트 추출 | %d자", i + 1, len(text))
    except Exception as e:
        logger.error("텍스트 추출 실패: %s", e)
    return texts


def pdf_to_images(pdf_bytes, dpi=200):
    image_bytes_list = []
    try:
        options = {"dpi": dpi, "first_page": 1}
        if MAX_PAGES is not None:
            options["last_page"] = MAX_PAGES
        images = pdf2image.convert_from_bytes(pdf_bytes, **options)
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            image_bytes_list.append(buf.getvalue())
            logger.info("페이지 %d 변환 완료 | %dx%d", i + 1, img.width, img.height)
    except Exception as e:
        logger.error("PDF 이미지 변환 실패: %s", e)
    return image_bytes_list


def parse_text_lines(texts):
    lines = []
    for text in texts:
        for line in text.split("\n"):
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def extract_text_from_health_check_pdf(pdf_bytes):
    """
    공단 PDF 전용 추출기.
    키워드와 수치를 top 좌표 기준으로 같은 줄에 합쳐서 반환.
    """
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                words = page.extract_words(x_tolerance=1, y_tolerance=1)
                value_words = [w for w in words if w['x0'] <= 460]
                merged_lines = _group_values_prefer_synthetic(value_words)
                page_text = "\n".join(merged_lines)
                if page_text.strip():
                    texts.append(page_text)
    except Exception as e:
        logger.error("health_check PDF 추출 실패: %s", e)
    return texts


def _group_words_by_top(words, tolerance=3.0):
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
    lines = []
    current_top = sorted_words[0]['top']
    current_texts = [sorted_words[0]['text']]
    for w in sorted_words[1:]:
        if abs(w['top'] - current_top) <= tolerance:
            current_texts.append(w['text'])
        else:
            lines.append(' '.join(current_texts))
            current_top = w['top']
            current_texts = [w['text']]
    lines.append(' '.join(current_texts))
    return lines


def _group_values_prefer_synthetic(words, top_tolerance=3.0, x_tolerance=8.0):
    """
    같은 top 좌표에 여러 단어가 있을 때,
    x0가 비슷한 위치(x_tolerance 이내)의 중복값은 x0가 작은 것(합성 레이어)만 남김.
    """
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))

    groups = []
    current_top = sorted_words[0]['top']
    current_group = [sorted_words[0]]

    for w in sorted_words[1:]:
        if abs(w['top'] - current_top) <= top_tolerance:
            current_group.append(w)
        else:
            groups.append(current_group)
            current_top = w['top']
            current_group = [w]
    groups.append(current_group)

    lines = []
    for group in groups:
        deduped = []
        used = set()
        for i, w in enumerate(group):
            if i in used:
                continue
            duplicates = [
                j for j, w2 in enumerate(group)
                if j != i and j not in used and abs(w2['x0'] - w['x0']) <= x_tolerance
            ]
            if duplicates:
                candidates = [i] + duplicates
                # 비해당이 있으면 무조건 비해당 선택
                na_candidates = [idx for idx in candidates if group[idx]['text'] == '비해당']
                if na_candidates:
                    deduped.append('비해당')
                else:
                    best = max(candidates, key=lambda idx: group[idx]['top'])
                    deduped.append(group[best]['text'])
                used.update(candidates)
            else:
                deduped.append(w['text'])
                used.add(i)

        lines.append(' '.join(deduped))
    return lines
