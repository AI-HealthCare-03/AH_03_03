"""
ai_worker/vision/ocr/pdf_handler.py
PDF 파일 처리 모듈.
텍스트 PDF → pdfplumber 직접 추출
스캔 PDF   → pdf2image 변환 후 OCR
"""

import io
import logging
from enum import Enum

import pdf2image
import pdfplumber

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 50
MAX_PAGES       = 5


class PdfType(str, Enum):
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
            for i, page in enumerate(pdf.pages[:MAX_PAGES]):
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
        images = pdf2image.convert_from_bytes(
            pdf_bytes, dpi=dpi,
            first_page=1, last_page=MAX_PAGES,
        )
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
