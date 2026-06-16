"""
ai_runtime/ocr/checkup/pdf_handler.py
PDF 파일 처리 모듈.
텍스트 PDF → pdfplumber 직접 추출
스캔 PDF   → pdf2image 변환 후 OCR
"""

import io
import logging
from enum import StrEnum

import pdfplumber

try:
    import pdf2image as _pdf2image
except ModuleNotFoundError:
    _pdf2image = None

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 50
MAX_PAGES: int | None = None
pdf2image = _pdf2image


class PdfImageDependencyError(RuntimeError):
    """Raised when optional PDF image conversion dependencies are not installed."""


class PdfType(StrEnum):
    TEXT = "text"
    SCAN = "scan"


def _get_pdf2image_module():
    if pdf2image is None:
        msg = "pdf2image is required for scanned checkup PDF OCR. Install OCR dependencies before converting PDFs."
        raise PdfImageDependencyError(msg)
    return pdf2image


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
    pdf2image_module = _get_pdf2image_module()
    image_bytes_list = []
    try:
        options = {"dpi": dpi, "first_page": 1}
        if MAX_PAGES is not None:
            options["last_page"] = MAX_PAGES
        images = pdf2image_module.convert_from_bytes(pdf_bytes, **options)
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
