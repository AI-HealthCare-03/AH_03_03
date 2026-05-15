"""
ai_worker/vision/ocr/router.py
건강검진표 OCR FastAPI 라우터.
이미지 및 PDF 모두 지원.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from .extractor import run_ocr, run_ocr_on_pdf
from .preprocessor import assess_quality
from .schemas import (
    ERROR_MAP,
    STATUS_MESSAGE,
    CheckupOcrResponse,
    ErrorResponse,
    ImageQualityReport,
    ImageQualityStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ocr", tags=["OCR - 건강검진표 수치 추출"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic"}
ALLOWED_PDF_TYPES   = {"application/pdf"}
MAX_SIZE_BYTES      = 20 * 1024 * 1024  # 20MB


async def validate_image(file):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        err = ERROR_MAP["unsupported_type"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(error_code="unsupported_type", message=err["message"]).model_dump(),
        )
    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE_BYTES:
        err = ERROR_MAP["image_too_large"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(error_code="image_too_large", message=err["message"]).model_dump(),
        )
    if len(image_bytes) < 1024:
        err = ERROR_MAP["image_too_small"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(error_code="image_too_small", message=err["message"]).model_dump(),
        )
    return image_bytes


async def validate_pdf(file):
    if file.content_type not in ALLOWED_PDF_TYPES:
        raise HTTPException(
            status_code=415,
            detail=ErrorResponse(error_code="unsupported_type", message="PDF 파일만 업로드할 수 있습니다.").model_dump(),
        )
    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=ErrorResponse(error_code="image_too_large", message="파일 크기가 너무 큽니다. 20MB 이하로 다시 시도해주세요.").model_dump(),
        )
    if len(pdf_bytes) < 1024:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(error_code="image_too_small", message="PDF 파일이 손상되었거나 너무 작습니다.").model_dump(),
        )
    return pdf_bytes


@router.post(
    "/checkup",
    response_model=CheckupOcrResponse,
    summary="건강검진표 이미지 OCR",
    description="""
건강검진표 이미지를 업로드하면 4대 만성질환 관련 수치를 추출합니다.

- extracted_data: 추출된 수치 (null → 수기 입력 필요)
- low_confidence_fields: 신뢰도 낮은 필드 → 사용자 확인 필요
- quality_report: 이미지 품질 및 재촬영 가이드
    """,
)
async def analyze_checkup_image(
    file: Annotated[UploadFile, File(description="건강검진표 이미지 (JPG/PNG/WEBP, 최대 20MB)")],
):
    image_bytes    = await validate_image(file)
    quality_report = assess_quality(image_bytes)

    if quality_report.status == ImageQualityStatus.SMALL:
        err = ERROR_MAP["quality_failed"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(error_code="quality_failed", message=quality_report.message).model_dump(),
        )

    try:
        data, low_conf, raw_texts, ocr_status = await run_ocr(image_bytes)
    except Exception:
        logger.exception("이미지 OCR 실행 중 오류")
        err = ERROR_MAP["ocr_error"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(error_code="ocr_error", message=err["message"]).model_dump(),
        ) from None

    return CheckupOcrResponse(
        ocr_status=ocr_status,
        message=STATUS_MESSAGE.get(ocr_status, STATUS_MESSAGE["failed"]),
        quality_report=quality_report,
        extracted_data=data,
        low_confidence_fields=low_conf,
        raw_text=raw_texts,
    )


@router.post(
    "/checkup/pdf",
    response_model=CheckupOcrResponse,
    summary="건강검진표 PDF OCR",
    description="""
건강검진표 PDF를 업로드하면 4대 만성질환 관련 수치를 추출합니다.

- 텍스트 PDF (공단 발급 디지털 PDF) → 텍스트 직접 추출
- 스캔 PDF (종이 스캔) → 이미지 변환 후 OCR
    """,
)
async def analyze_checkup_pdf(
    file: Annotated[UploadFile, File(description="건강검진표 PDF (최대 20MB)")],
):
    pdf_bytes = await validate_pdf(file)

    try:
        data, low_conf, raw_texts, ocr_status = await run_ocr_on_pdf(pdf_bytes)
    except Exception:
        logger.exception("PDF OCR 실행 중 오류")
        err = ERROR_MAP["ocr_error"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(error_code="ocr_error", message=err["message"]).model_dump(),
        ) from None


    quality_report = ImageQualityReport(
        status=ImageQualityStatus.GOOD,
        message="PDF 파일이 정상적으로 처리되었습니다.",
        guide=[],
    )

    return CheckupOcrResponse(
        ocr_status=ocr_status,
        message=STATUS_MESSAGE.get(ocr_status, STATUS_MESSAGE["failed"]),
        quality_report=quality_report,
        extracted_data=data,
        low_confidence_fields=low_conf,
        raw_text=raw_texts,
    )
