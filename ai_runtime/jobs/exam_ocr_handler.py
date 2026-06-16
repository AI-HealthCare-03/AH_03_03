from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from ai_runtime.cv.providers.gpt_vision import AnalysisType, VisionClient
from app.core import config
from app.dtos.exams import (
    ExamMeasurementCreateRequest,
    ExamMeasurementUpdateRequest,
    ExamOCRResponse,
    ExamReportUpdateRequest,
)
from app.models.exams import OCRStatus
from app.services import exams as exam_service

logger = logging.getLogger(__name__)

PDF_RENDER_ZOOM = 2.0
EXAM_MEASUREMENT_PAGE_KEYWORDS = {
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


@dataclass(frozen=True)
class ExamPdfImageConversionResult:
    page_count: int
    images: list[bytes]


async def run_exam_ocr_from_report(exam_report_id: int) -> ExamOCRResponse:
    report = await exam_service.get_exam_report(exam_report_id)
    if report is None:
        raise ValueError(f"exam_report_not_found:{exam_report_id}")

    stored_upload = exam_service._read_stored_exam_upload(report.file_path)
    if stored_upload is None:
        await exam_service.update_exam_report(exam_report_id, ExamReportUpdateRequest(ocr_status=OCRStatus.FAILED))
        raise FileNotFoundError(f"exam_upload_file_not_found:{exam_report_id}")

    image_bytes, media_type, filename = stored_upload
    await exam_service.update_exam_report(exam_report_id, ExamReportUpdateRequest(ocr_status=OCRStatus.PROCESSING))
    return await run_exam_ocr(
        exam_report_id=exam_report_id,
        image_bytes=image_bytes,
        image_media_type=media_type,
        image_filename=report.original_filename or filename,
    )


async def run_exam_ocr(
    exam_report_id: int,
    image_bytes: bytes | None = None,
    image_media_type: str | None = None,
    image_filename: str | None = None,
) -> ExamOCRResponse:
    provider_result = await _extract_exam_measurements_with_provider(
        image_bytes=image_bytes,
        image_media_type=image_media_type,
        image_filename=image_filename,
    )
    parsed_candidate_count = len(provider_result["measurements"])
    if parsed_candidate_count == 0:
        raw_text_preview = provider_result.get("raw_text_preview")
        logger.info(
            "건강검진 OCR 후보 없음 | provider=%s provider_message=%s content_type=%s file_extension=%s "
            "extracted_text_length=%s parsed_candidate_count=%s",
            provider_result["provider"],
            provider_result["provider_message"],
            image_media_type,
            Path(image_filename or "").suffix.lower() or None,
            len(raw_text_preview or ""),
            parsed_candidate_count,
        )
    existing_measurements = await exam_service.list_exam_measurements(exam_report_id)
    existing_by_key = {measurement.measurement_key: measurement for measurement in existing_measurements}
    saved_measurements = []

    for key, name, value, unit in provider_result["measurements"]:
        request = ExamMeasurementCreateRequest(
            measurement_key=key,
            measurement_name=name,
            value=value,
            unit=unit,
            ocr_confidence=provider_result["confidence"],
            is_user_confirmed=False,
        )
        existing = existing_by_key.get(key)
        if existing is None:
            saved_measurements.append(await exam_service.create_exam_measurement(exam_report_id, request))
            continue

        updated = await exam_service.update_exam_measurement(
            existing.id,
            ExamMeasurementUpdateRequest(**request.model_dump()),
        )
        saved_measurements.append(updated or existing)

    await exam_service.update_exam_report(
        exam_report_id,
        ExamReportUpdateRequest(ocr_status=OCRStatus.SUCCESS if saved_measurements else OCRStatus.FAILED),
    )
    return ExamOCRResponse(
        message=provider_result["message"],
        measurements=saved_measurements,
        ocr_provider=provider_result["provider"],
        fallback_used=provider_result["fallback_used"],
        provider_message=provider_result["provider_message"],
        raw_text_preview=provider_result["raw_text_preview"],
    )


async def _extract_exam_measurements_with_provider(
    image_bytes: bytes | None,
    image_media_type: str | None,
    image_filename: str | None = None,
) -> dict[str, Any]:
    provider = str(config.EXAM_OCR_PROVIDER or "fallback").lower()
    provider_order = _select_exam_ocr_provider_order(provider, image_media_type, image_filename)
    failure_reasons: list[str] = []
    logger.info(
        "건강검진 OCR provider routing | configured=%s content_type=%s filename=%s order=%s",
        provider,
        image_media_type,
        image_filename,
        ",".join(provider_order),
    )

    for provider_name in provider_order:
        if provider_name == "paddleocr":
            if not config.PADDLE_OCR_ENABLED:
                failure_reasons.append("paddleocr_disabled")
                continue
            result = await _extract_exam_measurements_with_paddleocr(image_bytes, image_media_type, image_filename)
            if result is not None:
                if failure_reasons:
                    result["fallback_used"] = True
                return result
            failure_reasons.append("paddleocr_failed_or_unavailable")
            continue

        if provider_name == "gpt_vision":
            if not config.EXAM_GPT_VISION_ENABLED:
                failure_reasons.append("gpt_vision_disabled")
                continue
            result = await _extract_exam_measurements_with_gpt_vision(image_bytes, image_media_type, image_filename)
            if result is not None:
                if failure_reasons:
                    result["fallback_used"] = True
                return result
            failure_reasons.append("gpt_vision_failed_or_unavailable")

    reason = ";".join(failure_reasons) if failure_reasons else f"{provider}_disabled"
    return _empty_exam_ocr_result(reason)


def _select_exam_ocr_provider_order(
    configured_provider: str,
    image_media_type: str | None,
    image_filename: str | None,
) -> list[str]:
    is_pdf = _is_pdf_upload(image_media_type, image_filename)
    is_image = bool(image_media_type and image_media_type.lower().startswith("image/"))

    if configured_provider == "fallback":
        return []
    if is_pdf and configured_provider == "gpt_vision":
        return ["gpt_vision", "paddleocr"]
    if is_pdf and configured_provider in {"auto", "paddleocr"}:
        return ["paddleocr", "gpt_vision"]
    if is_image and configured_provider in {"auto", "gpt_vision"}:
        return ["gpt_vision", "paddleocr"]
    if configured_provider == "paddleocr":
        return ["paddleocr", "gpt_vision"]
    if configured_provider == "gpt_vision":
        return ["gpt_vision", "paddleocr"]
    return []


def _is_pdf_upload(image_media_type: str | None, image_filename: str | None) -> bool:
    media_type = (image_media_type or "").lower().split(";", 1)[0].strip()
    filename = (image_filename or "").lower()
    return media_type == "application/pdf" or filename.endswith(".pdf")


def _convert_exam_pdf_to_png_images(pdf_bytes: bytes) -> ExamPdfImageConversionResult:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required to convert health exam PDF uploads for GPT Vision.") from exc

    images: list[bytes] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        page_count = document.page_count
        matrix = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(pixmap.tobytes("png"))
    return ExamPdfImageConversionResult(page_count=page_count, images=images)


def _extract_exam_pdf_page_texts(pdf_bytes: bytes) -> list[str]:
    try:
        import fitz
    except Exception:
        return []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            return [page.get_text("text") or "" for page in document]
    except Exception:
        logger.info("건강검진 PDF 페이지 텍스트 메타 추출 실패", exc_info=True)
        return []


def _score_exam_measurement_page(text: str) -> int:
    normalized = text.upper()
    return sum(weight * normalized.count(keyword.upper()) for keyword, weight in EXAM_MEASUREMENT_PAGE_KEYWORDS.items())


def _select_exam_measurement_page_indices(page_texts: list[str]) -> list[int]:
    if not page_texts:
        return []
    scores = [_score_exam_measurement_page(text) for text in page_texts]
    max_score = max(scores, default=0)
    if max_score < MEASUREMENT_PAGE_MIN_SCORE:
        return []
    return [index for index, score in enumerate(scores) if score == max_score]


def _merge_exam_extracted_data(target: dict[str, Any], page_data: dict[str, Any]) -> None:
    for key, value in page_data.items():
        existing = target.get(key)
        if existing is None or existing == "":
            target[key] = value
            continue
        if (
            exam_service.parse_exam_measurement_number(existing) is None
            and exam_service.parse_exam_measurement_number(value) is not None
        ):
            target[key] = value


def _gpt_vision_response_text_length(raw: object) -> int:
    if not isinstance(raw, dict):
        return 0
    text_fields = ("raw_text_preview", "raw_text", "text", "ocr_text")
    return sum(len(value) for key, value in raw.items() if key in text_fields and isinstance(value, str))


async def _extract_exam_measurements_with_gpt_vision(
    image_bytes: bytes | None,
    image_media_type: str | None,
    image_filename: str | None = None,
) -> dict[str, Any] | None:
    if not image_bytes or not config.OPENAI_API_KEY:
        return None
    file_extension = Path(image_filename or "").suffix.lower() or None
    vision_inputs: list[tuple[bytes, str]] = [(image_bytes, image_media_type or "image/jpeg")]
    all_pdf_vision_inputs: list[tuple[bytes, str]] = []
    page_count: int | None = None
    converted_image_count = 0
    selected_page_indices: list[int] = []
    page_selection_fallback_used = False

    if _is_pdf_upload(image_media_type, image_filename):
        conversion = _convert_exam_pdf_to_png_images(image_bytes)
        page_count = conversion.page_count
        converted_image_count = len(conversion.images)
        page_texts = _extract_exam_pdf_page_texts(image_bytes)
        selected_page_indices = _select_exam_measurement_page_indices(page_texts)
        logger.info(
            "GPT Vision 건강검진 OCR PDF 이미지 변환 | provider=gpt_vision content_type=%s file_ext=%s "
            "page_count=%s converted_image_count=%s selected_page_count=%s",
            image_media_type,
            file_extension,
            page_count,
            converted_image_count,
            len(selected_page_indices),
        )
        if not conversion.images:
            return None
        all_pdf_vision_inputs = [(converted_image, "image/png") for converted_image in conversion.images]
        if selected_page_indices:
            vision_inputs = [
                all_pdf_vision_inputs[index] for index in selected_page_indices if index < len(all_pdf_vision_inputs)
            ]
        else:
            vision_inputs = all_pdf_vision_inputs

    extracted_data: dict[str, Any] = {}
    extracted_text_length = 0
    try:
        client = VisionClient(api_key=config.OPENAI_API_KEY, model=config.EXAM_GPT_VISION_MODEL)
        extracted_data, extracted_text_length = await _collect_gpt_vision_exam_data(client, vision_inputs)
    except Exception as exc:
        logger.warning("GPT Vision 건강검진 OCR 실패: %s", exc, exc_info=True)
        return None

    measurements = exam_service._measurement_tuples_from_mapping(extracted_data)
    if not measurements and selected_page_indices and len(vision_inputs) < len(all_pdf_vision_inputs):
        page_selection_fallback_used = True
        logger.info(
            "GPT Vision 건강검진 OCR 측정 페이지 후보 부족으로 전체 페이지 fallback | provider=gpt_vision "
            "content_type=%s file_ext=%s page_count=%s converted_image_count=%s",
            image_media_type,
            file_extension,
            page_count,
            converted_image_count,
        )
        try:
            client = VisionClient(api_key=config.OPENAI_API_KEY, model=config.EXAM_GPT_VISION_MODEL)
            extracted_data, extracted_text_length = await _collect_gpt_vision_exam_data(client, all_pdf_vision_inputs)
            measurements = exam_service._measurement_tuples_from_mapping(extracted_data)
        except Exception as exc:
            logger.warning("GPT Vision 건강검진 OCR 전체 페이지 fallback 실패: %s", exc, exc_info=True)
            return None

    if not extracted_data:
        logger.info(
            "GPT Vision 건강검진 OCR 추출 데이터 없음 | provider=gpt_vision content_type=%s file_ext=%s "
            "page_count=%s converted_image_count=%s extracted_text_length=%s candidate_count=0",
            image_media_type,
            file_extension,
            page_count,
            converted_image_count,
            extracted_text_length,
        )
        return None
    if not measurements:
        logger.info(
            "GPT Vision 건강검진 OCR 파싱 후보 없음 | provider=gpt_vision content_type=%s file_ext=%s "
            "page_count=%s converted_image_count=%s extracted_text_length=%s extracted_field_count=%s "
            "candidate_count=0",
            image_media_type,
            file_extension,
            page_count,
            converted_image_count,
            extracted_text_length,
            len(extracted_data),
        )
        return None
    return {
        "provider": "gpt_vision",
        "fallback_used": page_selection_fallback_used,
        "provider_message": "gpt_vision_checkup_ocr",
        "message": "GPT Vision으로 측정값 후보를 생성했습니다. 검진 수치를 확인한 뒤 저장해주세요.",
        "measurements": measurements,
        "confidence": Decimal("0.9000"),
        "raw_text_preview": None,
    }


async def _collect_gpt_vision_exam_data(
    client: VisionClient,
    vision_inputs: list[tuple[bytes, str]],
) -> tuple[dict[str, Any], int]:
    extracted_data: dict[str, Any] = {}
    extracted_text_length = 0
    for vision_image_bytes, vision_media_type in vision_inputs:
        raw = await client.analyze(
            analysis_type=AnalysisType.CHECKUP,
            image_bytes=vision_image_bytes,
            media_type=vision_media_type,
        )
        extracted_text_length += _gpt_vision_response_text_length(raw)
        page_data = raw.get("extracted_data") if isinstance(raw, dict) else None
        if not isinstance(page_data, dict):
            logger.info("GPT Vision 건강검진 OCR 응답에 extracted_data 없음")
            continue
        _merge_exam_extracted_data(extracted_data, page_data)
    return extracted_data, extracted_text_length


async def _extract_exam_measurements_with_paddleocr(
    image_bytes: bytes | None,
    image_media_type: str | None,
    image_filename: str | None = None,
) -> dict[str, Any] | None:
    if not image_bytes:
        return None
    try:
        from ai_runtime.ocr.checkup.extractor import run_ocr, run_ocr_on_pdf
    except Exception as exc:
        logger.warning("PaddleOCR import 실패: %s", exc, exc_info=True)
        return None

    try:
        if _is_pdf_upload(image_media_type, image_filename):
            data, low_confidence_fields, raw_text, _status = await run_ocr_on_pdf(image_bytes)
        else:
            data, low_confidence_fields, raw_text, _status = await run_ocr(image_bytes)
    except Exception as exc:
        logger.warning("PaddleOCR 건강검진 OCR 실패: %s", exc, exc_info=True)
        return None

    values = data.model_dump() if hasattr(data, "model_dump") else {}
    measurements = exam_service._measurement_tuples_from_mapping(values)
    if not measurements:
        logger.info(
            "PaddleOCR 건강검진 OCR 파싱 후보 없음 | extracted_text_length=%s extracted_field_count=%s "
            "parsed_candidate_count=0",
            sum(len(str(line[0] if isinstance(line, tuple) else line)) for line in raw_text),
            len(values),
        )
        return None
    confidence = Decimal("0.7000") if low_confidence_fields else Decimal("0.9000")
    raw_preview = "\n".join(str(line[0] if isinstance(line, tuple) else line) for line in raw_text[:8])
    return {
        "provider": "paddleocr",
        "fallback_used": False,
        "provider_message": "paddleocr_checkup_ocr",
        "message": "PaddleOCR로 측정값 후보를 생성했습니다. 검진 수치를 확인한 뒤 저장해주세요.",
        "measurements": measurements,
        "confidence": confidence,
        "raw_text_preview": raw_preview or None,
    }


def _empty_exam_ocr_result(reason: str) -> dict[str, Any]:
    return {
        "provider": "none",
        "fallback_used": False,
        "provider_message": reason,
        "message": "인식된 측정값 후보가 없습니다. 파일을 다시 확인해주세요.",
        "measurements": [],
        "confidence": None,
        "raw_text_preview": None,
    }
