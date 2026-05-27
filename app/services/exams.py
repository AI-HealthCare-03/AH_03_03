import logging
import re
from datetime import datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any

from ai_runtime.cv.providers.gpt_vision import AnalysisType, VisionClient
from app.core import config
from app.dtos.exams import (
    ExamConfirmMeasurementRequest,
    ExamConfirmRequest,
    ExamMeasurementCreateRequest,
    ExamMeasurementUpdateRequest,
    ExamOCRResponse,
    ExamReportCreateRequest,
    ExamReportUpdateRequest,
)
from app.models.exams import ExamMeasurement, ExamReport, OCRStatus
from app.repositories import exam_repository, health_repository
from app.services.health import _with_calculated_bmi

FALLBACK_OCR_MEASUREMENTS = [
    ("height_cm", "키", "172.4", "cm"),
    ("weight_kg", "몸무게", "76.8", "kg"),
    ("bmi", "체질량지수", "25.8", "kg/m2"),
    ("waist_cm", "허리둘레", "89.0", "cm"),
    ("systolic_bp", "수축기혈압", "132", "mmHg"),
    ("diastolic_bp", "이완기혈압", "84", "mmHg"),
    ("fasting_glucose", "공복혈당", "108", "mg/dL"),
    ("hba1c", "당화혈색소", "5.8", "%"),
    ("total_cholesterol", "총콜레스테롤", "210", "mg/dL"),
    ("ldl", "LDL 콜레스테롤", "132", "mg/dL"),
    ("hdl", "HDL 콜레스테롤", "46", "mg/dL"),
    ("triglyceride", "중성지방", "158", "mg/dL"),
    ("ast", "AST", "26", "U/L"),
    ("alt", "ALT", "31", "U/L"),
    ("gamma_gtp", "감마지티피", "42", "U/L"),
    ("creatinine", "혈청크레아티닌", "0.92", "mg/dL"),
    ("egfr", "eGFR", "91", "mL/min/1.73m2"),
    ("hemoglobin", "혈색소", "14.8", "g/dL"),
    ("urine_protein", "요단백", "음성", None),
]

EXAM_MEASUREMENT_TO_HEALTH_FIELD = {
    "height": "height_cm",
    "height_cm": "height_cm",
    "weight": "weight_kg",
    "weight_kg": "weight_kg",
    "bmi": "bmi",
    "waist": "waist_cm",
    "waist_cm": "waist_cm",
    "systolic_bp": "systolic_bp",
    "diastolic_bp": "diastolic_bp",
    "fasting_glucose": "fasting_glucose",
    "hba1c": "hba1c",
    "total_cholesterol": "total_cholesterol",
    "triglyceride": "triglyceride",
    "hdl": "hdl_cholesterol",
    "hdl_cholesterol": "hdl_cholesterol",
    "ldl": "ldl_cholesterol",
    "ldl_cholesterol": "ldl_cholesterol",
}

HEALTH_INT_FIELDS = {
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "total_cholesterol",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglyceride",
}

NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
logger = logging.getLogger(__name__)


async def create_exam_report(user_id: int, request: ExamReportCreateRequest) -> ExamReport:
    return await exam_repository.create_exam_report(user_id, request.model_dump())


async def get_exam_report(exam_report_id: int) -> ExamReport | None:
    return await exam_repository.get_exam_report_by_id(exam_report_id)


async def list_exam_reports(user_id: int, limit: int = 20, offset: int = 0) -> list[ExamReport]:
    return await exam_repository.list_exam_reports_by_user(user_id, limit=limit, offset=offset)


async def update_exam_report(exam_report_id: int, request: ExamReportUpdateRequest) -> ExamReport | None:
    data = request.model_dump(exclude_unset=True)
    return await exam_repository.update_exam_report(exam_report_id, data)


async def create_exam_measurement(exam_report_id: int, request: ExamMeasurementCreateRequest) -> ExamMeasurement:
    return await exam_repository.create_exam_measurement(exam_report_id, request.model_dump())


async def get_exam_measurement(measurement_id: int) -> ExamMeasurement | None:
    return await exam_repository.get_exam_measurement_by_id(measurement_id)


async def create_exam_measurements(
    exam_report_id: int, measurements: list[ExamMeasurementCreateRequest]
) -> list[ExamMeasurement]:
    data = [measurement.model_dump() for measurement in measurements]
    return await exam_repository.create_exam_measurements(exam_report_id, data)


async def list_exam_measurements(exam_report_id: int) -> list[ExamMeasurement]:
    return await exam_repository.list_exam_measurements(exam_report_id)


async def update_exam_measurement(measurement_id: int, request: ExamMeasurementUpdateRequest) -> ExamMeasurement | None:
    data = request.model_dump(exclude_unset=True)
    return await exam_repository.update_exam_measurement(measurement_id, data)


async def confirm_exam_report(exam_report_id: int, request: ExamConfirmRequest | None = None) -> ExamReport | None:
    report = await get_exam_report(exam_report_id)
    if report is None:
        return None

    if request is not None and request.measurements:
        await apply_confirm_measurement_updates(exam_report_id, request.measurements)

    measurements = await list_exam_measurements(exam_report_id)
    await sync_exam_measurements_to_latest_health_record(report, measurements)
    return await exam_repository.confirm_exam_report(exam_report_id)


async def apply_confirm_measurement_updates(
    exam_report_id: int, measurements: list[ExamConfirmMeasurementRequest]
) -> list[ExamMeasurement]:
    existing_measurements = await list_exam_measurements(exam_report_id)
    existing_by_key = {measurement.measurement_key: measurement for measurement in existing_measurements}
    saved_measurements: list[ExamMeasurement] = []

    for measurement in measurements:
        key = measurement.measurement_key or measurement.key
        if key is None or not key.strip():
            continue
        value = None if measurement.value is None else str(measurement.value)
        existing = existing_by_key.get(key)
        if existing is None:
            saved_measurements.append(
                await create_exam_measurement(
                    exam_report_id,
                    ExamMeasurementCreateRequest(
                        measurement_key=key,
                        measurement_name=measurement.measurement_name or key,
                        value=value,
                        unit=measurement.unit,
                        ocr_confidence=measurement.ocr_confidence,
                        is_user_confirmed=measurement.is_user_confirmed
                        if measurement.is_user_confirmed is not None
                        else True,
                    ),
                )
            )
            continue

        update_data: dict[str, Any] = {
            "measurement_key": key,
            "is_user_confirmed": measurement.is_user_confirmed if measurement.is_user_confirmed is not None else True,
        }
        if measurement.measurement_name is not None:
            update_data["measurement_name"] = measurement.measurement_name
        if measurement.value is not None:
            update_data["value"] = value
        if measurement.unit is not None:
            update_data["unit"] = measurement.unit
        if measurement.ocr_confidence is not None:
            update_data["ocr_confidence"] = measurement.ocr_confidence

        update_request = ExamMeasurementUpdateRequest(**update_data)
        updated = await update_exam_measurement(existing.id, update_request)
        saved_measurements.append(updated or existing)
    return saved_measurements


async def sync_exam_measurements_to_latest_health_record(
    report: ExamReport, measurements: list[ExamMeasurement]
) -> dict[str, Any]:
    data = build_health_record_update_from_exam_measurements(measurements)
    if not data:
        return {}

    latest_record = await health_repository.get_latest_health_record_by_user(report.user_id)
    if latest_record is not None:
        await health_repository.update_health_record(latest_record.id, data)
        return data

    measured_at = (
        datetime.combine(report.exam_date, time.min, tzinfo=config.TIMEZONE)
        if report.exam_date
        else datetime.now(config.TIMEZONE)
    )
    await health_repository.create_health_record(report.user_id, {"measured_at": measured_at, **data})
    return data


def build_health_record_update_from_exam_measurements(measurements: list[ExamMeasurement]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for measurement in measurements:
        health_field = EXAM_MEASUREMENT_TO_HEALTH_FIELD.get(measurement.measurement_key)
        if health_field is None:
            continue
        parsed_value = parse_exam_measurement_number(measurement.value)
        if parsed_value is None:
            continue
        data[health_field] = _coerce_health_record_value(health_field, parsed_value)
    return _with_calculated_bmi(data)


def parse_exam_measurement_number(value: object) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int | float):
        return Decimal(str(value))

    text = str(value).strip()
    if not text:
        return None
    match = NUMBER_PATTERN.search(text.replace(",", ""))
    if match is None:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def _coerce_health_record_value(field_name: str, value: Decimal) -> int | Decimal:
    if field_name in HEALTH_INT_FIELDS:
        return int(value)
    return value


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
    existing_measurements = await list_exam_measurements(exam_report_id)
    existing_by_key = {measurement.measurement_key: measurement for measurement in existing_measurements}
    saved_measurements: list[ExamMeasurement] = []

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
            saved_measurements.append(await create_exam_measurement(exam_report_id, request))
            continue

        updated = await update_exam_measurement(existing.id, ExamMeasurementUpdateRequest(**request.model_dump()))
        saved_measurements.append(updated or existing)

    await update_exam_report(exam_report_id, ExamReportUpdateRequest(ocr_status=OCRStatus.SUCCESS))
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
                return result
            failure_reasons.append("paddleocr_failed_or_unavailable")
            continue

        if provider_name == "gpt_vision":
            if not config.EXAM_GPT_VISION_ENABLED:
                failure_reasons.append("gpt_vision_disabled")
                continue
            result = await _extract_exam_measurements_with_gpt_vision(image_bytes, image_media_type)
            if result is not None:
                return result
            failure_reasons.append("gpt_vision_failed_or_unavailable")

    reason = ";".join(failure_reasons) if failure_reasons else f"{provider}_disabled"
    return _fallback_exam_ocr_result(reason)


def _select_exam_ocr_provider_order(
    configured_provider: str,
    image_media_type: str | None,
    image_filename: str | None,
) -> list[str]:
    is_pdf = _is_pdf_upload(image_media_type, image_filename)
    is_image = bool(image_media_type and image_media_type.lower().startswith("image/"))

    if configured_provider == "fallback":
        return []
    if is_pdf and configured_provider in {"auto", "gpt_vision", "paddleocr"}:
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


async def _extract_exam_measurements_with_gpt_vision(
    image_bytes: bytes | None,
    image_media_type: str | None,
) -> dict[str, Any] | None:
    if not image_bytes or not config.OPENAI_API_KEY:
        return None
    try:
        client = VisionClient(api_key=config.OPENAI_API_KEY, model=config.EXAM_GPT_VISION_MODEL)
        raw = await client.analyze(
            analysis_type=AnalysisType.CHECKUP,
            image_bytes=image_bytes,
            media_type=image_media_type or "image/jpeg",
        )
    except Exception as exc:
        logger.warning("GPT Vision 건강검진 OCR 실패: %s", exc, exc_info=True)
        return None

    extracted_data = raw.get("extracted_data") if isinstance(raw, dict) else None
    if not isinstance(extracted_data, dict):
        return None
    measurements = _measurement_tuples_from_mapping(extracted_data)
    if not measurements:
        return None
    return {
        "provider": "gpt_vision",
        "fallback_used": False,
        "provider_message": "gpt_vision_checkup_ocr",
        "message": "GPT Vision으로 측정값 후보를 생성했습니다. 검진 수치를 확인한 뒤 저장해주세요.",
        "measurements": measurements,
        "confidence": Decimal("0.9000"),
        "raw_text_preview": None,
    }


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
    measurements = _measurement_tuples_from_mapping(values)
    if not measurements:
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


def _fallback_exam_ocr_result(reason: str) -> dict[str, Any]:
    return {
        "provider": "fallback",
        "fallback_used": True,
        "provider_message": reason,
        "message": "측정값 후보가 생성되었습니다. 현재 fallback 기반 결과이므로 검진 수치를 확인한 뒤 저장해주세요.",
        "measurements": FALLBACK_OCR_MEASUREMENTS,
        "confidence": Decimal("0.9700"),
        "raw_text_preview": None,
    }


def _measurement_tuples_from_mapping(values: dict[str, Any]) -> list[tuple[str, str, str, str | None]]:
    names = {key: name for key, name, _value, _unit in FALLBACK_OCR_MEASUREMENTS}
    units = {key: unit for key, _name, _value, unit in FALLBACK_OCR_MEASUREMENTS}
    measurements = []
    for key, health_key in EXAM_MEASUREMENT_TO_HEALTH_FIELD.items():
        if key not in values:
            continue
        value = values.get(key)
        if value is None or value == "":
            continue
        measurement_key = key
        measurement_name = names.get(measurement_key) or names.get(health_key) or measurement_key
        measurements.append(
            (measurement_key, measurement_name, str(value), units.get(measurement_key) or units.get(health_key))
        )
    return measurements
