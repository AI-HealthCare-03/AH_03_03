import re
from datetime import datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any

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


async def run_exam_ocr(exam_report_id: int) -> ExamOCRResponse:
    # Current demo path creates provider/fallback candidates; confirm syncs reviewed values into HealthRecord.
    existing_measurements = await list_exam_measurements(exam_report_id)
    existing_by_key = {measurement.measurement_key: measurement for measurement in existing_measurements}
    saved_measurements: list[ExamMeasurement] = []

    for key, name, value, unit in FALLBACK_OCR_MEASUREMENTS:
        request = ExamMeasurementCreateRequest(
            measurement_key=key,
            measurement_name=name,
            value=value,
            unit=unit,
            ocr_confidence=Decimal("0.9700"),
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
        message="측정값 후보가 생성되었습니다. 현재 provider/fallback 기반 결과이므로 검진 수치를 확인한 뒤 저장해주세요.",
        measurements=saved_measurements,
    )
