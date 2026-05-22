from decimal import Decimal

from app.dtos.exams import (
    ExamConfirmRequest,
    ExamDummyOCRResponse,
    ExamMeasurementCreateRequest,
    ExamMeasurementUpdateRequest,
    ExamReportCreateRequest,
    ExamReportUpdateRequest,
)
from app.models.exams import ExamMeasurement, ExamReport, OCRStatus
from app.repositories import exam_repository

DUMMY_OCR_MEASUREMENTS = [
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
    _ = request
    return await exam_repository.confirm_exam_report(exam_report_id)


async def run_dummy_ocr(exam_report_id: int) -> ExamDummyOCRResponse:
    existing_measurements = await list_exam_measurements(exam_report_id)
    existing_by_key = {measurement.measurement_key: measurement for measurement in existing_measurements}
    saved_measurements: list[ExamMeasurement] = []

    for key, name, value, unit in DUMMY_OCR_MEASUREMENTS:
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
    return ExamDummyOCRResponse(
        message="자동 인식 측정값이 생성되었습니다. 검진 수치를 확인한 뒤 저장해주세요.",
        measurements=saved_measurements,
    )
