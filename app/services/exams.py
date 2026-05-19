from typing import Any

from app.dtos.exams import (
    ExamConfirmRequest,
    ExamMeasurementCreateRequest,
    ExamMeasurementUpdateRequest,
    ExamReportCreateRequest,
)
from app.models.exams import ExamMeasurement, ExamReport
from app.repositories import exam_repository


async def create_exam_report(user_id: int, request: ExamReportCreateRequest) -> ExamReport:
    return await exam_repository.create_exam_report(user_id, request.model_dump())


async def get_exam_report(exam_report_id: int) -> ExamReport | None:
    return await exam_repository.get_exam_report_by_id(exam_report_id)


async def list_exam_reports(user_id: int, limit: int = 20, offset: int = 0) -> list[ExamReport]:
    return await exam_repository.list_exam_reports_by_user(user_id, limit=limit, offset=offset)


async def update_exam_report(exam_report_id: int, data: dict[str, Any]) -> ExamReport | None:
    return await exam_repository.update_exam_report(exam_report_id, data)


async def create_exam_measurement(exam_report_id: int, request: ExamMeasurementCreateRequest) -> ExamMeasurement:
    return await exam_repository.create_exam_measurement(exam_report_id, request.model_dump())


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
