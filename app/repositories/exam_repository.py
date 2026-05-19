from datetime import datetime
from typing import Any

from app.core import config
from app.models.exams import ExamMeasurement, ExamReport, OCRStatus


async def create_exam_report(user_id: int, data: dict[str, Any]) -> ExamReport:
    return await ExamReport.create(user_id=user_id, **data)


async def get_exam_report_by_id(exam_report_id: int) -> ExamReport | None:
    return await ExamReport.get_or_none(id=exam_report_id)


async def list_exam_reports_by_user(user_id: int, limit: int = 20, offset: int = 0) -> list[ExamReport]:
    return await ExamReport.filter(user_id=user_id).order_by("-uploaded_at").offset(offset).limit(limit)


async def update_exam_report(exam_report_id: int, data: dict[str, Any]) -> ExamReport | None:
    report = await get_exam_report_by_id(exam_report_id)
    if report is None:
        return None

    for key, value in data.items():
        setattr(report, key, value)
    await report.save(update_fields=list(data.keys()) if data else None)
    return report


async def create_exam_measurement(exam_report_id: int, data: dict[str, Any]) -> ExamMeasurement:
    return await ExamMeasurement.create(exam_report_id=exam_report_id, **data)


async def get_exam_measurement_by_id(measurement_id: int) -> ExamMeasurement | None:
    return await ExamMeasurement.get_or_none(id=measurement_id)


async def create_exam_measurements(exam_report_id: int, measurements: list[dict[str, Any]]) -> list[ExamMeasurement]:
    objects = [ExamMeasurement(exam_report_id=exam_report_id, **measurement) for measurement in measurements]
    if not objects:
        return []
    await ExamMeasurement.bulk_create(objects)
    return objects


async def list_exam_measurements(exam_report_id: int) -> list[ExamMeasurement]:
    return await ExamMeasurement.filter(exam_report_id=exam_report_id).order_by("id")


async def update_exam_measurement(measurement_id: int, data: dict[str, Any]) -> ExamMeasurement | None:
    measurement = await get_exam_measurement_by_id(measurement_id)
    if measurement is None:
        return None

    for key, value in data.items():
        setattr(measurement, key, value)
    await measurement.save(update_fields=list(data.keys()) if data else None)
    return measurement


async def confirm_exam_report(exam_report_id: int) -> ExamReport | None:
    report = await get_exam_report_by_id(exam_report_id)
    if report is None:
        return None

    report.is_confirmed = True
    report.confirmed_at = datetime.now(config.TIMEZONE)
    report.ocr_status = OCRStatus.CONFIRMED
    await report.save(update_fields=["is_confirmed", "confirmed_at", "ocr_status"])
    return report
