from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel

from app.dtos.base import BaseSerializerModel
from app.models.exams import OCRStatus


class ExamReportCreateRequest(BaseModel):
    original_filename: str
    file_path: str
    exam_date: date | None = None
    ocr_status: OCRStatus = OCRStatus.PENDING
    uploaded_at: datetime


class ExamReportResponse(BaseSerializerModel):
    id: int
    user_id: int
    original_filename: str
    file_path: str
    exam_date: date | None
    ocr_status: OCRStatus
    is_confirmed: bool
    uploaded_at: datetime
    confirmed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ExamMeasurementCreateRequest(BaseModel):
    measurement_key: str
    measurement_name: str
    value: str | None = None
    unit: str | None = None
    ocr_confidence: Decimal | None = None
    is_user_confirmed: bool = False


class ExamMeasurementUpdateRequest(BaseModel):
    measurement_key: str | None = None
    measurement_name: str | None = None
    value: str | None = None
    unit: str | None = None
    ocr_confidence: Decimal | None = None
    is_user_confirmed: bool | None = None


class ExamMeasurementResponse(BaseSerializerModel):
    id: int
    exam_report_id: int
    measurement_key: str
    measurement_name: str
    value: str | None
    unit: str | None
    ocr_confidence: Decimal | None
    is_user_confirmed: bool
    created_at: datetime
    updated_at: datetime


class ExamReportDetailResponse(ExamReportResponse):
    measurements: list[ExamMeasurementResponse]


class ExamConfirmRequest(BaseModel):
    measurements: list[ExamMeasurementUpdateRequest] | None = None
