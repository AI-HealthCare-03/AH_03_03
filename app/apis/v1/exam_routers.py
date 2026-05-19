from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.dependencies.security import get_request_user
from app.dtos.exams import (
    ExamConfirmRequest,
    ExamMeasurementCreateRequest,
    ExamMeasurementResponse,
    ExamMeasurementUpdateRequest,
    ExamReportCreateRequest,
    ExamReportResponse,
)
from app.models.users import User
from app.services import exams as exam_service

exam_router = APIRouter(prefix="/exams", tags=["exams"])


@exam_router.post("", response_model=ExamReportResponse, status_code=status.HTTP_201_CREATED)
async def create_exam_report(request: ExamReportCreateRequest, user: Annotated[User, Depends(get_request_user)]):
    return await exam_service.create_exam_report(user.id, request)


@exam_router.get("", response_model=list[ExamReportResponse])
async def list_exam_reports(user: Annotated[User, Depends(get_request_user)], limit: int = 20, offset: int = 0):
    return await exam_service.list_exam_reports(user.id, limit=limit, offset=offset)


@exam_router.get("/{exam_id}", response_model=ExamReportResponse | None)
async def get_exam_report(exam_id: int):
    return await exam_service.get_exam_report(exam_id)


@exam_router.patch("/{exam_id}", response_model=ExamReportResponse | None)
async def update_exam_report(exam_id: int, data: dict):
    return await exam_service.update_exam_report(exam_id, data)


@exam_router.post(
    "/{exam_id}/measurements", response_model=ExamMeasurementResponse, status_code=status.HTTP_201_CREATED
)
async def create_exam_measurement(exam_id: int, request: ExamMeasurementCreateRequest):
    return await exam_service.create_exam_measurement(exam_id, request)


@exam_router.post(
    "/{exam_id}/measurements/bulk",
    response_model=list[ExamMeasurementResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_exam_measurements(exam_id: int, measurements: list[ExamMeasurementCreateRequest]):
    return await exam_service.create_exam_measurements(exam_id, measurements)


@exam_router.get("/{exam_id}/measurements", response_model=list[ExamMeasurementResponse])
async def list_exam_measurements(exam_id: int):
    return await exam_service.list_exam_measurements(exam_id)


@exam_router.patch("/measurements/{measurement_id}", response_model=ExamMeasurementResponse | None)
async def update_exam_measurement(measurement_id: int, request: ExamMeasurementUpdateRequest):
    return await exam_service.update_exam_measurement(measurement_id, request)


@exam_router.post("/{exam_id}/confirm", response_model=ExamReportResponse | None)
async def confirm_exam_report(exam_id: int, request: ExamConfirmRequest | None = None):
    return await exam_service.confirm_exam_report(exam_id, request)
