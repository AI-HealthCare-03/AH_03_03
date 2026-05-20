from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.apis.v1.dependencies import ensure_found, ensure_owner, get_request_user_with_firebase
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
async def create_exam_report(
    request: ExamReportCreateRequest, user: Annotated[User, Depends(get_request_user_with_firebase)]
):
    return await exam_service.create_exam_report(user.id, request)


@exam_router.get("", response_model=list[ExamReportResponse])
async def list_exam_reports(
    user: Annotated[User, Depends(get_request_user_with_firebase)], limit: int = 20, offset: int = 0
):
    return await exam_service.list_exam_reports(user.id, limit=limit, offset=offset)


@exam_router.get("/{exam_id}", response_model=ExamReportResponse)
async def get_exam_report(exam_id: int, user: Annotated[User, Depends(get_request_user_with_firebase)]):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    return report


@exam_router.patch("/{exam_id}", response_model=ExamReportResponse)
async def update_exam_report(exam_id: int, data: dict, user: Annotated[User, Depends(get_request_user_with_firebase)]):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    updated = await exam_service.update_exam_report(exam_id, data)
    return ensure_found(updated, "검진표를 찾을 수 없습니다.")


@exam_router.post(
    "/{exam_id}/measurements", response_model=ExamMeasurementResponse, status_code=status.HTTP_201_CREATED
)
async def create_exam_measurement(
    exam_id: int,
    request: ExamMeasurementCreateRequest,
    user: Annotated[User, Depends(get_request_user_with_firebase)],
):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    return await exam_service.create_exam_measurement(exam_id, request)


@exam_router.post(
    "/{exam_id}/measurements/bulk",
    response_model=list[ExamMeasurementResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_exam_measurements(
    exam_id: int,
    measurements: list[ExamMeasurementCreateRequest],
    user: Annotated[User, Depends(get_request_user_with_firebase)],
):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    return await exam_service.create_exam_measurements(exam_id, measurements)


@exam_router.get("/{exam_id}/measurements", response_model=list[ExamMeasurementResponse])
async def list_exam_measurements(exam_id: int, user: Annotated[User, Depends(get_request_user_with_firebase)]):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    return await exam_service.list_exam_measurements(exam_id)


@exam_router.patch("/measurements/{measurement_id}", response_model=ExamMeasurementResponse)
async def update_exam_measurement(
    measurement_id: int,
    request: ExamMeasurementUpdateRequest,
    user: Annotated[User, Depends(get_request_user_with_firebase)],
):
    measurement = ensure_found(
        await exam_service.get_exam_measurement(measurement_id), "검진 측정값을 찾을 수 없습니다."
    )
    await measurement.fetch_related("exam_report")
    ensure_owner(measurement.exam_report.user_id, user)
    updated = await exam_service.update_exam_measurement(measurement_id, request)
    return ensure_found(updated, "검진 측정값을 찾을 수 없습니다.")


@exam_router.post("/{exam_id}/confirm", response_model=ExamReportResponse)
async def confirm_exam_report(
    exam_id: int,
    user: Annotated[User, Depends(get_request_user_with_firebase)],
    request: ExamConfirmRequest | None = None,
):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    confirmed = await exam_service.confirm_exam_report(exam_id, request)
    return ensure_found(confirmed, "검진표를 찾을 수 없습니다.")
