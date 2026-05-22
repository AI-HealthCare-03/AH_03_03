from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from app.apis.v1.dependencies import ensure_found, ensure_owner, get_request_user
from app.dtos.exams import (
    ExamConfirmRequest,
    ExamDummyOCRResponse,
    ExamMeasurementCreateRequest,
    ExamMeasurementResponse,
    ExamMeasurementUpdateRequest,
    ExamOCRResponse,
    ExamReportCreateRequest,
    ExamReportResponse,
    ExamReportUpdateRequest,
)
from app.models.users import User
from app.services import exams as exam_service
from app.services.sensitive_access_logs import safe_record_sensitive_access

exam_router = APIRouter(prefix="/exams", tags=["exams"])


@exam_router.post("", response_model=ExamReportResponse, status_code=status.HTTP_201_CREATED)
async def create_exam_report(request: ExamReportCreateRequest, user: Annotated[User, Depends(get_request_user)]):
    return await exam_service.create_exam_report(user.id, request)


@exam_router.get("", response_model=list[ExamReportResponse])
async def list_exam_reports(
    request: Request,
    user: Annotated[User, Depends(get_request_user)],
    limit: int = 20,
    offset: int = 0,
):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="EXAM_REPORT",
        access_reason="exam_reports.list",
    )
    return await exam_service.list_exam_reports(user.id, limit=limit, offset=offset)


@exam_router.get("/{exam_id}", response_model=ExamReportResponse)
async def get_exam_report(exam_id: int, request: Request, user: Annotated[User, Depends(get_request_user)]):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=report.user_id,
        resource_type="EXAM_REPORT",
        resource_id=report.id,
        access_reason="exam_reports.detail",
    )
    return report


async def _run_exam_ocr(exam_id: int, user: User) -> ExamDummyOCRResponse:
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    return await exam_service.run_dummy_ocr(exam_id)


@exam_router.post("/{exam_id}/ocr", response_model=ExamOCRResponse)
async def run_exam_ocr(exam_id: int, user: Annotated[User, Depends(get_request_user)]):
    return await _run_exam_ocr(exam_id, user)


@exam_router.post("/{exam_id}/dummy-ocr", response_model=ExamDummyOCRResponse, deprecated=True)
async def run_dummy_ocr(exam_id: int, user: Annotated[User, Depends(get_request_user)]):
    return await _run_exam_ocr(exam_id, user)


@exam_router.patch("/{exam_id}", response_model=ExamReportResponse)
async def update_exam_report(
    exam_id: int,
    request: ExamReportUpdateRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    updated = await exam_service.update_exam_report(exam_id, request)
    return ensure_found(updated, "검진표를 찾을 수 없습니다.")


@exam_router.post(
    "/{exam_id}/measurements", response_model=ExamMeasurementResponse, status_code=status.HTTP_201_CREATED
)
async def create_exam_measurement(
    exam_id: int,
    request: ExamMeasurementCreateRequest,
    user: Annotated[User, Depends(get_request_user)],
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
    user: Annotated[User, Depends(get_request_user)],
):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    return await exam_service.create_exam_measurements(exam_id, measurements)


@exam_router.get("/{exam_id}/measurements", response_model=list[ExamMeasurementResponse])
async def list_exam_measurements(exam_id: int, request: Request, user: Annotated[User, Depends(get_request_user)]):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=report.user_id,
        resource_type="EXAM_MEASUREMENT",
        resource_id=report.id,
        access_reason="exam_measurements.list",
    )
    return await exam_service.list_exam_measurements(exam_id)


@exam_router.patch("/measurements/{measurement_id}", response_model=ExamMeasurementResponse)
async def update_exam_measurement(
    measurement_id: int,
    request: ExamMeasurementUpdateRequest,
    user: Annotated[User, Depends(get_request_user)],
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
    user: Annotated[User, Depends(get_request_user)],
    request: ExamConfirmRequest | None = None,
):
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    confirmed = await exam_service.confirm_exam_report(exam_id, request)
    return ensure_found(confirmed, "검진표를 찾을 수 없습니다.")
