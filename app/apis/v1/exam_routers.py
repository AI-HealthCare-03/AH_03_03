from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from ai_runtime.common.image_normalizer import ImageNormalizationError, normalize_upload_image
from app.apis.v1.dependencies import ensure_found, ensure_owner, get_request_user
from app.dtos.exams import (
    ExamConfirmRequest,
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


async def _run_exam_ocr(
    exam_id: int,
    user: User,
    image_bytes: bytes | None = None,
    image_media_type: str | None = None,
    image_filename: str | None = None,
) -> ExamOCRResponse:
    report = ensure_found(await exam_service.get_exam_report(exam_id), "검진표를 찾을 수 없습니다.")
    ensure_owner(report.user_id, user)
    return await exam_service.run_exam_ocr(
        exam_id,
        image_bytes=image_bytes,
        image_media_type=image_media_type,
        image_filename=image_filename,
    )


@exam_router.post("/{exam_id}/ocr", response_model=ExamOCRResponse)
async def run_exam_ocr(exam_id: int, request: Request, user: Annotated[User, Depends(get_request_user)]):
    image_bytes, image_media_type, image_filename = await _read_optional_upload(request)
    if image_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="검진표 이미지 또는 PDF 파일을 업로드해주세요.",
        )
    return await _run_exam_ocr(
        exam_id,
        user,
        image_bytes=image_bytes,
        image_media_type=image_media_type,
        image_filename=image_filename,
    )


@exam_router.post("/{exam_id}/dummy-ocr", response_model=ExamOCRResponse, deprecated=True, include_in_schema=False)
async def run_legacy_exam_ocr(exam_id: int, user: Annotated[User, Depends(get_request_user)]):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="더미 OCR 경로는 사용하지 않습니다. 실제 검진표 파일 업로드 경로를 사용해주세요.",
    )


async def _read_optional_upload(request: Request) -> tuple[bytes | None, str | None, str | None]:
    if "multipart/form-data" not in request.headers.get("content-type", ""):
        return None, None, None
    form = await request.form()
    upload = form.get("image") or form.get("file")
    if _is_upload(upload):
        filename = getattr(upload, "filename", None)
        normalized_image = _normalize_uploaded_image(await upload.read(), upload.content_type, filename)
        return normalized_image.data, normalized_image.media_type, filename
    return None, None, None


def _is_upload(value: object) -> bool:
    return isinstance(value, UploadFile) or (hasattr(value, "read") and hasattr(value, "filename"))


def _normalize_uploaded_image(image_bytes: bytes, media_type: str | None, filename: str | None):
    try:
        return normalize_upload_image(image_bytes, media_type, filename)
    except ImageNormalizationError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc


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
