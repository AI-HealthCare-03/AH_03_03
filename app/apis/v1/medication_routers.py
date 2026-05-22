from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from app.apis.v1.dependencies import ensure_found, ensure_owner, get_request_user
from app.dtos.medications import (
    MedicationCreateRequest,
    MedicationOCRConfirmRequest,
    MedicationOCRConfirmResponse,
    MedicationOCRDummyRequest,
    MedicationOCRDummyResponse,
    MedicationRecordCreateRequest,
    MedicationRecordResponse,
    MedicationRecordUpdateRequest,
    MedicationResponse,
    MedicationUpdateRequest,
)
from app.models.users import User
from app.services import medications as medication_service
from app.services.sensitive_access_logs import safe_record_sensitive_access

medication_router = APIRouter(prefix="/medications", tags=["medications"])


@medication_router.post("", response_model=MedicationResponse, status_code=status.HTTP_201_CREATED)
async def create_medication(request: MedicationCreateRequest, user: Annotated[User, Depends(get_request_user)]):
    return await medication_service.create_medication(user.id, request)


@medication_router.get("", response_model=list[MedicationResponse])
async def list_medications(
    request: Request,
    user: Annotated[User, Depends(get_request_user)],
    is_active: bool | None = None,
    medication_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="MEDICATION",
        access_reason="medications.list",
    )
    return await medication_service.list_medications(
        user_id=user.id,
        is_active=is_active,
        medication_type=medication_type,
        limit=limit,
        offset=offset,
    )


@medication_router.post("/dummy-ocr", response_model=MedicationOCRDummyResponse)
async def run_dummy_medication_ocr(
    request: MedicationOCRDummyRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    _ = user
    return await medication_service.run_dummy_medication_ocr(request)


@medication_router.post(
    "/ocr-confirm",
    response_model=MedicationOCRConfirmResponse,
    status_code=status.HTTP_201_CREATED,
)
async def confirm_medication_ocr(
    request: MedicationOCRConfirmRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    return await medication_service.confirm_medication_ocr(user.id, request)


@medication_router.get("/{medication_id}", response_model=MedicationResponse)
async def get_medication(medication_id: int, request: Request, user: Annotated[User, Depends(get_request_user)]):
    medication = ensure_found(
        await medication_service.get_medication(medication_id), "복약/영양제 정보를 찾을 수 없습니다."
    )
    ensure_owner(medication.user_id, user)
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=medication.user_id,
        resource_type="MEDICATION",
        resource_id=medication.id,
        access_reason="medications.detail",
    )
    return medication


@medication_router.patch("/{medication_id}", response_model=MedicationResponse)
async def update_medication(
    medication_id: int,
    request: MedicationUpdateRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    medication = ensure_found(
        await medication_service.get_medication(medication_id), "복약/영양제 정보를 찾을 수 없습니다."
    )
    ensure_owner(medication.user_id, user)
    updated = await medication_service.update_medication(medication_id, request)
    return ensure_found(updated, "복약/영양제 정보를 찾을 수 없습니다.")


@medication_router.patch("/{medication_id}/deactivate", response_model=MedicationResponse)
async def deactivate_medication(medication_id: int, user: Annotated[User, Depends(get_request_user)]):
    medication = ensure_found(
        await medication_service.get_medication(medication_id), "복약/영양제 정보를 찾을 수 없습니다."
    )
    ensure_owner(medication.user_id, user)
    deactivated = await medication_service.deactivate_medication(medication_id)
    return ensure_found(deactivated, "복약/영양제 정보를 찾을 수 없습니다.")


@medication_router.delete("/{medication_id}")
async def delete_medication(medication_id: int, user: Annotated[User, Depends(get_request_user)]):
    medication = ensure_found(
        await medication_service.get_medication(medication_id), "복약/영양제 정보를 찾을 수 없습니다."
    )
    ensure_owner(medication.user_id, user)
    deleted_count = await medication_service.delete_medication(medication_id)
    return {"deleted_count": deleted_count}


@medication_router.post(
    "/{medication_id}/records", response_model=MedicationRecordResponse, status_code=status.HTTP_201_CREATED
)
async def create_medication_record(
    medication_id: int,
    request: MedicationRecordCreateRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    medication = ensure_found(
        await medication_service.get_medication(medication_id), "복약/영양제 정보를 찾을 수 없습니다."
    )
    ensure_owner(medication.user_id, user)
    return await medication_service.create_medication_record(medication_id, user.id, request)


@medication_router.get("/{medication_id}/records", response_model=list[MedicationRecordResponse])
async def list_medication_records(
    medication_id: int,
    request: Request,
    user: Annotated[User, Depends(get_request_user)],
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    medication = ensure_found(
        await medication_service.get_medication(medication_id), "복약/영양제 정보를 찾을 수 없습니다."
    )
    ensure_owner(medication.user_id, user)
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=medication.user_id,
        resource_type="MEDICATION",
        resource_id=medication.id,
        access_reason="medication_records.list",
    )
    return await medication_service.list_medication_records(
        user_id=user.id,
        medication_id=medication_id,
        status=status,
        limit=limit,
        offset=offset,
    )


@medication_router.patch("/records/{record_id}", response_model=MedicationRecordResponse)
async def update_medication_record(
    record_id: int,
    request: MedicationRecordUpdateRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    ensure_found(
        await medication_service.get_medication_record(record_id, user_id=user.id),
        "복약 기록을 찾을 수 없습니다.",
    )
    updated = await medication_service.update_medication_record(record_id, user.id, request)
    return ensure_found(updated, "복약 기록을 찾을 수 없습니다.")


@medication_router.delete("/records/{record_id}")
async def delete_medication_record(record_id: int, user: Annotated[User, Depends(get_request_user)]):
    ensure_found(
        await medication_service.get_medication_record(record_id, user_id=user.id),
        "복약 기록을 찾을 수 없습니다.",
    )
    deleted_count = await medication_service.delete_medication_record(record_id, user.id)
    return {"deleted_count": deleted_count}
