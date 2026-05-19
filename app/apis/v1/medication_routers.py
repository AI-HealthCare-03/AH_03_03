from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.apis.v1.dependencies import ensure_found, ensure_owner
from app.dependencies.security import get_request_user
from app.dtos.medications import (
    MedicationCreateRequest,
    MedicationRecordCreateRequest,
    MedicationRecordResponse,
    MedicationRecordUpdateRequest,
    MedicationResponse,
    MedicationUpdateRequest,
)
from app.models.users import User
from app.services import medications as medication_service

medication_router = APIRouter(prefix="/medications", tags=["medications"])


@medication_router.post("", response_model=MedicationResponse, status_code=status.HTTP_201_CREATED)
async def create_medication(request: MedicationCreateRequest, user: Annotated[User, Depends(get_request_user)]):
    return await medication_service.create_medication(user.id, request)


@medication_router.get("", response_model=list[MedicationResponse])
async def list_medications(
    user: Annotated[User, Depends(get_request_user)],
    is_active: bool | None = None,
    medication_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    return await medication_service.list_medications(
        user_id=user.id,
        is_active=is_active,
        medication_type=medication_type,
        limit=limit,
        offset=offset,
    )


@medication_router.get("/{medication_id}", response_model=MedicationResponse)
async def get_medication(medication_id: int, user: Annotated[User, Depends(get_request_user)]):
    medication = ensure_found(
        await medication_service.get_medication(medication_id), "복약/영양제 정보를 찾을 수 없습니다."
    )
    ensure_owner(medication.user_id, user)
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
    user: Annotated[User, Depends(get_request_user)],
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
):
    medication = ensure_found(
        await medication_service.get_medication(medication_id), "복약/영양제 정보를 찾을 수 없습니다."
    )
    ensure_owner(medication.user_id, user)
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
    record = ensure_found(await medication_service.get_medication_record(record_id), "복약 기록을 찾을 수 없습니다.")
    ensure_owner(record.user_id, user)
    updated = await medication_service.update_medication_record(record_id, request)
    return ensure_found(updated, "복약 기록을 찾을 수 없습니다.")
