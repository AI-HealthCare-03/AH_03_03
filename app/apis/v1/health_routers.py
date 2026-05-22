from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from app.apis.v1.dependencies import ensure_found, ensure_owner, get_request_user
from app.dtos.health import (
    HealthAnalysisReadinessResponse,
    HealthRecordCreateRequest,
    HealthRecordResponse,
    HealthRecordUpdateRequest,
)
from app.models.users import User
from app.services import health as health_service
from app.services.sensitive_access_logs import safe_record_sensitive_access

health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.post("/records", response_model=HealthRecordResponse, status_code=status.HTTP_201_CREATED)
async def create_health_record(request: HealthRecordCreateRequest, user: Annotated[User, Depends(get_request_user)]):
    return await health_service.create_health_record(user.id, request)


@health_router.get("/records", response_model=list[HealthRecordResponse])
async def list_health_records(
    request: Request,
    user: Annotated[User, Depends(get_request_user)],
    limit: int = 20,
    offset: int = 0,
):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="HEALTH_RECORD",
        access_reason="health_records.list",
    )
    return await health_service.list_health_records(user.id, limit=limit, offset=offset)


@health_router.get("/records/latest", response_model=HealthRecordResponse | None)
async def get_latest_health_record(request: Request, user: Annotated[User, Depends(get_request_user)]):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="HEALTH_RECORD",
        access_reason="health_records.latest",
    )
    return await health_service.get_latest_health_record(user.id)


@health_router.get("/analysis-readiness", response_model=HealthAnalysisReadinessResponse)
async def get_analysis_readiness(user: Annotated[User, Depends(get_request_user)]):
    return await health_service.get_analysis_readiness(user.id)


@health_router.get("/records/{record_id}", response_model=HealthRecordResponse)
async def get_health_record(record_id: int, request: Request, user: Annotated[User, Depends(get_request_user)]):
    record = ensure_found(await health_service.get_health_record(record_id), "건강 기록을 찾을 수 없습니다.")
    ensure_owner(record.user_id, user)
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=record.user_id,
        resource_type="HEALTH_RECORD",
        resource_id=record.id,
        access_reason="health_records.detail",
    )
    return record


@health_router.patch("/records/{record_id}", response_model=HealthRecordResponse)
async def update_health_record(
    record_id: int,
    request: HealthRecordUpdateRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    record = ensure_found(await health_service.get_health_record(record_id), "건강 기록을 찾을 수 없습니다.")
    ensure_owner(record.user_id, user)
    updated = await health_service.update_health_record(record_id, request)
    return ensure_found(updated, "건강 기록을 찾을 수 없습니다.")


@health_router.delete("/records/{record_id}")
async def delete_health_record(record_id: int, user: Annotated[User, Depends(get_request_user)]):
    record = ensure_found(await health_service.get_health_record(record_id), "건강 기록을 찾을 수 없습니다.")
    ensure_owner(record.user_id, user)
    deleted_count = await health_service.delete_health_record(record_id)
    return {"deleted_count": deleted_count}
