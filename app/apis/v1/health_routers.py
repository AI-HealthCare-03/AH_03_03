from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.dependencies.security import get_request_user
from app.dtos.health import HealthRecordCreateRequest, HealthRecordResponse, HealthRecordUpdateRequest
from app.models.users import User
from app.services import health as health_service

health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.post("/records", response_model=HealthRecordResponse, status_code=status.HTTP_201_CREATED)
async def create_health_record(request: HealthRecordCreateRequest, user: Annotated[User, Depends(get_request_user)]):
    return await health_service.create_health_record(user.id, request)


@health_router.get("/records", response_model=list[HealthRecordResponse])
async def list_health_records(user: Annotated[User, Depends(get_request_user)], limit: int = 20, offset: int = 0):
    return await health_service.list_health_records(user.id, limit=limit, offset=offset)


@health_router.get("/records/latest", response_model=HealthRecordResponse | None)
async def get_latest_health_record(user: Annotated[User, Depends(get_request_user)]):
    return await health_service.get_latest_health_record(user.id)


@health_router.get("/records/{record_id}", response_model=HealthRecordResponse | None)
async def get_health_record(record_id: int):
    return await health_service.get_health_record(record_id)


@health_router.patch("/records/{record_id}", response_model=HealthRecordResponse | None)
async def update_health_record(record_id: int, request: HealthRecordUpdateRequest):
    return await health_service.update_health_record(record_id, request)


@health_router.delete("/records/{record_id}")
async def delete_health_record(record_id: int):
    deleted_count = await health_service.delete_health_record(record_id)
    return {"deleted_count": deleted_count}
