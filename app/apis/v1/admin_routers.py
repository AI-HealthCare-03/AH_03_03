from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.apis.v1.dependencies import require_admin_user, require_monitor_user
from app.apis.v1.system_routers import get_system_health
from app.dtos.admin import (
    AdminSensitiveAccessLogListResponse,
    AdminSummaryResponse,
    AdminSystemErrorLogListResponse,
    AdminSystemHealthResponse,
    AdminUsersSummaryResponse,
)
from app.models.users import User
from app.services.admin_monitoring import AdminMonitoringService

admin_router = APIRouter(prefix="/admin", tags=["admin"])


def get_admin_monitoring_service() -> AdminMonitoringService:
    return AdminMonitoringService()


AdminMonitoringServiceDep = Annotated[AdminMonitoringService, Depends(get_admin_monitoring_service)]


@admin_router.get("/summary", response_model=AdminSummaryResponse)
async def get_admin_summary(
    _: Annotated[User, Depends(require_monitor_user)],
    service: AdminMonitoringServiceDep,
) -> AdminSummaryResponse:
    return await service.get_summary()


@admin_router.get("/system/health", response_model=AdminSystemHealthResponse)
async def get_admin_system_health(
    _: Annotated[User, Depends(require_monitor_user)],
) -> dict[str, object]:
    return await get_system_health()


@admin_router.get("/system/errors", response_model=AdminSystemErrorLogListResponse)
async def list_admin_system_errors(
    _: Annotated[User, Depends(require_monitor_user)],
    service: AdminMonitoringServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    date_from: date | None = None,
    date_to: date | None = None,
    status_code: int | None = None,
) -> AdminSystemErrorLogListResponse:
    items, total, safe_limit = await service.list_system_errors(
        limit=limit,
        date_from=date_from,
        date_to=date_to,
        status_code=status_code,
    )
    return AdminSystemErrorLogListResponse(
        items=items,
        total=total,
        limit=safe_limit,
        filters={
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
            "status_code": status_code,
        },
    )


@admin_router.get("/sensitive-access-logs", response_model=AdminSensitiveAccessLogListResponse)
async def list_admin_sensitive_access_logs(
    _: Annotated[User, Depends(require_admin_user)],
    service: AdminMonitoringServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    date_from: date | None = None,
    date_to: date | None = None,
    resource_type: str | None = None,
) -> AdminSensitiveAccessLogListResponse:
    items, total, safe_limit = await service.list_sensitive_access_logs(
        limit=limit,
        date_from=date_from,
        date_to=date_to,
        resource_type=resource_type,
    )
    return AdminSensitiveAccessLogListResponse(
        items=items,
        total=total,
        limit=safe_limit,
        filters={
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
            "resource_type": resource_type,
        },
    )


@admin_router.get("/users/summary", response_model=AdminUsersSummaryResponse)
async def get_admin_users_summary(
    _: Annotated[User, Depends(require_monitor_user)],
    service: AdminMonitoringServiceDep,
) -> AdminUsersSummaryResponse:
    return await service.get_users_summary()
