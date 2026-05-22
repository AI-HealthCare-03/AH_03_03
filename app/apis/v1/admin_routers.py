from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.apis.v1.dependencies import ensure_found, require_admin_user, require_monitor_user, require_operator_user
from app.apis.v1.system_routers import get_system_health
from app.dtos.admin import (
    AdminSensitiveAccessLogListResponse,
    AdminSummaryResponse,
    AdminSystemErrorLogListResponse,
    AdminSystemHealthResponse,
    AdminUsersSummaryResponse,
)
from app.dtos.faqs import FAQCreateRequest, FAQResponse, FAQUpdateRequest, InquiryAnswerRequest, InquiryResponse
from app.models.users import User
from app.services import faqs as faq_service
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


@admin_router.get("/faqs", response_model=list[FAQResponse])
async def list_admin_faqs(
    _: Annotated[User, Depends(require_operator_user)],
    category: str | None = None,
    is_active: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[FAQResponse]:
    return await faq_service.list_faqs(category=category, is_active=is_active, limit=limit, offset=offset)


@admin_router.post("/faqs", response_model=FAQResponse, status_code=status.HTTP_201_CREATED)
async def create_admin_faq(
    request: FAQCreateRequest,
    _: Annotated[User, Depends(require_operator_user)],
) -> FAQResponse:
    return await faq_service.create_faq(request)


@admin_router.patch("/faqs/{faq_id}", response_model=FAQResponse)
async def update_admin_faq(
    faq_id: int,
    request: FAQUpdateRequest,
    _: Annotated[User, Depends(require_operator_user)],
) -> FAQResponse:
    ensure_found(await faq_service.get_faq(faq_id), "FAQ를 찾을 수 없습니다.")
    updated = await faq_service.update_faq(faq_id, request)
    return ensure_found(updated, "FAQ를 찾을 수 없습니다.")


@admin_router.delete("/faqs/{faq_id}", response_model=FAQResponse)
async def deactivate_admin_faq(
    faq_id: int,
    _: Annotated[User, Depends(require_operator_user)],
) -> FAQResponse:
    ensure_found(await faq_service.get_faq(faq_id), "FAQ를 찾을 수 없습니다.")
    deactivated = await faq_service.deactivate_faq(faq_id)
    return ensure_found(deactivated, "FAQ를 찾을 수 없습니다.")


@admin_router.get("/inquiries", response_model=list[InquiryResponse])
async def list_admin_inquiries(
    _: Annotated[User, Depends(require_operator_user)],
    status_filter: str | None = Query(default=None, alias="status"),
    category: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[InquiryResponse]:
    return await faq_service.list_inquiries(status=status_filter, category=category, limit=limit, offset=offset)


@admin_router.get("/inquiries/{inquiry_id}", response_model=InquiryResponse)
async def get_admin_inquiry(
    inquiry_id: int,
    _: Annotated[User, Depends(require_operator_user)],
) -> InquiryResponse:
    return ensure_found(await faq_service.get_inquiry(inquiry_id), "문의글을 찾을 수 없습니다.")


@admin_router.post("/inquiries/{inquiry_id}/answer", response_model=InquiryResponse)
async def answer_admin_inquiry(
    inquiry_id: int,
    request: InquiryAnswerRequest,
    _: Annotated[User, Depends(require_operator_user)],
) -> InquiryResponse:
    ensure_found(await faq_service.get_inquiry(inquiry_id), "문의글을 찾을 수 없습니다.")
    answered = await faq_service.answer_inquiry(inquiry_id, request.answer)
    return ensure_found(answered, "문의글을 찾을 수 없습니다.")
