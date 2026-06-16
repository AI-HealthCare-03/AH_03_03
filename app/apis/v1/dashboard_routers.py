from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.apis.v1.dependencies import get_request_user
from app.dtos.dashboard import (
    DashboardChallengesResponse,
    DashboardDietsResponse,
    DashboardHealthResponse,
    DashboardMedicationsResponse,
    DashboardRiskTrendResponse,
    DashboardSummaryResponse,
    DashboardTrendsResponse,
)
from app.models.users import User
from app.services import dashboard as dashboard_service
from app.services.sensitive_access_logs import safe_record_sensitive_access

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@dashboard_router.get("/summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary(request: Request, user: Annotated[User, Depends(get_request_user)]):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="DASHBOARD",
        access_reason="dashboard.summary",
    )
    return await dashboard_service.get_dashboard_summary(user.id)


@dashboard_router.get("/health", response_model=DashboardHealthResponse)
async def get_dashboard_health(request: Request, user: Annotated[User, Depends(get_request_user)]):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="DASHBOARD",
        access_reason="dashboard.health",
    )
    return await dashboard_service.get_dashboard_health(user.id)


@dashboard_router.get("/challenges", response_model=DashboardChallengesResponse)
async def get_dashboard_challenges(user: Annotated[User, Depends(get_request_user)]):
    return await dashboard_service.get_dashboard_challenges(user.id)


@dashboard_router.get("/diets", response_model=DashboardDietsResponse)
async def get_dashboard_diets(user: Annotated[User, Depends(get_request_user)]):
    return await dashboard_service.get_dashboard_diets(user.id)


@dashboard_router.get("/medications", response_model=DashboardMedicationsResponse)
async def get_dashboard_medications(user: Annotated[User, Depends(get_request_user)]):
    return await dashboard_service.get_dashboard_medications(user.id)


@dashboard_router.get("/trends", response_model=DashboardTrendsResponse)
async def get_dashboard_trends(
    request: Request,
    user: Annotated[User, Depends(get_request_user)],
    period: str = "week",
):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="DASHBOARD",
        access_reason="dashboard.trends",
    )
    try:
        return await dashboard_service.get_dashboard_trends(user.id, period)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@dashboard_router.get("/risk-trend", response_model=DashboardRiskTrendResponse)
async def get_dashboard_risk_trend(
    request: Request,
    user: Annotated[User, Depends(get_request_user)],
    period: str = "all",
):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="DASHBOARD",
        access_reason="dashboard.risk_trend",
    )
    try:
        return await dashboard_service.get_dashboard_risk_trend(user.id, period)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
