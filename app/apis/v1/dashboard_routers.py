from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies.security import get_request_user
from app.dtos.dashboard import (
    DashboardChallengesResponse,
    DashboardDietsResponse,
    DashboardHealthResponse,
    DashboardMedicationsResponse,
    DashboardSummaryResponse,
)
from app.models.users import User
from app.services import challenges as challenge_service
from app.services import diets as diet_service
from app.services import health as health_service
from app.services import medications as medication_service
from app.services import notifications as notification_service

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@dashboard_router.get("/summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary(user: Annotated[User, Depends(get_request_user)]):
    latest_health_record = await health_service.get_latest_health_record(user.id)
    unread_notifications = await notification_service.list_unread_notifications(user.id, limit=1000)
    active_challenges = await challenge_service.list_active_challenges(limit=1000)
    return {
        "latest_health_record": latest_health_record,
        "unread_notification_count": len(unread_notifications),
        "active_challenge_count": len(active_challenges),
    }


@dashboard_router.get("/health", response_model=DashboardHealthResponse)
async def get_dashboard_health(user: Annotated[User, Depends(get_request_user)]):
    return {
        "latest_health_record": await health_service.get_latest_health_record(user.id),
        "recent_health_records": await health_service.list_health_records(user.id, limit=5),
    }


@dashboard_router.get("/challenges", response_model=DashboardChallengesResponse)
async def get_dashboard_challenges(user: Annotated[User, Depends(get_request_user)]):
    return {
        "active_challenges": await challenge_service.list_active_challenges(limit=10),
        "user_challenges": await challenge_service.list_user_challenges(user.id, limit=10),
    }


@dashboard_router.get("/diets", response_model=DashboardDietsResponse)
async def get_dashboard_diets(user: Annotated[User, Depends(get_request_user)]):
    return {"recent_diet_records": await diet_service.list_diet_records(user.id, limit=10)}


@dashboard_router.get("/medications", response_model=DashboardMedicationsResponse)
async def get_dashboard_medications(user: Annotated[User, Depends(get_request_user)]):
    return {
        "active_medications": await medication_service.list_medications(user.id, is_active=True, limit=10),
        "recent_medication_records": await medication_service.list_medication_records(user_id=user.id, limit=10),
    }
