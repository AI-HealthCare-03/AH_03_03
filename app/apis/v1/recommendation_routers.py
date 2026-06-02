from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.apis.v1.dependencies import get_request_user
from app.dtos.recommendations import TodayRecommendationsResponse
from app.models.users import User
from app.services import recommendations as recommendation_service
from app.services.sensitive_access_logs import safe_record_sensitive_access

recommendation_router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@recommendation_router.get("/today", response_model=TodayRecommendationsResponse)
async def get_today_recommendations(request: Request, user: Annotated[User, Depends(get_request_user)]):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="RECOMMENDATION",
        access_reason="recommendations.today",
    )
    return await recommendation_service.get_today_recommendations(user.id)
