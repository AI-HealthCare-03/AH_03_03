from typing import Annotated

from fastapi import APIRouter, Depends

from app.apis.v1.dependencies import get_request_user_with_firebase
from app.dtos.mypage import MyPageSummaryResponse
from app.models.users import User
from app.services import health as health_service
from app.services import notifications as notification_service
from app.services import settings as setting_service

mypage_router = APIRouter(prefix="/mypage", tags=["mypage"])


@mypage_router.get("/summary", response_model=MyPageSummaryResponse)
async def get_mypage_summary(user: Annotated[User, Depends(get_request_user_with_firebase)]):
    unread_notifications = await notification_service.list_unread_notifications(user.id, limit=1000)
    return {
        "user": user,
        "settings": await setting_service.get_user_settings(user.id),
        "latest_health_record": await health_service.get_latest_health_record(user.id),
        "unread_notification_count": len(unread_notifications),
    }
