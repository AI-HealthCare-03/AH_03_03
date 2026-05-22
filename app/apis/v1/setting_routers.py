from typing import Annotated

from fastapi import APIRouter, Depends

from app.apis.v1.dependencies import ensure_found, get_request_user
from app.dtos.settings import UserSettingResponse, UserSettingUpdateRequest
from app.models.users import User
from app.services import settings as setting_service

setting_router = APIRouter(prefix="/settings", tags=["settings"])


@setting_router.get("/me", response_model=UserSettingResponse)
async def get_user_settings(user: Annotated[User, Depends(get_request_user)]):
    return await setting_service.get_or_create_user_settings(user.id)


@setting_router.patch("/me", response_model=UserSettingResponse)
async def update_user_settings(request: UserSettingUpdateRequest, user: Annotated[User, Depends(get_request_user)]):
    setting = await setting_service.update_user_settings(user.id, request)
    return ensure_found(setting, "사용자 설정을 찾을 수 없습니다.")
