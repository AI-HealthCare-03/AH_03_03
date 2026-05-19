from app.dtos.settings import UserSettingCreateRequest, UserSettingUpdateRequest
from app.models.settings import UserSetting
from app.repositories import setting_repository


async def get_user_settings(user_id: int) -> UserSetting | None:
    return await setting_repository.get_user_setting_by_user(user_id)


async def get_or_create_user_settings(user_id: int) -> UserSetting:
    setting = await get_user_settings(user_id)
    if setting is not None:
        return setting
    return await setting_repository.create_user_setting(user_id, UserSettingCreateRequest().model_dump())


async def update_user_settings(user_id: int, request: UserSettingUpdateRequest) -> UserSetting | None:
    return await setting_repository.update_user_setting(user_id, request.model_dump(exclude_unset=True))
