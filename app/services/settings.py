from app.dtos.settings import UserSettingCreateRequest, UserSettingUpdateRequest
from app.models.settings import UserSetting
from app.repositories import setting_repository


def _normalize_time_value(value):
    if hasattr(value, "tzinfo") and value.tzinfo is not None:
        return value.replace(tzinfo=None)
    return value


def _normalize_setting_data(data: dict) -> dict:
    normalized = dict(data)
    for key in ("challenge_reminder_time", "diet_reminder_time"):
        if key in normalized:
            normalized[key] = _normalize_time_value(normalized[key])
    return normalized


async def get_user_settings(user_id: int) -> UserSetting | None:
    return await setting_repository.get_user_setting_by_user(user_id)


async def get_or_create_user_settings(user_id: int) -> UserSetting:
    setting = await get_user_settings(user_id)
    if setting is not None:
        return setting
    return await setting_repository.create_user_setting(user_id, UserSettingCreateRequest().model_dump())


async def update_user_settings(user_id: int, request: UserSettingUpdateRequest) -> UserSetting | None:
    data = _normalize_setting_data(request.model_dump(exclude_unset=True))
    await get_or_create_user_settings(user_id)
    setting = await setting_repository.update_user_setting(user_id, data)
    if setting is None:
        return None

    from app.services import notifications as notification_service

    await notification_service.sync_all_user_reminder_schedules(user_id)
    return setting
