from typing import Any

from app.models.settings import UserSetting


async def create_user_setting(user_id: int, data: dict[str, Any]) -> UserSetting:
    return await UserSetting.create(user_id=user_id, **data)


async def get_user_setting_by_user(user_id: int) -> UserSetting | None:
    return await UserSetting.get_or_none(user_id=user_id)


async def get_user_setting_by_id(setting_id: int) -> UserSetting | None:
    return await UserSetting.get_or_none(id=setting_id)


async def update_user_setting(user_id: int, data: dict[str, Any]) -> UserSetting | None:
    setting = await get_user_setting_by_user(user_id)
    if setting is None:
        return None
    for key, value in data.items():
        setattr(setting, key, value)
    await setting.save(update_fields=list(data.keys()) if data else None)
    return setting


async def delete_user_setting(user_id: int) -> int:
    return await UserSetting.filter(user_id=user_id).delete()
