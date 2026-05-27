from datetime import datetime, time

from pydantic import BaseModel, ConfigDict


class UserSettingCreateRequest(BaseModel):
    notification_enabled: bool = True
    challenge_reminder_enabled: bool = True
    challenge_reminder_time: time | None = None
    medication_reminder_enabled: bool = True
    diet_reminder_enabled: bool = False
    marketing_agreed: bool = False
    sensitive_data_agreed: bool = False


class UserSettingUpdateRequest(BaseModel):
    notification_enabled: bool | None = None
    challenge_reminder_enabled: bool | None = None
    challenge_reminder_time: time | None = None
    medication_reminder_enabled: bool | None = None
    diet_reminder_enabled: bool | None = None
    marketing_agreed: bool | None = None
    sensitive_data_agreed: bool | None = None


class UserSettingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    notification_enabled: bool
    challenge_reminder_enabled: bool
    challenge_reminder_time: time | None
    medication_reminder_enabled: bool
    diet_reminder_enabled: bool
    marketing_agreed: bool
    sensitive_data_agreed: bool
    created_at: datetime
    updated_at: datetime
