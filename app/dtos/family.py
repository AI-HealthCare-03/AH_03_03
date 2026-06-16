from datetime import datetime, time

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models.family import (
    FamilyInviteStatus,
    FamilyMemberRole,
    FamilyMemberStatus,
    FamilyRelationType,
    FamilyStatus,
)


class FamilyGroupCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class FamilyGroupUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)


class FamilyGroupResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    owner_user_id: int
    status: FamilyStatus
    created_at: datetime
    updated_at: datetime


class FamilyMemberCreateUnregisteredRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)
    relation_type: FamilyRelationType
    phone_number: str | None = Field(default=None, max_length=30)
    email: EmailStr | None = None


class FamilyMemberResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    family_id: int
    user_id: int | None
    display_name: str
    phone_number: str | None
    email: str | None
    relation_type: FamilyRelationType
    member_role: FamilyMemberRole
    status: FamilyMemberStatus
    is_registered: bool
    created_at: datetime
    updated_at: datetime


class FamilyGroupDetailResponse(FamilyGroupResponse):
    members: list[FamilyMemberResponse]


class FamilyInviteCreateRequest(BaseModel):
    invitee_email: EmailStr | None = None
    invitee_phone: str | None = Field(default=None, max_length=30)
    invitee_user_id: int | None = None
    relation_type: FamilyRelationType
    member_role: FamilyMemberRole = FamilyMemberRole.MEMBER


class FamilyInviteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    family_id: int
    inviter_user_id: int
    invitee_user_id: int | None
    invitee_email: str | None
    invitee_phone: str | None
    relation_type: FamilyRelationType
    member_role: FamilyMemberRole
    status: FamilyInviteStatus
    expires_at: datetime
    used_at: datetime | None
    created_at: datetime
    invite_code: str | None = None


class FamilySentInviteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    family_id: int
    inviter_user_id: int
    invitee_user_id: int | None
    invitee_email: str | None
    invitee_phone: str | None
    relation_type: FamilyRelationType
    member_role: FamilyMemberRole
    status: FamilyInviteStatus
    expires_at: datetime
    used_at: datetime | None
    created_at: datetime


class FamilyInviteAcceptCodeRequest(BaseModel):
    code: str = Field(min_length=8, max_length=8)

    @field_validator("code")
    @classmethod
    def strip_code(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped.isdigit() or len(stripped) != 8:
            raise ValueError("초대 코드는 8자리 숫자입니다.")
        return stripped


class FamilyInvitePreviewCodeRequest(BaseModel):
    invite_code: str = Field(min_length=8, max_length=8)

    @field_validator("invite_code")
    @classmethod
    def strip_invite_code(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped.isdigit() or len(stripped) != 8:
            raise ValueError("초대 코드는 8자리 숫자입니다.")
        return stripped


class FamilyInvitePreviewResponse(BaseModel):
    invite_id: int
    family_id: int
    family_name: str
    inviter_display_name: str
    invitee_email: str | None
    status: FamilyInviteStatus
    expires_at: datetime


class FamilyShareSettingUpdateRequest(BaseModel):
    share_health_records: bool | None = None
    share_analysis_results: bool | None = None
    share_diet_records: bool | None = None
    share_medications: bool | None = None
    share_challenges: bool | None = None
    share_exam_reports: bool | None = None
    share_challenge_status: bool | None = None
    share_medication_status: bool | None = None
    share_diet_status: bool | None = None
    share_health_report_summary: bool | None = None
    share_raw_health_values: bool | None = None
    share_ocr_original: bool | None = None
    receive_analysis_alerts: bool | None = None
    receive_abnormal_value_alerts: bool | None = None
    receive_medication_alerts: bool | None = None


class FamilyShareSettingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    family_id: int
    owner_user_id: int
    viewer_user_id: int
    share_health_records: bool
    share_analysis_results: bool
    share_diet_records: bool
    share_medications: bool
    share_challenges: bool
    share_exam_reports: bool
    share_challenge_status: bool
    share_medication_status: bool
    share_diet_status: bool
    share_health_report_summary: bool
    share_raw_health_values: bool
    share_ocr_original: bool
    receive_analysis_alerts: bool
    receive_abnormal_value_alerts: bool
    receive_medication_alerts: bool
    created_at: datetime
    updated_at: datetime


class FamilyActionShareSettingUpdateRequest(BaseModel):
    share_challenge_status: bool | None = None
    share_medication_status: bool | None = None
    share_diet_status: bool | None = None
    share_health_report_summary: bool | None = None
    share_raw_health_values: bool | None = None
    share_ocr_original: bool | None = None


class FamilyNotificationSettingUpdateRequest(BaseModel):
    notify_challenge_missed: bool | None = None
    notify_challenge_completed: bool | None = None
    notify_medication_missed: bool | None = None
    notify_diet_missed: bool | None = None
    notify_report_updated: bool | None = None
    channel_in_app: bool | None = None
    channel_push: bool | None = None
    quiet_hours_start: time | None = None
    quiet_hours_end: time | None = None


class FamilyNotificationSettingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_user_id: int
    family_user_id: int
    notify_challenge_missed: bool
    notify_challenge_completed: bool
    notify_medication_missed: bool
    notify_diet_missed: bool
    notify_report_updated: bool
    channel_in_app: bool
    channel_push: bool
    quiet_hours_start: time | None
    quiet_hours_end: time | None
    created_at: datetime
    updated_at: datetime
