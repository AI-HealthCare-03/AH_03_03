from datetime import datetime, time

from pydantic import BaseModel, ConfigDict, field_validator


def _normalize_optional_time(value: object) -> object:
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


class MedicationCreateRequest(BaseModel):
    name: str
    medication_type: str
    dosage: str | None = None
    frequency: str | None = None
    reminder_time: time | None = None
    is_active: bool = True
    memo: str | None = None

    @field_validator("reminder_time", mode="before")
    @classmethod
    def normalize_reminder_time(cls, value: object) -> object:
        return _normalize_optional_time(value)


class MedicationUpdateRequest(BaseModel):
    name: str | None = None
    medication_type: str | None = None
    dosage: str | None = None
    frequency: str | None = None
    reminder_time: time | None = None
    is_active: bool | None = None
    memo: str | None = None

    @field_validator("reminder_time", mode="before")
    @classmethod
    def normalize_reminder_time(cls, value: object) -> object:
        return _normalize_optional_time(value)


class MedicationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    name: str
    medication_type: str
    dosage: str | None
    frequency: str | None
    reminder_time: time | None
    is_active: bool
    memo: str | None
    created_at: datetime
    updated_at: datetime


class MedicationListResponse(BaseModel):
    items: list[MedicationResponse]
    total: int


class MedicationRecordCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheduled_at: datetime | None = None
    taken_at: datetime | None = None
    is_taken: bool = False
    status: str = "PENDING"
    memo: str | None = None


class MedicationRecordUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheduled_at: datetime | None = None
    taken_at: datetime | None = None
    is_taken: bool | None = None
    status: str | None = None
    memo: str | None = None


class MedicationRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    medication_id: int
    user_id: int
    scheduled_at: datetime | None
    taken_at: datetime | None
    is_taken: bool
    status: str
    memo: str | None
    created_at: datetime
    updated_at: datetime


class MedicationDetailResponse(MedicationResponse):
    records: list[MedicationRecordResponse]
