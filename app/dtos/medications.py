from datetime import datetime, time

from pydantic import BaseModel, ConfigDict, Field


class MedicationCreateRequest(BaseModel):
    name: str
    medication_type: str
    dosage: str | None = None
    frequency: str | None = None
    reminder_time: time | None = None
    is_active: bool = True
    memo: str | None = None


class MedicationUpdateRequest(BaseModel):
    name: str | None = None
    medication_type: str | None = None
    dosage: str | None = None
    frequency: str | None = None
    reminder_time: time | None = None
    is_active: bool | None = None
    memo: str | None = None


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


class MedicationOCRRequest(BaseModel):
    source_type: str | None = "PRESCRIPTION"
    image_filename: str | None = None
    memo: str | None = None
    raw_text: str | None = None


class MedicationOCRItem(BaseModel):
    temp_id: str | None = None
    name: str
    dosage: str | None = None
    frequency: str | None = None
    time_slots: list[str] = Field(default_factory=list)
    duration_days: int | None = None
    memo: str | None = None
    confidence: float | None = None


class MedicationOCRResponse(BaseModel):
    is_dummy: bool = False
    source_type: str
    ocr_confidence: float
    items: list[MedicationOCRItem]
    message: str
    source: str = "rule_based_medication_ocr"
    raw_text: str | None = None
    parser_warnings: list[str] = Field(default_factory=list)


class MedicationOCRConfirmItem(BaseModel):
    name: str
    dosage: str | None = None
    frequency: str | None = None
    time_slots: list[str] = Field(default_factory=list)
    duration_days: int | None = None
    memo: str | None = None


class MedicationOCRConfirmRequest(BaseModel):
    items: list[MedicationOCRConfirmItem]


class MedicationOCRConfirmResponse(BaseModel):
    created_count: int
    created_medication_ids: list[int]
    skipped_count: int = 0
    message: str


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
