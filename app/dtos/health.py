from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from app.dtos.base import BaseSerializerModel


class HealthRecordCreateRequest(BaseModel):
    height_cm: Decimal | None = None
    weight_kg: Decimal | None = None
    bmi: Decimal | None = None
    waist_cm: Decimal | None = None
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    fasting_glucose: int | None = None
    hba1c: Decimal | None = None
    total_cholesterol: int | None = None
    ldl_cholesterol: int | None = None
    hdl_cholesterol: int | None = None
    triglyceride: int | None = None
    has_diabetes: bool | None = None
    has_obesity: bool | None = None
    has_dyslipidemia: bool | None = None
    has_hypertension: bool | None = None
    occupation_code: str | None = None
    family_htn: str | None = None
    family_dm: str | None = None
    family_dyslipidemia: str | None = None
    smoking_status: str | None = None
    drinking_frequency: str | None = None
    drinking_amount: str | None = None
    walking_days_per_week: int | None = Field(default=None, ge=0, le=7)
    strength_days_per_week: int | None = Field(default=None, ge=0, le=7)
    sleep_hours: Decimal | None = None
    measured_at: datetime


class HealthRecordUpdateRequest(BaseModel):
    height_cm: Decimal | None = None
    weight_kg: Decimal | None = None
    bmi: Decimal | None = None
    waist_cm: Decimal | None = None
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    fasting_glucose: int | None = None
    hba1c: Decimal | None = None
    total_cholesterol: int | None = None
    ldl_cholesterol: int | None = None
    hdl_cholesterol: int | None = None
    triglyceride: int | None = None
    has_diabetes: bool | None = None
    has_obesity: bool | None = None
    has_dyslipidemia: bool | None = None
    has_hypertension: bool | None = None
    occupation_code: str | None = None
    family_htn: str | None = None
    family_dm: str | None = None
    family_dyslipidemia: str | None = None
    smoking_status: str | None = None
    drinking_frequency: str | None = None
    drinking_amount: str | None = None
    walking_days_per_week: int | None = Field(default=None, ge=0, le=7)
    strength_days_per_week: int | None = Field(default=None, ge=0, le=7)
    sleep_hours: Decimal | None = None
    measured_at: datetime | None = None


class HealthRecordResponse(BaseSerializerModel):
    id: int
    user_id: int
    height_cm: Decimal | None
    weight_kg: Decimal | None
    bmi: Decimal | None
    waist_cm: Decimal | None
    systolic_bp: int | None
    diastolic_bp: int | None
    fasting_glucose: int | None
    hba1c: Decimal | None
    total_cholesterol: int | None
    ldl_cholesterol: int | None
    hdl_cholesterol: int | None
    triglyceride: int | None
    has_diabetes: bool | None
    has_obesity: bool | None
    has_dyslipidemia: bool | None
    has_hypertension: bool | None
    occupation_code: str | None
    family_htn: str | None
    family_dm: str | None
    family_dyslipidemia: str | None
    smoking_status: str | None
    drinking_frequency: str | None
    drinking_amount: str | None
    walking_days_per_week: int | None
    strength_days_per_week: int | None
    is_smoker: bool | None
    drinks_alcohol: bool | None
    exercise_days_per_week: int | None
    sleep_hours: Decimal | None
    measured_at: datetime
    created_at: datetime
    updated_at: datetime


class HealthRecordListResponse(BaseModel):
    items: list[HealthRecordResponse]
    total: int


class HealthAnalysisReadinessResponse(BaseModel):
    is_ready: bool
    basic_ready: bool | None = None
    precision_ready: bool | None = None
    latest_health_record_id: int | None = None
    missing_fields: list[str]
    missing_basic_fields: list[str] = []
    missing_precision_fields: list[str] = []
    message: str
