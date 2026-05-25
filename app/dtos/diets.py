from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DietRecordCreateRequest(BaseModel):
    meal_type: str | None = None
    meal_time: datetime | None = None
    description: str | None = None
    image_path: str | None = None
    detected_foods: dict[str, Any] | list[dict[str, Any]] | None = None
    nutrition_summary: dict[str, Any] | None = None
    diet_score: float | None = None
    diet_feedback: str | None = None
    analysis_method: str | None = None
    is_user_corrected: bool = False
    memo: str | None = None


class DietRecordUpdateRequest(BaseModel):
    meal_type: str | None = None
    meal_time: datetime | None = None
    description: str | None = None
    image_path: str | None = None
    detected_foods: dict[str, Any] | list[dict[str, Any]] | None = None
    nutrition_summary: dict[str, Any] | None = None
    diet_score: float | None = None
    diet_feedback: str | None = None
    analysis_method: str | None = None
    is_user_corrected: bool | None = None
    memo: str | None = None


class DietRecordResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    meal_type: str | None
    meal_time: datetime | None
    description: str | None
    image_path: str | None
    detected_foods: dict[str, Any] | list[dict[str, Any]] | None
    nutrition_summary: dict[str, Any] | None
    diet_score: float | None
    diet_feedback: str | None
    analysis_method: str | None
    is_user_corrected: bool
    memo: str | None
    created_at: datetime
    updated_at: datetime


class DietRecordListResponse(BaseModel):
    items: list[DietRecordResponse]
    total: int


class DietPhotoResultCreateRequest(BaseModel):
    detected_foods: dict[str, Any] | list[dict[str, Any]] | None = None
    confidence_payload: dict[str, Any] | None = None
    raw_output: dict[str, Any] | None = None
    is_dummy: bool = False


class DietPhotoResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    diet_record_id: int
    detected_foods: dict[str, Any] | list[dict[str, Any]] | None
    confidence_payload: dict[str, Any] | None
    raw_output: dict[str, Any] | None
    created_at: datetime


class DietAnalyzePhotoResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    diet_record_id: int
    detected_foods: dict[str, Any] | list[dict[str, Any]] | None
    confidence_payload: dict[str, Any] | None
    raw_output: dict[str, Any] | None
    created_at: datetime


class DietRecordDetailResponse(DietRecordResponse):
    photo_results: list[DietPhotoResultResponse]


class DietAnalyzeRequest(BaseModel):
    meal_type: str | None = None
    meal_time: datetime | None = None
    description: str | None = None
    image_path: str | None = None
    memo: str | None = None


class DietAnalyzeResponse(BaseModel):
    message: str
    diet_record: DietRecordResponse
    photo_result: DietAnalyzePhotoResultResponse
    detected_foods: list[dict[str, Any]]
    nutrition_summary: dict[str, Any]
    diet_score: float
    diet_feedback: str
    disease_scores: dict[str, float | None] | None = None
    food_score_details: list[dict[str, Any]] = []
    scoring_source: str | None = None
    explanation: dict[str, Any] | None = None
    warnings: list[str] = []
    recommended_actions: list[str] = []
