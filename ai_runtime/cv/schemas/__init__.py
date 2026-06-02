"""
ai_runtime/cv/schemas/__init__.py

MVP 기준 GPT Vision 응답 스키마.
추후 API 명세서 확정되면 에러코드 및 필드 보완 예정.
추후 식약처 영양성분 DB 매칭 시 NutritionInfo 필드 기준으로 연동 예정.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── 분석 상태 ─────────────────────────────────────────────────────────────────


class AnalysisStatus(StrEnum):
    SUCCESS = "success"
    LOW_CONFIDENCE = "low_confidence"
    PARTIAL = "partial"
    FAILED = "failed"


STATUS_MESSAGE: dict[str, str] = {
    "success": "분석이 완료되었습니다.",
    "low_confidence": "인식 정확도가 낮습니다. 결과를 확인하고 수정해주세요.",
    "partial": "일부 항목만 인식되었습니다. 누락된 항목을 직접 입력해주세요.",
    "failed": "분석에 실패했습니다. 다시 시도해주세요.",
}


# ── 에러 ──────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error_code: str
    message: str


ERROR_MAP: dict[str, dict] = {
    "image_too_large": {
        "status_code": 413,
        "message": "이미지 크기가 너무 큽니다. 10MB 이하의 사진으로 다시 시도해주세요.",
    },
    "image_too_small": {
        "status_code": 422,
        "message": "이미지가 손상되었거나 너무 작습니다. 다시 촬영해주세요.",
    },
    "unsupported_type": {
        "status_code": 415,
        "message": "지원하지 않는 이미지 형식입니다. JPG, PNG, WEBP, HEIC로 다시 업로드해주세요.",
    },
    "image_blurry": {
        "status_code": 422,
        "message": "사진이 흐립니다. 선명하게 다시 촬영해주세요.",
    },
    "image_too_dark": {
        "status_code": 422,
        "message": "사진이 너무 어둡습니다. 밝은 곳에서 다시 촬영해주세요.",
    },
    "not_food_image": {
        "status_code": 422,
        "message": "음식 이미지가 아닌 것 같습니다. 올바른 사진으로 다시 시도해주세요.",
    },
    "no_food_detected": {
        "status_code": 200,
        "message": "음식을 감지하지 못했습니다. 다시 촬영하거나 직접 입력해주세요.",
    },
    "no_text_detected": {
        "status_code": 422,
        "message": "글자를 인식하지 못했습니다. 글자가 잘 보이도록 다시 촬영해주세요.",
    },
    "vision_api_error": {
        "status_code": 500,
        "message": "AI 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
    },
    "vision_api_timeout": {
        "status_code": 504,
        "message": "분석 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
    },
    "parse_error": {
        "status_code": 500,
        "message": "분석 결과를 처리하는 중 오류가 발생했습니다. 다시 시도해주세요.",
    },
    "server_error": {
        "status_code": 500,
        "message": "서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
    },
    "rate_limit": {
        "status_code": 429,
        "message": "요청이 너무 많습니다. 잠시 후 다시 시도해주세요.",
    },
}


# ── 공통 응답 베이스 ──────────────────────────────────────────────────────────


class BaseAnalysisResponse(BaseModel):
    analysis_type: str
    analysis_status: AnalysisStatus
    message: str = Field(description="사용자에게 보여줄 한글 상태 메시지")
    raw_result: dict[str, Any] = Field(description="GPT Vision 원본 응답 (개발용)")


# ── 식단 ──────────────────────────────────────────────────────────────────────


class NutritionInfo(BaseModel):
    칼로리: float | None = Field(None, description="kcal")
    탄수화물: float | None = Field(None, description="g")
    당류: float | None = Field(None, description="g")
    단백질: float | None = Field(None, description="g")
    지방: float | None = Field(None, description="g")
    포화지방: float | None = Field(None, description="g")
    식이섬유: float | None = Field(None, description="g")
    나트륨: float | None = Field(None, description="mg")
    영양성분_신뢰도: float = Field(default=0.0)
    추정값_여부: bool = Field(default=True)


class FoodItem(BaseModel):
    name: str
    nutrient_category: str
    cooking_method: str | None = None
    estimated_amount: str | None = None
    amount_requires_input: bool = False
    nutrition: NutritionInfo | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    search_keyword: str


class DietAnalysisResponse(BaseAnalysisResponse):
    analysis_type: str = "diet"
    foods: list[FoodItem] = Field(default_factory=list)
    requires_user_confirmation: bool = True


# ── 처방전 ────────────────────────────────────────────────────────────────────


class MedicationItem(BaseModel):
    drug_name: str
    dosage: str | None = None
    quantity: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str | None = None


class PrescriptionAnalysisResponse(BaseAnalysisResponse):
    analysis_type: str = "prescription"
    medications: list[MedicationItem] = Field(default_factory=list)
    requires_manual_input: list[str] = Field(default_factory=list)


# ── 건강검진표 ────────────────────────────────────────────────────────────────


class CheckupExtractedData(BaseModel):
    systolic_bp: str | float | None = None
    diastolic_bp: str | float | None = None
    fasting_glucose: str | float | None = None
    hb: str | float | None = Field(None, description="혈색소(Hemoglobin) g/dL")
    total_cholesterol: str | float | None = None
    triglyceride: str | float | None = None
    hdl: str | float | None = None
    ldl: str | float | None = None
    height_cm: str | float | None = None
    weight_kg: str | float | None = None
    bmi: str | float | None = Field(None, description="키/몸무게 기반 자체 계산값. 검진표 수치 없어도 제공")
    bmi_calculated: bool = Field(False, description="True = 키·몸무게로 자체 계산 / False = 검진표 원본 수치")
    waist_cm: str | float | None = None


class CheckupAnalysisResponse(BaseAnalysisResponse):
    analysis_type: str = "checkup"
    analysis_status: AnalysisStatus = AnalysisStatus.SUCCESS
    extracted_data: CheckupExtractedData = Field(default_factory=CheckupExtractedData)
    requires_manual_input: list[str] = Field(default_factory=list)
