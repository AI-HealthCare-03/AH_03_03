"""
ai_worker/vision/schemas.py

MVP 기준 GPT Vision 응답 스키마.
추후 API 명세서 확정되면 에러코드 및 필드 보완 예정.
추후 식약처 영양성분 DB 매칭 시 NutritionInfo 필드 기준으로 연동 예정.
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── 분석 상태 ─────────────────────────────────────────────────────────────────

class AnalysisStatus(str, Enum):
    SUCCESS = "success"                # 분석 성공
    LOW_CONFIDENCE = "low_confidence"  # 신뢰도 낮음 → 사용자 확인 필요
    PARTIAL = "partial"                # 일부만 인식
    FAILED = "failed"                  # 분석 실패


# 사용자 노출 한글 메시지
STATUS_MESSAGE: dict[str, str] = {
    "success":        "분석이 완료되었습니다.",
    "low_confidence": "인식 정확도가 낮습니다. 결과를 확인하고 수정해주세요.",
    "partial":        "일부 항목만 인식되었습니다. 누락된 항목을 직접 입력해주세요.",
    "failed":         "분석에 실패했습니다. 다시 시도해주세요.",
}


# ── 에러 (REQ-CV-005 기준) ────────────────────────────────────────────────────
# TODO: API 명세서 CV 엔드포인트 확정 후 상태코드 및 에러코드 보완

class ErrorResponse(BaseModel):
    error_code: str
    message: str  # 사용자에게 보여줄 한글 메시지


ERROR_MAP: dict[str, dict] = {
    # 이미지 품질 문제
    "image_too_large":  {"status_code": 413, "message": "이미지 크기가 너무 큽니다. 10MB 이하의 사진으로 다시 시도해주세요."},
    "image_too_small":  {"status_code": 422, "message": "이미지가 손상되었거나 너무 작습니다. 다시 촬영해주세요."},
    "unsupported_type": {"status_code": 415, "message": "지원하지 않는 이미지 형식입니다. JPG, PNG, WEBP로 다시 업로드해주세요."},
    "image_blurry":     {"status_code": 422, "message": "사진이 흐립니다. 선명하게 다시 촬영해주세요."},
    "image_too_dark":   {"status_code": 422, "message": "사진이 너무 어둡습니다. 밝은 곳에서 다시 촬영해주세요."},
    "not_food_image":   {"status_code": 422, "message": "음식 이미지가 아닌 것 같습니다. 올바른 사진으로 다시 시도해주세요."},
    # 인식 실패
    "no_food_detected": {"status_code": 200, "message": "음식을 감지하지 못했습니다. 다시 촬영하거나 직접 입력해주세요."},
    "no_text_detected": {"status_code": 422, "message": "글자를 인식하지 못했습니다. 글자가 잘 보이도록 다시 촬영해주세요."},
    # Vision API 오류
    "vision_api_error":   {"status_code": 500, "message": "AI 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."},
    "vision_api_timeout": {"status_code": 504, "message": "분석 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."},
    "parse_error":        {"status_code": 500, "message": "분석 결과를 처리하는 중 오류가 발생했습니다. 다시 시도해주세요."},
    # 서버 오류
    "server_error":     {"status_code": 500, "message": "서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."},
    # Rate Limit (NFR-SEC-020)
    "rate_limit":       {"status_code": 429, "message": "요청이 너무 많습니다. 잠시 후 다시 시도해주세요."},
}


# ── 공통 응답 베이스 ──────────────────────────────────────────────────────────

class BaseAnalysisResponse(BaseModel):
    analysis_type: str
    analysis_status: AnalysisStatus
    message: str = Field(description="사용자에게 보여줄 한글 상태 메시지")
    raw_result: dict[str, Any] = Field(description="GPT Vision 원본 응답 (개발용)")


# ── 식단 ──────────────────────────────────────────────────────────────────────

class NutritionInfo(BaseModel):
    """
    GPT Vision 추정 영양성분.
    추후 식약처 영양성분 DB 매칭 시 이 필드 기준으로 연동 예정.

    4대 만성질환 관련 핵심 영양소:
        고혈압       → 나트륨(↓), 식이섬유(↑)
        당뇨         → 당류(↓), 탄수화물(↓), 식이섬유(↑)
        이상지질혈증  → 포화지방(↓), 식이섬유(↑)
        비만         → 칼로리(↓), 지방(↓)
    """
    칼로리: float | None = Field(None, description="kcal")
    탄수화물: float | None = Field(None, description="g — 당뇨 관련")
    당류: float | None = Field(None, description="g — 당뇨 관련")
    단백질: float | None = Field(None, description="g")
    지방: float | None = Field(None, description="g — 비만 관련")
    포화지방: float | None = Field(None, description="g — 이상지질혈증 관련")
    식이섬유: float | None = Field(None, description="g — 고혈압/당뇨/이상지질혈증 관련")
    나트륨: float | None = Field(None, description="mg — 고혈압 관련")
    영양성분_신뢰도: float = Field(
        default=0.0,
        description="GPT 추정 신뢰도. 이미지 기반이므로 보통 0.5 이하"
    )
    추정값_여부: bool = Field(
        default=True,
        description="항상 True. 정확한 수치는 식약처 DB 매칭 후 확정"
    )


class FoodItem(BaseModel):
    name: str = Field(description="음식명")
    nutrient_category: str = Field(
        description="탄수화물|단백질|지방|식이섬유|비타민|미네랄|항산화|건강식"
    )
    cooking_method: str | None = Field(
        None,
        description="조리법 (예: 튀김, 구이, 찜, 볶음, 생것)"
    )
    estimated_amount: str | None = Field(
        None,
        description="추정 용량 (예: 200g, 1공기). 불확실하면 null → 사용자 직접 입력"
    )
    amount_requires_input: bool = Field(
        default=False,
        description="용량 추정 불가 시 True → 프론트에서 입력창 표시"
    )
    nutrition: NutritionInfo | None = Field(
        None,
        description="추정 영양성분. 추정 불가 시 null"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="음식 인식 신뢰도")
    search_keyword: str = Field(description="식약처/영양DB 검색 키워드")


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
    """4대 만성질환(고혈압·당뇨·이상지질혈증·비만) 관련 수치."""
    systolic_bp: float | None = None        # 수축기 혈압 (mmHg)
    diastolic_bp: float | None = None       # 이완기 혈압 (mmHg)
    fasting_glucose: float | None = None    # 공복혈당 (mg/dL)
    hba1c: float | None = None              # 당화혈색소 (%)
    total_cholesterol: float | None = None  # 총콜레스테롤 (mg/dL)
    triglyceride: float | None = None       # 중성지방 (mg/dL)
    hdl: float | None = None                # HDL (mg/dL)
    ldl: float | None = None                # LDL (mg/dL)
    height_cm: float | None = None          # 키 (cm)
    weight_kg: float | None = None          # 몸무게 (kg)
    bmi: float | None = None                # BMI
    waist_cm: float | None = None           # 허리둘레 (cm)


class CheckupAnalysisResponse(BaseAnalysisResponse):
    analysis_type: str = "checkup"
    extracted_data: CheckupExtractedData = Field(default_factory=CheckupExtractedData)
    confidence_per_field: dict[str, float] = Field(default_factory=dict)
    unreadable_fields: list[str] = Field(default_factory=list)