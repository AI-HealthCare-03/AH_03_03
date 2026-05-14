"""
ai_worker/vision/router.py

GPT Vision 분석 FastAPI 라우터.
MVP 기준 3개 엔드포인트 제공 (식단 / 처방전 / 건강검진표).

팀 통합 시 app/main.py에 아래 추가:
    from ai_worker.vision.router import router as vision_router
    app.include_router(vision_router)
"""

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from .client import AnalysisType, VisionClient
from .schemas import (
    ERROR_MAP,
    STATUS_MESSAGE,
    CheckupAnalysisResponse,
    CheckupExtractedData,
    DietAnalysisResponse,
    ErrorResponse,
    FoodItem,
    MedicationItem,
    PrescriptionAnalysisResponse,
)
from .settings import VisionSettings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cv", tags=["Vision - GPT 이미지 분석"])


# ── 의존성 ────────────────────────────────────────────────────────────────────

@lru_cache
def get_settings() -> VisionSettings:
    return VisionSettings()


def get_vision_client(
    settings: VisionSettings = Depends(get_settings),
) -> VisionClient:
    return VisionClient(api_key=settings.openai_api_key, model=settings.openai_model)


# ── 이미지 유효성 검사 ────────────────────────────────────────────────────────

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


async def validate_image(file: UploadFile) -> bytes:
    """업로드 이미지 형식 및 크기 검증."""

    if file.content_type not in ALLOWED_TYPES:
        err = ERROR_MAP["unsupported_type"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(
                error_code="unsupported_type",
                message=err["message"],
            ).model_dump(),
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_SIZE_BYTES:
        err = ERROR_MAP["image_too_large"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(
                error_code="image_too_large",
                message=err["message"],
            ).model_dump(),
        )

    if len(image_bytes) < 1024:
        err = ERROR_MAP["image_too_small"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(
                error_code="image_too_small",
                message=err["message"],
            ).model_dump(),
        )

    return image_bytes


# ── 공통 Vision 호출 ──────────────────────────────────────────────────────────

async def call_vision(
    analysis_type: str,
    file: UploadFile,
    client: VisionClient,
) -> dict:
    """이미지 검증 후 GPT Vision 호출. 에러 시 HTTPException 발생."""

    image_bytes = await validate_image(file)

    try:
        return await client.analyze(
            analysis_type=analysis_type,
            image_bytes=image_bytes,
            media_type=file.content_type or "image/jpeg",
        )
    except ValueError:
        err = ERROR_MAP["parse_error"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(
                error_code="parse_error",
                message=err["message"],
            ).model_dump(),
        )
    except Exception:
        logger.exception("GPT Vision 호출 실패 | type=%s", analysis_type)
        err = ERROR_MAP["vision_api_error"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(
                error_code="vision_api_error",
                message=err["message"],
            ).model_dump(),
        )


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@router.post(
    "/diet",
    response_model=DietAnalysisResponse,
    summary="식단 이미지 분석",
    description="""
식단 사진을 업로드하면 음식명, 카테고리, 신뢰도를 분석합니다.

- 결과는 사용자 확인 후 DIET 도메인에서 저장합니다.
- 영양성분 계산은 DIET 도메인에서 처리합니다.
    """,
    responses={
        413: {"model": ErrorResponse, "description": "이미지 크기 초과"},
        415: {"model": ErrorResponse, "description": "지원하지 않는 이미지 형식"},
        422: {"model": ErrorResponse, "description": "이미지 품질 문제"},
        500: {"model": ErrorResponse, "description": "GPT Vision 오류"},
    },
)
async def analyze_diet(
    file: UploadFile = File(..., description="식단 이미지 (JPG/PNG/WEBP, 최대 10MB)"),
    client: VisionClient = Depends(get_vision_client),
) -> DietAnalysisResponse:

    raw = await call_vision(AnalysisType.DIET, file, client)
    status = raw.get("analysis_status", "failed")

    return DietAnalysisResponse(
        analysis_status=status,
        message=STATUS_MESSAGE.get(status, STATUS_MESSAGE["failed"]),
        foods=[FoodItem(**f) for f in raw.get("foods", [])],
        requires_user_confirmation=True,
        raw_result=raw,
    )


@router.post(
    "/prescription",
    response_model=PrescriptionAnalysisResponse,
    summary="처방전 / 약봉투 분석",
    description="""
약 봉투 또는 처방전 사진을 업로드하면 약품명, 용량, 수량을 추출합니다.

- 복용법(횟수, 식전/후/간)은 추출하지 않습니다. 사용자가 직접 선택합니다.
- 인식 불확실 항목은 requires_manual_input 목록으로 반환됩니다.
    """,
    responses={
        413: {"model": ErrorResponse, "description": "이미지 크기 초과"},
        415: {"model": ErrorResponse, "description": "지원하지 않는 이미지 형식"},
        422: {"model": ErrorResponse, "description": "이미지 품질 문제"},
        500: {"model": ErrorResponse, "description": "GPT Vision 오류"},
    },
)
async def analyze_prescription(
    file: UploadFile = File(..., description="처방전 또는 약봉투 이미지"),
    client: VisionClient = Depends(get_vision_client),
) -> PrescriptionAnalysisResponse:

    raw = await call_vision(AnalysisType.PRESCRIPTION, file, client)
    status = raw.get("analysis_status", "failed")

    return PrescriptionAnalysisResponse(
        analysis_status=status,
        message=STATUS_MESSAGE.get(status, STATUS_MESSAGE["failed"]),
        medications=[MedicationItem(**m) for m in raw.get("medications", [])],
        requires_manual_input=raw.get("requires_manual_input", []),
        raw_result=raw,
    )


@router.post(
    "/checkup",
    response_model=CheckupAnalysisResponse,
    summary="건강검진표 수치 추출",
    description="""
건강검진 결과지 사진을 업로드하면 4대 만성질환 관련 수치를 추출합니다.

- 추출 항목: 혈압, 혈당, 당화혈색소, 콜레스테롤, 중성지방, HDL, LDL, 체중, BMI, 허리둘레
- 정상/비정상 판정은 제공하지 않습니다.
- 추출된 수치는 사용자 확인 후 저장하세요.
    """,
    responses={
        413: {"model": ErrorResponse, "description": "이미지 크기 초과"},
        415: {"model": ErrorResponse, "description": "지원하지 않는 이미지 형식"},
        422: {"model": ErrorResponse, "description": "이미지 품질 문제"},
        500: {"model": ErrorResponse, "description": "GPT Vision 오류"},
    },
)
async def analyze_checkup(
    file: UploadFile = File(..., description="건강검진 결과지 이미지"),
    client: VisionClient = Depends(get_vision_client),
) -> CheckupAnalysisResponse:

    raw = await call_vision(AnalysisType.CHECKUP, file, client)
    status = raw.get("analysis_status", "failed")

    return CheckupAnalysisResponse(
        analysis_status=status,
        message=STATUS_MESSAGE.get(status, STATUS_MESSAGE["failed"]),
        extracted_data=CheckupExtractedData(**raw.get("extracted_data", {})),
        confidence_per_field=raw.get("confidence_per_field", {}),
        unreadable_fields=raw.get("unreadable_fields", []),
        raw_result=raw,
    )