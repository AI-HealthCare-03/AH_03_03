"""
ai_worker/vision/router.py

GPT Vision 분석 FastAPI 라우터.
MVP 기준 3개 엔드포인트 제공 (식단 / 처방전 / 건강검진표).
"""

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from .client import AnalysisType, VisionClient
from .schemas import (
    ERROR_MAP,
    STATUS_MESSAGE,
    AnalysisStatus,
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
    image_bytes = await validate_image(file)

    try:
        return await client.analyze(
            analysis_type=analysis_type,
            image_bytes=image_bytes,
            media_type=file.content_type or "image/jpeg",
        )
    except ValueError as e:
        err = ERROR_MAP["parse_error"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(
                error_code="parse_error",
                message=err["message"],
            ).model_dump(),
        ) from e
    except Exception as e:
        logger.exception("GPT Vision 호출 실패 | type=%s", analysis_type)
        err = ERROR_MAP["vision_api_error"]
        raise HTTPException(
            status_code=err["status_code"],
            detail=ErrorResponse(
                error_code="vision_api_error",
                message=err["message"],
            ).model_dump(),
        ) from e


# ── BMI 계산 헬퍼 ─────────────────────────────────────────────────────────────


def calculate_bmi(height_cm, weight_kg) -> float | None:
    """키(cm)와 몸무게(kg)로 BMI를 계산합니다."""
    try:
        h = float(height_cm)
        w = float(weight_kg)
        if h > 0 and w > 0:
            return round(w / ((h / 100) ** 2), 1)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


# ── 엔드포인트 ────────────────────────────────────────────────────────────────


@router.post(
    "/diet",
    response_model=DietAnalysisResponse,
    summary="식단 이미지 분석",
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
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
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
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
건강검진 결과지 사진을 업로드하면 만성질환 및 기초 건강 수치를 추출합니다.

- 추출 항목: 혈압, 혈당, 혈색소(Hb), 콜레스테롤, 중성지방, HDL, LDL, 키, 체중, 허리둘레
- BMI: 검진표에 수치가 있으면 그대로 사용, 없으면 키·몸무게로 자체 계산 (bmi_calculated=True)
    """,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_checkup(
    file: UploadFile = File(..., description="건강검진 결과지 이미지"),
    client: VisionClient = Depends(get_vision_client),
) -> CheckupAnalysisResponse:
    raw = await call_vision(AnalysisType.CHECKUP, file, client)

    extracted = raw.get("extracted_data", {})
    requires_manual: list[str] = raw.get("requires_manual_input", [])
    bmi_calculated = False

    # ── BMI 처리 ──────────────────────────────────────────────────────────────
    # 검진표에 BMI 수치가 있으면 그대로, 없으면 키·몸무게로 자체 계산
    bmi_raw = extracted.get("bmi")
    height   = extracted.get("height_cm")
    weight   = extracted.get("weight_kg")

    if bmi_raw is not None:
        # 검진표 원본 수치 사용
        extracted["bmi"] = bmi_raw
        bmi_calculated = False
    elif height is not None and weight is not None:
        # 구형 검진표 등 BMI 수치 없는 경우 → 자체 계산
        calculated = calculate_bmi(height, weight)
        if calculated is not None:
            extracted["bmi"] = calculated
            bmi_calculated = True
            logger.info("BMI 자체 계산 | height=%.1f weight=%.1f bmi=%.1f", float(height), float(weight), calculated)
        else:
            extracted["bmi"] = None
            if "bmi" not in requires_manual:
                requires_manual.append("bmi")
    else:
        extracted["bmi"] = None
        if "bmi" not in requires_manual:
            requires_manual.append("bmi")

    # ── 상태 재평가 ───────────────────────────────────────────────────────────
    gpt_status = raw.get("analysis_status", "success")
    if requires_manual and gpt_status == AnalysisStatus.SUCCESS:
        gpt_status = AnalysisStatus.PARTIAL

    return CheckupAnalysisResponse(
        analysis_status=gpt_status,
        message=STATUS_MESSAGE.get(gpt_status, STATUS_MESSAGE["failed"]),
        extracted_data=CheckupExtractedData(
            **extracted,
            bmi_calculated=bmi_calculated,
        ),
        requires_manual_input=requires_manual,
        raw_result=raw,
    )
