"""Draft FastAPI router for X2 health stage classification.

This router is intentionally not included in app/main.py yet. It is a future
integration surface for OCR-confirmed or health-record mapped values.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_runtime.ml.X2.health_stage_classifier import (
    SOURCE_VARIABLE_MAP,
    SUPPORTED_DISEASES,
    StageResult,
    classify_abo,
    classify_all,
    classify_anem,
    classify_ckd,
    classify_dl,
    classify_dm,
    classify_fl,
    classify_htn,
    classify_kf,
    classify_lf,
    classify_obe,
)

stage_router = APIRouter(prefix="/health/stage", tags=["health-stage"])

FEATURE_UNITS = {
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "fasting_glucose": "mg/dL",
    "hba1c": "%",
    "total_cholesterol": "mg/dL",
    "ldl_cholesterol": "mg/dL",
    "triglyceride": "mg/dL",
    "hdl_cholesterol": "mg/dL",
    "bmi": "kg/m²",
    "height_cm": "cm",
    "weight_kg": "kg",
    "hemoglobin": "g/dL",
    "ast": "U/L",
    "alt": "U/L",
    "waist_cm": "cm",
    "gamma_gtp": "IU/L",
    "urine_protein": "정성 (음성/-/미량/±/+1~+4)",
    "creatinine": "mg/dL",
    "egfr": "mL/min/1.73m²",
}


class HealthStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    systolic_bp: int | None = Field(default=None, ge=40, le=300)
    diastolic_bp: int | None = Field(default=None, ge=20, le=200)
    fasting_glucose: int | None = Field(default=None, ge=30, le=1000)
    hba1c: Decimal | None = Field(default=None, ge=Decimal("3.0"), le=Decimal("20.0"))
    total_cholesterol: int | None = Field(default=None, ge=50, le=700)
    ldl_cholesterol: int | None = Field(default=None, ge=0, le=500)
    triglyceride: int | None = Field(default=None, ge=0, le=2000)
    hdl_cholesterol: int | None = Field(default=None, ge=0, le=200)
    bmi: Decimal | None = Field(default=None, ge=Decimal("10.0"), le=Decimal("80.0"))
    height_cm: Decimal | None = Field(default=None, ge=Decimal("50.0"), le=Decimal("250.0"))
    weight_kg: Decimal | None = Field(default=None, ge=Decimal("20.0"), le=Decimal("300.0"))
    hemoglobin: Decimal | None = Field(default=None, ge=Decimal("3.0"), le=Decimal("25.0"))
    gender: str | None = None
    ast: int | None = Field(default=None, ge=1, le=10000)
    alt: int | None = Field(default=None, ge=1, le=10000)
    waist_cm: Decimal | None = Field(default=None, ge=Decimal("30.0"), le=Decimal("250.0"))
    gamma_gtp: int | None = Field(default=None, ge=1, le=10000)
    urine_protein: str | None = None  # 예: "음성", "-", "미량", "±", "+1"~"+4", "1"~"4"
    creatinine: Decimal | None = Field(default=None, ge=Decimal("0.1"), le=Decimal("30.0"))
    egfr: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("200.0"))

    @field_validator("gender")
    @classmethod
    def normalize_gender_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def classifier_kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class HealthStageResponse(BaseModel):
    results: dict[str, dict[str, Any]]
    notice: str = "입력 수치 기준의 참고용 판정이며 의료 진단이 아닙니다."


@stage_router.post("", response_model=HealthStageResponse)
async def classify_health_stage(
    request: HealthStageRequest,
) -> HealthStageResponse:
    results = classify_all(**request.classifier_kwargs())
    return _response(results)


@stage_router.post("/{code}", response_model=HealthStageResponse)
async def classify_health_stage_by_code(code: str, request: HealthStageRequest) -> HealthStageResponse:
    normalized_code = code.strip().upper()
    kwargs = request.classifier_kwargs()
    if normalized_code == "HTN":
        result = classify_htn(
            systolic_bp=kwargs.get("systolic_bp"),
            diastolic_bp=kwargs.get("diastolic_bp"),
        )
    elif normalized_code == "DM":
        result = classify_dm(
            fasting_glucose=kwargs.get("fasting_glucose"),
            hba1c=kwargs.get("hba1c"),
        )
    elif normalized_code == "DL":
        result = classify_dl(
            total_cholesterol=kwargs.get("total_cholesterol"),
            ldl_cholesterol=kwargs.get("ldl_cholesterol"),
            triglyceride=kwargs.get("triglyceride"),
            hdl_cholesterol=kwargs.get("hdl_cholesterol"),
        )
    elif normalized_code == "OBE":
        result = classify_obe(
            bmi=kwargs.get("bmi"),
            height_cm=kwargs.get("height_cm"),
            weight_kg=kwargs.get("weight_kg"),
        )
    elif normalized_code == "ANEM":
        result = classify_anem(hemoglobin=kwargs.get("hemoglobin"), gender=kwargs.get("gender"))
    elif normalized_code == "FL":
        result = classify_fl(
            ast=kwargs.get("ast"),
            alt=kwargs.get("alt"),
            bmi=kwargs.get("bmi"),
            height_cm=kwargs.get("height_cm"),
            weight_kg=kwargs.get("weight_kg"),
            gender=kwargs.get("gender"),
        )
    elif normalized_code == "ABO":
        result = classify_abo(waist_cm=kwargs.get("waist_cm"), gender=kwargs.get("gender"))
    elif normalized_code == "LF":
        result = classify_lf(
            ast=kwargs.get("ast"),
            alt=kwargs.get("alt"),
            gamma_gtp=kwargs.get("gamma_gtp"),
            gender=kwargs.get("gender"),
        )
    elif normalized_code == "KF":
        result = classify_kf(
            urine_protein=kwargs.get("urine_protein"),
            creatinine=kwargs.get("creatinine"),
            egfr=kwargs.get("egfr"),
        )
    elif normalized_code == "CKD":
        result = classify_ckd(egfr=kwargs.get("egfr"))
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"지원하지 않는 판정 코드입니다. 지원 코드: {', '.join(SUPPORTED_DISEASES)}",
        )
    return _response({normalized_code: result})


@stage_router.get("/info")
async def get_health_stage_info() -> dict[str, Any]:
    return {
        "supported_codes": list(SUPPORTED_DISEASES),
        "service_fields": [
            "systolic_bp",
            "diastolic_bp",
            "fasting_glucose",
            "hba1c",
            "total_cholesterol",
            "ldl_cholesterol",
            "triglyceride",
            "hdl_cholesterol",
            "bmi",
            "height_cm",
            "weight_kg",
            "hemoglobin",
            "gender",
            "ast",
            "alt",
            "waist_cm",
            "gamma_gtp",
            "urine_protein",
            "creatinine",
            "egfr",
        ],
        "feature_units": FEATURE_UNITS,
        "source_variable_map": SOURCE_VARIABLE_MAP,
        "notice": "기준표 기반 참고용 판정입니다. 의료 진단이 아니며 의료기관 상담을 권장할 수 있습니다.",
    }


def _response(results: dict[str, StageResult]) -> HealthStageResponse:
    return HealthStageResponse(results={key: value.to_dict() for key, value in results.items()})
