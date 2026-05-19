# ================================================================
# health_stage_router.py
# FastAPI 라우터 — 건강 단계 룰 기반 판정 API
#
# 기존 FastAPI 앱에 붙이는 방법:
#   from health_stage_router import router as health_router
#   app.include_router(health_router, prefix="/api/v1")
#
# 의존성:
#   pip install fastapi pydantic
#   health_stage_classifier.py 가 같은 디렉토리에 있어야 함
#
# 엔드포인트:
#   POST /api/v1/health/stage        — 5개 질환 통합 판정
#   POST /api/v1/health/stage/{code} — 단일 질환 판정 (HTN/DM/DL/OBE/ANEM)
#   GET  /api/v1/health/stage/info   — 판정 기준 안내
# ================================================================

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from health_stage_classifier import (
    StageResult,
    classify_all,
    classify_anem,
    classify_dl,
    classify_dm,
    classify_htn,
    classify_obe,
)
from pydantic import BaseModel, Field, validator

router = APIRouter(tags=["건강 단계 판정"])


# ================================================================
# Request / Response 스키마
# ================================================================
class HealthCheckInput(BaseModel):
    """건강검진 수치 입력 — 전부 Optional, 있는 것만 보내면 됨"""

    # 혈압
    sbp: float | None = Field(None, ge=60, le=250, description="수축기 혈압 (mmHg)")
    dbp: float | None = Field(None, ge=40, le=150, description="이완기 혈압 (mmHg)")

    # 혈당
    glu: float | None = Field(None, ge=50, le=600, description="공복혈당 (mg/dL)")
    hba1c: float | None = Field(None, ge=3.0, le=20, description="당화혈색소 (%)")

    # 지질
    chol: float | None = Field(None, ge=50, le=700, description="총콜레스테롤 (mg/dL)")
    ldl: float | None = Field(None, ge=10, le=500, description="LDL 콜레스테롤 (mg/dL)")
    tg: float | None = Field(None, ge=20, le=3000, description="중성지방 (mg/dL)")
    hdl: float | None = Field(None, ge=10, le=200, description="HDL 콜레스테롤 (mg/dL)")

    # 비만
    bmi: float | None = Field(None, ge=10, le=70, description="체질량지수 (kg/m²)")
    height_cm: float | None = Field(None, ge=100, le=250, description="신장 (cm)")
    weight_kg: float | None = Field(None, ge=20, le=300, description="체중 (kg)")

    # 빈혈
    hb: float | None = Field(None, ge=3.0, le=25, description="헤모글로빈 (g/dL)")
    sex: str | None = Field(None, description="성별 — 'M' 또는 'F'")

    @validator("sex")
    def validate_sex(self, v):
        if v is not None and str(v).upper() not in ("M", "F"):
            raise ValueError("sex는 'M' 또는 'F' 만 허용됩니다")
        return v


class StageResultSchema(BaseModel):
    """단일 질환 판정 결과"""

    disease: str
    stage: int | None
    label: str
    detail: str
    missing: list[str]
    is_normal: bool

    @classmethod
    def from_result(cls, r: StageResult) -> StageResultSchema:
        return cls(
            disease=r.disease,
            stage=r.stage,
            label=r.label,
            detail=r.detail,
            missing=r.missing,
            is_normal=r.is_normal(),
        )


class HealthStageResponse(BaseModel):
    """통합 판정 응답"""

    results: dict[str, StageResultSchema]
    classifiable: list[str]  # 판정된 질환 코드
    unclassifiable: list[str]  # 수치 부족으로 판정 불가 질환
    has_risk: bool  # 1개 이상 비정상 단계 존재 여부
    highest_stage: dict  # 가장 높은 단계 질환 요약


# ================================================================
# 헬퍼
# ================================================================
def _build_response(raw: dict[str, StageResult]) -> HealthStageResponse:
    results = {k: StageResultSchema.from_result(v) for k, v in raw.items()}

    classifiable = [k for k, v in raw.items() if v.is_classifiable()]
    unclassifiable = [k for k, v in raw.items() if not v.is_classifiable()]
    has_risk = any(not v.is_normal() for v in raw.values() if v.is_classifiable())

    # 가장 높은 단계 질환
    staged = {k: v for k, v in raw.items() if v.stage is not None}
    if staged:
        worst_key = max(staged, key=lambda k: staged[k].stage)
        worst = staged[worst_key]
        highest_stage = {
            "disease": worst_key,
            "stage": worst.stage,
            "label": worst.label,
        }
    else:
        highest_stage = {}

    return HealthStageResponse(
        results=results,
        classifiable=classifiable,
        unclassifiable=unclassifiable,
        has_risk=has_risk,
        highest_stage=highest_stage,
    )


# ================================================================
# 엔드포인트
# ================================================================


# ── 통합 판정 ─────────────────────────────────────────────────
@router.post(
    "/health/stage",
    response_model=HealthStageResponse,
    summary="5개 질환 통합 판정",
    description="입력된 검진 수치로 HTN/DM/DL/OBE/ANEM 단계를 한 번에 판정합니다. 수치가 없는 항목은 '판정 불가'로 반환됩니다.",
)
async def stage_all(body: HealthCheckInput):
    raw = classify_all(
        sbp=body.sbp,
        dbp=body.dbp,
        glu=body.glu,
        hba1c=body.hba1c,
        chol=body.chol,
        ldl=body.ldl,
        tg=body.tg,
        hdl=body.hdl,
        bmi=body.bmi,
        height_cm=body.height_cm,
        weight_kg=body.weight_kg,
        hb=body.hb,
        sex=body.sex,
    )
    return _build_response(raw)


# ── 단일 질환 판정 ─────────────────────────────────────────────
SINGLE_CLASSIFIERS = {
    "HTN": lambda b: classify_htn(sbp=b.sbp, dbp=b.dbp),
    "DM": lambda b: classify_dm(glu=b.glu, hba1c=b.hba1c),
    "DL": lambda b: classify_dl(chol=b.chol, ldl=b.ldl, tg=b.tg, hdl=b.hdl),
    "OBE": lambda b: classify_obe(bmi=b.bmi, height_cm=b.height_cm, weight_kg=b.weight_kg),
    "ANEM": lambda b: classify_anem(hb=b.hb, sex=b.sex),
}


@router.post(
    "/health/stage/{code}",
    response_model=StageResultSchema,
    summary="단일 질환 판정",
    description="code: HTN / DM / DL / OBE / ANEM",
)
async def stage_single(code: str, body: HealthCheckInput):
    code = code.upper()
    if code not in SINGLE_CLASSIFIERS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 질환 코드입니다. 가능한 값: {list(SINGLE_CLASSIFIERS.keys())}",
        )
    result = SINGLE_CLASSIFIERS[code](body)
    return StageResultSchema.from_result(result)


# ── 판정 기준 안내 ─────────────────────────────────────────────
@router.get(
    "/health/stage/info",
    summary="판정 기준 안내",
)
async def stage_info():
    return {
        "HTN": {
            "name": "고혈압",
            "inputs": ["sbp (수축기혈압 mmHg)", "dbp (이완기혈압 mmHg)"],
            "stages": {
                0: "정상 — sbp<120 AND dbp<80",
                1: "주의혈압 — sbp 120~129 AND dbp<80",
                2: "고혈압 전단계 — sbp 130~139 OR dbp 80~89",
                3: "고혈압 1단계 — sbp 140~159 OR dbp 90~99",
                4: "고혈압 2단계 — sbp≥160 OR dbp≥100",
            },
            "reference": "AHA/ACC 2017",
        },
        "DM": {
            "name": "당뇨병",
            "inputs": ["glu (공복혈당 mg/dL)", "hba1c (당화혈색소 %)"],
            "stages": {
                0: "정상 — glu<100 AND hba1c<5.7",
                1: "공복혈당장애 — glu 100~125 OR hba1c 5.7~6.4",
                2: "당뇨병 의심 — glu≥126 OR hba1c≥6.5",
            },
            "reference": "ADA 2023 / 대한당뇨병학회",
        },
        "DL": {
            "name": "이상지질혈증",
            "inputs": ["chol (총콜레스테롤)", "ldl (LDL)", "tg (중성지방)", "hdl (HDL)"],
            "stages": {
                0: "정상",
                1: "경계",
                2: "위험",
                3: "고위험",
            },
            "reference": "한국지질동맥경화학회 2022",
        },
        "OBE": {
            "name": "비만",
            "inputs": ["bmi (kg/m²) 또는 height_cm + weight_kg"],
            "stages": {
                0: "저체중 — BMI<18.5",
                1: "정상 — BMI 18.5~22.9",
                2: "비만 전단계 — BMI 23~24.9",
                3: "비만 1단계 — BMI 25~29.9",
                4: "비만 2단계 — BMI 30~34.9",
                5: "비만 3단계 — BMI≥35",
            },
            "reference": "대한비만학회 2022 (아시아-태평양 기준)",
        },
        "ANEM": {
            "name": "빈혈",
            "inputs": ["hb (헤모글로빈 g/dL)", "sex (M/F)"],
            "stages": {
                0: "정상 — 남 Hb≥13.0 / 여 Hb≥12.0",
                1: "경증 빈혈 — Hb 11.0~기준 미만",
                2: "중등도 빈혈 — Hb 8.0~10.9",
                3: "중증 빈혈 — Hb<8.0",
            },
            "reference": "WHO 기준",
        },
    }
