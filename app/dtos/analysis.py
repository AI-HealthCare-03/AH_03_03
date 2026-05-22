from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from app.dtos.base import BaseSerializerModel
from app.models.analysis import AnalysisMode, AnalysisType, FactorDirection, RiskLevel


class AnalysisRunRequest(BaseModel):
    analysis_type: AnalysisType
    health_record_id: int | None = None
    exam_report_id: int | None = None
    mode: AnalysisMode = AnalysisMode.BASIC


class DummyAnalysisRunRequest(BaseModel):
    health_record_id: int
    mode: AnalysisMode = AnalysisMode.BASIC


class AnalysisRunByHealthRecordRequest(DummyAnalysisRunRequest):
    pass


class AnalysisResultCreateRequest(BaseModel):
    health_record_id: int
    async_job_id: int | None = None
    analysis_type: AnalysisType
    analysis_mode: AnalysisMode = AnalysisMode.BASIC
    risk_score: Decimal
    risk_level: RiskLevel
    summary: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    analyzed_at: datetime


class AnalysisResultResponse(BaseSerializerModel):
    id: int
    user_id: int
    health_record_id: int
    async_job_id: int | None
    analysis_type: AnalysisType
    analysis_mode: AnalysisMode
    risk_score: Decimal
    risk_level: RiskLevel
    summary: str | None
    model_name: str | None
    model_version: str | None
    analyzed_at: datetime
    created_at: datetime
    updated_at: datetime


class AnalysisResultFactorCreateRequest(BaseModel):
    factor_key: str
    factor_name: str
    factor_value: str | None = None
    contribution_score: Decimal | None = None
    direction: FactorDirection
    display_order: int = 0


class AnalysisResultFactorResponse(BaseSerializerModel):
    id: int
    analysis_result_id: int
    factor_key: str
    factor_name: str
    factor_value: str | None
    contribution_score: Decimal | None
    direction: FactorDirection
    display_order: int
    created_at: datetime


class AnalysisSnapshotCreateRequest(BaseModel):
    input_payload: dict[str, Any]
    output_payload: dict[str, Any]
    shap_payload: dict[str, Any] | None = None
    model_payload: dict[str, Any] | None = None


class AnalysisSnapshotResponse(BaseSerializerModel):
    id: int
    analysis_result_id: int
    input_payload: dict[str, Any]
    output_payload: dict[str, Any]
    shap_payload: dict[str, Any] | None
    model_payload: dict[str, Any] | None
    created_at: datetime


class AnalysisResultDetailResponse(BaseModel):
    result: AnalysisResultResponse
    factors: list[AnalysisResultFactorResponse]
    snapshot: AnalysisSnapshotResponse | None = None


class DummyAnalysisResultResponse(BaseModel):
    analysis_result_id: int
    analysis_type: AnalysisType
    analysis_mode: AnalysisMode
    risk_score: Decimal
    risk_level: RiskLevel
    guide_message: str
    challenge_recommendation_ids: list[int]
    factor_count: int = 0


class AnalysisRunResultResponse(DummyAnalysisResultResponse):
    pass
