from typing import Any

from pydantic import BaseModel

from app.dtos.challenges import ChallengeResponse, UserChallengeResponse
from app.dtos.diets import DietRecordResponse
from app.dtos.health import HealthRecordResponse
from app.models.analysis import AnalysisMode, AnalysisType, RiskLevel


class DashboardAnalysisResultResponse(BaseModel):
    id: int
    analysis_type: AnalysisType
    analysis_mode: AnalysisMode = AnalysisMode.BASIC
    risk_level: RiskLevel
    risk_score: float
    service_band: str | None = None
    service_band_label: str | None = None
    service_band_percent: int | None = None
    legacy_risk_level: str | None = None
    result_source: str | None = None
    x2_stage_code: str | None = None
    x2_stage_label: str | None = None
    x2_available: bool | None = None
    x2_missing_fields: list[str] | None = None
    selected_exam_report_id: int | None = None
    x2_measurement_source: str | None = None
    summary: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    analyzed_at: str
    created_at: str


class DashboardRiskFactorResponse(BaseModel):
    analysis_result_id: int
    analysis_type: AnalysisType
    analysis_mode: AnalysisMode = AnalysisMode.BASIC
    factor_key: str
    factor_name: str
    factor_value: str | None = None
    contribution_score: float | None = None
    direction: str


class DashboardSummaryResponse(BaseModel):
    latest_health_record: HealthRecordResponse | None = None
    unread_notification_count: int
    active_challenge_count: int
    active_medication_count: int = 0
    latest_analysis_results: list[DashboardAnalysisResultResponse] = []
    top_risk_factors: list[DashboardRiskFactorResponse] = []
    overall_risk_level: RiskLevel | None = None
    overall_risk_score: float | None = None


class DashboardHealthResponse(BaseModel):
    latest_health_record: HealthRecordResponse | None = None
    recent_health_records: list[HealthRecordResponse]


class DashboardChallengesResponse(BaseModel):
    active_challenges: list[ChallengeResponse]
    user_challenges: list[UserChallengeResponse]


class DashboardDietsResponse(BaseModel):
    recent_diet_records: list[DietRecordResponse]


class DashboardMedicationsResponse(BaseModel):
    active_medications: list[Any]
    recent_medication_records: list[Any]


class DashboardTrendsResponse(BaseModel):
    period: str
    date_from: str | None = None
    date_to: str | None = None
    glucose: list[dict[str, Any]]
    blood_pressure: list[dict[str, Any]]
    weight: list[dict[str, Any]]
    challenge_completion_rate: list[dict[str, Any]]
    diet_score: list[dict[str, Any]]


class DashboardRiskTrendPointResponse(BaseModel):
    analyzed_at: str
    risk_score: float
    risk_level: RiskLevel
    service_band: str | None = None
    service_band_label: str | None = None
    service_band_percent: int | None = None
    legacy_risk_level: str | None = None


class DashboardRiskTrendSeriesResponse(BaseModel):
    disease_type: AnalysisType
    points: list[DashboardRiskTrendPointResponse]


class DashboardRiskTrendResponse(BaseModel):
    period: str
    date_from: str | None = None
    date_to: str | None = None
    series: list[DashboardRiskTrendSeriesResponse]
