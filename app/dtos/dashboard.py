from typing import Any

from pydantic import BaseModel

from app.dtos.health import HealthRecordResponse


class DashboardSummaryResponse(BaseModel):
    latest_health_record: HealthRecordResponse | None = None
    unread_notification_count: int
    active_challenge_count: int
    active_medication_count: int = 0


class DashboardHealthResponse(BaseModel):
    latest_health_record: HealthRecordResponse | None = None
    recent_health_records: list[HealthRecordResponse]


class DashboardChallengesResponse(BaseModel):
    active_challenges: list[Any]
    user_challenges: list[Any]


class DashboardDietsResponse(BaseModel):
    recent_diet_records: list[Any]


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
