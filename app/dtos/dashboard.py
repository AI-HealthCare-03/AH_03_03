from typing import Any

from pydantic import BaseModel


class DashboardSummaryResponse(BaseModel):
    latest_health_record: Any | None = None
    unread_notification_count: int
    active_challenge_count: int


class DashboardHealthResponse(BaseModel):
    latest_health_record: Any | None = None
    recent_health_records: list[Any]


class DashboardChallengesResponse(BaseModel):
    active_challenges: list[Any]
    user_challenges: list[Any]


class DashboardDietsResponse(BaseModel):
    recent_diet_records: list[Any]


class DashboardMedicationsResponse(BaseModel):
    active_medications: list[Any]
    recent_medication_records: list[Any]
