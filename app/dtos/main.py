from typing import Any

from pydantic import BaseModel


class MainPublicResponse(BaseModel):
    service_title: str
    service_description: str
    health_highlights: list[dict[str, Any]]
    challenge_highlights: list[dict[str, Any]]
    locked_features: list[dict[str, Any]]
    cta_buttons: list[dict[str, str]]


class MainSummaryResponse(BaseModel):
    user_profile_summary: dict[str, Any]
    latest_health_summary: Any | None = None
    latest_analysis_summary: Any | None = None
    active_challenge_summary: dict[str, Any]
    dashboard_summary: dict[str, Any]
    today_tasks: list[dict[str, Any]]
    notification_summary: dict[str, Any]
    recent_records: dict[str, Any]
    ai_comment: str
