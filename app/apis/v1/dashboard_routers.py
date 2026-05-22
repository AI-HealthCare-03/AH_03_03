from datetime import date, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.apis.v1.dependencies import get_request_user
from app.core import config
from app.dtos.dashboard import (
    DashboardChallengesResponse,
    DashboardDietsResponse,
    DashboardHealthResponse,
    DashboardMedicationsResponse,
    DashboardSummaryResponse,
    DashboardTrendsResponse,
)
from app.models.users import User
from app.services import analysis as analysis_service
from app.services import challenges as challenge_service
from app.services import diets as diet_service
from app.services import health as health_service
from app.services import medications as medication_service
from app.services import notifications as notification_service
from app.services.sensitive_access_logs import safe_record_sensitive_access

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])

TREND_PERIOD_DAYS = {
    "week": 7,
    "month": 30,
    "quarter": 90,
    "three_months": 90,
    "year": 365,
    "all": None,
}


def _normalize_period(period: str) -> tuple[str, date | None, date]:
    if period not in TREND_PERIOD_DAYS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="지원하지 않는 기간입니다.")
    normalized_period = "quarter" if period == "three_months" else period
    date_to = datetime.now(config.TIMEZONE).date()
    days = TREND_PERIOD_DAYS[period]
    date_from = None if days is None else date_to - timedelta(days=days - 1)
    return normalized_period, date_from, date_to


def _in_range(value: date, date_from: date | None, date_to: date) -> bool:
    if date_from is not None and value < date_from:
        return False
    return value <= date_to


def _overall_risk_level(results):
    levels = [result.risk_level for result in results]
    if any(level.value == "HIGH" for level in levels):
        return "HIGH"
    if any(level.value == "MEDIUM" for level in levels):
        return "MEDIUM"
    if levels:
        return "LOW"
    return None


@dashboard_router.get("/summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary(request: Request, user: Annotated[User, Depends(get_request_user)]):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="DASHBOARD",
        access_reason="dashboard.summary",
    )
    latest_health_record = await health_service.get_latest_health_record(user.id)
    unread_notifications = await notification_service.list_unread_notifications(user.id, limit=1000)
    active_challenge_count = await challenge_service.count_active_user_challenges(user.id)
    active_medications = await medication_service.list_medications(user.id, is_active=True, limit=1000)
    latest_analysis_results = await analysis_service.list_latest_analysis_results(user.id)
    top_risk_factors = []
    for result in latest_analysis_results:
        for factor in await analysis_service.list_analysis_factors(result.id):
            top_risk_factors.append(
                {
                    "analysis_result_id": result.id,
                    "analysis_type": result.analysis_type,
                    "analysis_mode": result.analysis_mode,
                    "factor_key": factor.factor_key,
                    "factor_name": factor.factor_name,
                    "factor_value": factor.factor_value,
                    "contribution_score": float(factor.contribution_score)
                    if factor.contribution_score is not None
                    else None,
                    "direction": factor.direction.value
                    if hasattr(factor.direction, "value")
                    else str(factor.direction),
                }
            )
    top_risk_factors.sort(key=lambda item: item["contribution_score"] or 0, reverse=True)
    analysis_scores = [float(result.risk_score) for result in latest_analysis_results if result.risk_score is not None]
    return {
        "latest_health_record": latest_health_record,
        "unread_notification_count": len(unread_notifications),
        "active_challenge_count": active_challenge_count,
        "active_medication_count": len(active_medications),
        "latest_analysis_results": [
            {
                "id": result.id,
                "analysis_type": result.analysis_type,
                "analysis_mode": result.analysis_mode,
                "risk_level": result.risk_level,
                "risk_score": float(result.risk_score),
                "summary": result.summary,
                "analyzed_at": result.analyzed_at.isoformat(),
                "created_at": result.created_at.isoformat(),
            }
            for result in latest_analysis_results
        ],
        "top_risk_factors": top_risk_factors[:5],
        "overall_risk_level": _overall_risk_level(latest_analysis_results),
        "overall_risk_score": max(analysis_scores) if analysis_scores else None,
    }


@dashboard_router.get("/health", response_model=DashboardHealthResponse)
async def get_dashboard_health(request: Request, user: Annotated[User, Depends(get_request_user)]):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="DASHBOARD",
        access_reason="dashboard.health",
    )
    return {
        "latest_health_record": await health_service.get_latest_health_record(user.id),
        "recent_health_records": await health_service.list_health_records(user.id, limit=5),
    }


@dashboard_router.get("/challenges", response_model=DashboardChallengesResponse)
async def get_dashboard_challenges(user: Annotated[User, Depends(get_request_user)]):
    return {
        "active_challenges": await challenge_service.list_active_challenges(limit=10),
        "user_challenges": await challenge_service.list_user_challenges(user.id, limit=10),
    }


@dashboard_router.get("/diets", response_model=DashboardDietsResponse)
async def get_dashboard_diets(user: Annotated[User, Depends(get_request_user)]):
    return {"recent_diet_records": await diet_service.list_diet_records(user.id, limit=10)}


@dashboard_router.get("/medications", response_model=DashboardMedicationsResponse)
async def get_dashboard_medications(user: Annotated[User, Depends(get_request_user)]):
    return {
        "active_medications": await medication_service.list_medications(user.id, is_active=True, limit=10),
        "recent_medication_records": await medication_service.list_medication_records(user_id=user.id, limit=10),
    }


@dashboard_router.get("/trends", response_model=DashboardTrendsResponse)
async def get_dashboard_trends(
    request: Request,
    user: Annotated[User, Depends(get_request_user)],
    period: str = "week",
):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="DASHBOARD",
        access_reason="dashboard.trends",
    )
    normalized_period, date_from, date_to = _normalize_period(period)
    query_limit = 1000 if date_from is None else TREND_PERIOD_DAYS[period] or 1000
    health_records = [
        record
        for record in await health_service.list_health_records(user.id, limit=query_limit)
        if _in_range(record.measured_at.date(), date_from, date_to)
    ]
    diet_records = [
        record
        for record in await diet_service.list_diet_records(user.id, limit=query_limit)
        if _in_range(record.created_at.date(), date_from, date_to)
    ]
    user_challenges = await challenge_service.list_user_challenges(user.id, limit=1000)

    challenge_rates = []
    for user_challenge in user_challenges:
        logs = [
            log
            for log in await challenge_service.list_challenge_logs(user_challenge.id)
            if _in_range(log.log_date, date_from, date_to)
        ]
        if not logs and not _in_range(user_challenge.started_at.date(), date_from, date_to):
            continue
        if logs:
            completed_count = sum(1 for log in logs if log.is_completed)
            rate = round(completed_count / len(logs) * 100, 2)
        else:
            rate = 0.0
        challenge_rates.append(
            {
                "date": user_challenge.started_at.date().isoformat(),
                "value": rate,
                "user_challenge_id": user_challenge.id,
            }
        )

    return {
        "period": normalized_period,
        "date_from": date_from.isoformat() if date_from is not None else None,
        "date_to": date_to.isoformat(),
        "glucose": [
            {
                "date": record.measured_at.date().isoformat(),
                "value": record.fasting_glucose,
            }
            for record in health_records
            if record.fasting_glucose is not None
        ],
        "blood_pressure": [
            {
                "date": record.measured_at.date().isoformat(),
                "systolic": record.systolic_bp,
                "diastolic": record.diastolic_bp,
            }
            for record in health_records
            if record.systolic_bp is not None or record.diastolic_bp is not None
        ],
        "weight": [
            {
                "date": record.measured_at.date().isoformat(),
                "value": float(record.weight_kg),
            }
            for record in health_records
            if record.weight_kg is not None
        ],
        "challenge_completion_rate": challenge_rates,
        "diet_score": [
            {
                "date": record.created_at.date().isoformat(),
                "value": record.diet_score,
            }
            for record in diet_records
            if record.diet_score is not None
        ],
    }
