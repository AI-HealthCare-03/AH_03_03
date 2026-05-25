from datetime import date, datetime, timedelta
from typing import Any

from app.core import config
from app.models.analysis import RiskLevel
from app.services import analysis as analysis_service
from app.services import challenges as challenge_service
from app.services import diets as diet_service
from app.services import health as health_service
from app.services import medications as medication_service
from app.services import notifications as notification_service

TREND_PERIOD_DAYS: dict[str, int | None] = {
    "week": 7,
    "month": 30,
    "quarter": 90,
    "three_months": 90,
    "year": 365,
    "all": None,
}


def normalize_period(period: str) -> tuple[str, date | None, date]:
    if period not in TREND_PERIOD_DAYS:
        raise ValueError("지원하지 않는 기간입니다.")
    normalized_period = "quarter" if period == "three_months" else period
    date_to = datetime.now(config.TIMEZONE).date()
    days = TREND_PERIOD_DAYS[period]
    date_from = None if days is None else date_to - timedelta(days=days - 1)
    return normalized_period, date_from, date_to


def _in_range(value: date, date_from: date | None, date_to: date) -> bool:
    if date_from is not None and value < date_from:
        return False
    return value <= date_to


def _overall_risk_level(results: list[Any]) -> str | None:
    levels = [result.risk_level for result in results]
    if any(level == RiskLevel.HIGH or getattr(level, "value", None) == "HIGH" for level in levels):
        return "HIGH"
    if any(level == RiskLevel.MEDIUM or getattr(level, "value", None) == "MEDIUM" for level in levels):
        return "MEDIUM"
    if levels:
        return "LOW"
    return None


async def get_dashboard_summary(user_id: int) -> dict[str, Any]:
    latest_health_record = await health_service.get_latest_health_record(user_id)
    unread_notifications = await notification_service.list_unread_notifications(user_id, limit=1000)
    active_challenge_count = await challenge_service.count_active_user_challenges(user_id)
    active_medications = await medication_service.list_medications(user_id, is_active=True, limit=1000)
    latest_analysis_results = await analysis_service.list_latest_analysis_results(user_id)
    top_risk_factors = await _build_top_risk_factors(latest_analysis_results)
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
                "model_name": result.model_name,
                "model_version": result.model_version,
                "analyzed_at": result.analyzed_at.isoformat(),
                "created_at": result.created_at.isoformat(),
            }
            for result in latest_analysis_results
        ],
        "top_risk_factors": top_risk_factors,
        "overall_risk_level": _overall_risk_level(latest_analysis_results),
        "overall_risk_score": max(analysis_scores) if analysis_scores else None,
    }


async def _build_top_risk_factors(latest_analysis_results: list[Any]) -> list[dict[str, Any]]:
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
    return top_risk_factors[:5]


async def get_dashboard_health(user_id: int) -> dict[str, Any]:
    return {
        "latest_health_record": await health_service.get_latest_health_record(user_id),
        "recent_health_records": await health_service.list_health_records(user_id, limit=5),
    }


async def get_dashboard_challenges(user_id: int) -> dict[str, Any]:
    return {
        "active_challenges": await challenge_service.list_active_challenges(limit=10),
        "user_challenges": await challenge_service.list_user_challenges(user_id, limit=10),
    }


async def get_dashboard_diets(user_id: int) -> dict[str, Any]:
    return {"recent_diet_records": await diet_service.list_diet_records(user_id, limit=10)}


async def get_dashboard_medications(user_id: int) -> dict[str, Any]:
    return {
        "active_medications": await medication_service.list_medications(user_id, is_active=True, limit=10),
        "recent_medication_records": await medication_service.list_medication_records(user_id=user_id, limit=10),
    }


async def get_dashboard_trends(user_id: int, period: str) -> dict[str, Any]:
    normalized_period, date_from, date_to = normalize_period(period)
    query_limit = 1000 if date_from is None else TREND_PERIOD_DAYS[period] or 1000
    health_records = [
        record
        for record in await health_service.list_health_records(user_id, limit=query_limit)
        if _in_range(record.measured_at.date(), date_from, date_to)
    ]
    diet_records = [
        record
        for record in await diet_service.list_diet_records(user_id, limit=query_limit)
        if _in_range(record.created_at.date(), date_from, date_to)
    ]
    return {
        "period": normalized_period,
        "date_from": date_from.isoformat() if date_from is not None else None,
        "date_to": date_to.isoformat(),
        "glucose": _build_glucose_series(health_records),
        "blood_pressure": _build_blood_pressure_series(health_records),
        "weight": _build_weight_series(health_records),
        "challenge_completion_rate": await _build_challenge_completion_rates(user_id, date_from, date_to),
        "diet_score": _build_diet_score_series(diet_records),
    }


def _build_glucose_series(health_records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "date": record.measured_at.date().isoformat(),
            "value": record.fasting_glucose,
        }
        for record in health_records
        if record.fasting_glucose is not None
    ]


def _build_blood_pressure_series(health_records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "date": record.measured_at.date().isoformat(),
            "systolic": record.systolic_bp,
            "diastolic": record.diastolic_bp,
        }
        for record in health_records
        if record.systolic_bp is not None or record.diastolic_bp is not None
    ]


def _build_weight_series(health_records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "date": record.measured_at.date().isoformat(),
            "value": float(record.weight_kg),
        }
        for record in health_records
        if record.weight_kg is not None
    ]


async def _build_challenge_completion_rates(
    user_id: int,
    date_from: date | None,
    date_to: date,
) -> list[dict[str, Any]]:
    challenge_rates = []
    user_challenges = await challenge_service.list_user_challenges(user_id, limit=1000)
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
    return challenge_rates


def _build_diet_score_series(diet_records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "date": record.created_at.date().isoformat(),
            "value": record.diet_score,
        }
        for record in diet_records
        if record.diet_score is not None
    ]
