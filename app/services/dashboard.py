from datetime import date, datetime, timedelta
from typing import Any

from app.core import config
from app.dtos.challenges import ChallengeResponse, UserChallengeResponse
from app.dtos.diets import DietRecordResponse
from app.models.analysis import AnalysisResult, AnalysisType, RiskLevel
from app.services import analysis as analysis_service
from app.services import challenges as challenge_service
from app.services import diets as diet_service
from app.services import health as health_service
from app.services import medications as medication_service
from app.services import notifications as notification_service

TREND_PERIOD_DAYS: dict[str, int | None] = {
    "today": 1,
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
    priority = {
        RiskLevel.LOW.value: 0,
        RiskLevel.ATTENTION.value: 1,
        RiskLevel.CAUTION.value: 2,
        RiskLevel.HIGH_CAUTION.value: 3,
        "MEDIUM": 2,
        "HIGH": 3,
    }
    levels = [getattr(result.risk_level, "value", result.risk_level) for result in results]
    if levels:
        return max(levels, key=lambda level: priority.get(str(level), 0))
    return None


async def get_dashboard_summary(user_id: int) -> dict[str, Any]:
    latest_health_record = await health_service.get_latest_health_record(user_id)
    unread_notifications = await notification_service.list_unread_notifications(user_id, limit=1000)
    active_challenge_count = await challenge_service.count_active_user_challenges(user_id)
    active_medications = await medication_service.list_medications(user_id, is_active=True, limit=1000)
    latest_analysis_results = await analysis_service.list_latest_analysis_results(user_id)
    latest_analysis_result_responses = await analysis_service.list_analysis_result_responses(latest_analysis_results)
    top_risk_factors = await _build_top_risk_factors(latest_analysis_results)
    analysis_scores = [float(result.risk_score) for result in latest_analysis_results if result.risk_score is not None]

    return {
        "latest_health_record": latest_health_record,
        "unread_notification_count": len(unread_notifications),
        "active_challenge_count": active_challenge_count,
        "active_medication_count": len(active_medications),
        "latest_analysis_results": [
            _dashboard_analysis_result_response(payload) for payload in latest_analysis_result_responses
        ],
        "top_risk_factors": top_risk_factors,
        "overall_risk_level": _overall_risk_level(latest_analysis_results),
        "overall_risk_score": max(analysis_scores) if analysis_scores else None,
    }


def _dashboard_analysis_result_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": payload["id"],
        "analysis_type": payload["analysis_type"],
        "analysis_mode": payload["analysis_mode"],
        "risk_level": payload["risk_level"],
        "risk_score": float(payload["risk_score"]),
        "service_band": payload.get("service_band"),
        "service_band_label": payload.get("service_band_label"),
        "service_band_percent": payload.get("service_band_percent"),
        "legacy_risk_level": payload.get("legacy_risk_level"),
        "result_source": payload.get("result_source"),
        "x2_stage_code": payload.get("x2_stage_code"),
        "x2_stage_label": payload.get("x2_stage_label"),
        "x2_available": payload.get("x2_available"),
        "x2_missing_fields": payload.get("x2_missing_fields"),
        "selected_exam_report_id": payload.get("selected_exam_report_id"),
        "x2_measurement_source": payload.get("x2_measurement_source"),
        "summary": payload.get("summary"),
        "model_name": payload.get("model_name"),
        "model_version": payload.get("model_version"),
        "analyzed_at": _isoformat_payload_datetime(payload.get("analyzed_at")),
        "created_at": _isoformat_payload_datetime(payload.get("created_at")),
    }


def _isoformat_payload_datetime(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


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
    active_challenges = await challenge_service.list_active_challenges(limit=10)
    user_challenges = await challenge_service.list_user_challenges(user_id, limit=10)
    user_challenge_summaries = await _build_user_challenge_dashboard_responses(user_challenges)
    return {
        "active_challenges": [
            ChallengeResponse.model_validate(challenge).model_dump(mode="json") for challenge in active_challenges
        ],
        "user_challenges": user_challenge_summaries,
    }


async def _build_user_challenge_dashboard_responses(user_challenges: list[Any]) -> list[dict[str, Any]]:
    challenge_ids = {int(user_challenge.challenge_id) for user_challenge in user_challenges}
    challenge_by_id = {}
    for challenge_id in challenge_ids:
        challenge = await challenge_service.get_challenge(challenge_id)
        if challenge is not None:
            challenge_by_id[challenge_id] = challenge
    responses = []
    for user_challenge in user_challenges:
        payload = UserChallengeResponse.model_validate(user_challenge).model_dump(mode="json")
        challenge = challenge_by_id.get(int(user_challenge.challenge_id))
        if challenge is not None:
            payload.update(
                {
                    "challenge_title": challenge.title,
                    "challenge_description": challenge.description,
                    "challenge_category": challenge.category.value
                    if hasattr(challenge.category, "value")
                    else challenge.category,
                    "challenge_difficulty": challenge.difficulty.value
                    if hasattr(challenge.difficulty, "value")
                    else challenge.difficulty,
                    "challenge_status": challenge.status.value
                    if hasattr(challenge.status, "value")
                    else challenge.status,
                    "challenge_duration_days": challenge.duration_days,
                }
            )
        responses.append(payload)
    return responses


async def get_dashboard_diets(user_id: int) -> dict[str, Any]:
    diet_records = await diet_service.list_diet_records(user_id, limit=10)
    return {
        "recent_diet_records": [
            DietRecordResponse.model_validate(record).model_dump(mode="json") for record in diet_records
        ]
    }


async def get_dashboard_medications(user_id: int) -> dict[str, Any]:
    return {
        "active_medications": await medication_service.list_medications(user_id, is_active=True, limit=10),
        "recent_medication_records": await medication_service.list_medication_records(user_id=user_id, limit=10),
    }


async def get_dashboard_trends(user_id: int, period: str) -> dict[str, Any]:
    normalized_period, date_from, date_to = normalize_period(period)
    query_limit = 1000 if date_from is None else TREND_PERIOD_DAYS[period] or 1000
    health_records = _sort_health_records_for_trend(
        [
            record
            for record in await health_service.list_health_records(user_id, limit=query_limit)
            if _in_range(record.measured_at.date(), date_from, date_to)
        ]
    )
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


async def get_dashboard_risk_trend(user_id: int, period: str = "all") -> dict[str, Any]:
    normalized_period, date_from, date_to = normalize_period(period)
    results = await AnalysisResult.filter(user_id=user_id).order_by("analysis_type", "analyzed_at").limit(1000)
    series_by_disease: dict[AnalysisType, list[dict[str, Any]]] = {}

    for result in results:
        analyzed_at = result.analyzed_at.astimezone(config.TIMEZONE)
        if not _in_range(analyzed_at.date(), date_from, date_to):
            continue
        series_by_disease.setdefault(result.analysis_type, []).append(
            {
                "analyzed_at": analyzed_at.isoformat(),
                "risk_score": float(result.risk_score),
                "risk_level": result.risk_level,
            }
            | analysis_service._risk_level_alias_fields(result.risk_level)
        )

    return {
        "period": normalized_period,
        "date_from": date_from.isoformat() if date_from is not None else None,
        "date_to": date_to.isoformat(),
        "series": [
            {
                "disease_type": disease_type,
                "points": points,
            }
            for disease_type, points in series_by_disease.items()
        ],
    }


def _health_record_sort_key(record: Any) -> tuple[datetime, datetime, int]:
    measured_at = getattr(record, "measured_at", None)
    created_at = getattr(record, "created_at", None)
    record_id = getattr(record, "id", None)
    fallback_datetime = datetime.min.replace(tzinfo=config.TIMEZONE)
    return (
        measured_at if isinstance(measured_at, datetime) else fallback_datetime,
        created_at if isinstance(created_at, datetime) else fallback_datetime,
        int(record_id) if record_id is not None else 0,
    )


def _sort_health_records_for_trend(health_records: list[Any]) -> list[Any]:
    return sorted(health_records, key=_health_record_sort_key)


def _health_record_trend_metadata(record: Any) -> dict[str, Any]:
    measured_at = getattr(record, "measured_at", None)
    created_at = getattr(record, "created_at", None)
    return {
        "id": getattr(record, "id", None),
        "measured_at": measured_at.isoformat() if hasattr(measured_at, "isoformat") else None,
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else None,
    }


def _build_glucose_series(health_records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "date": record.measured_at.date().isoformat(),
            "value": record.fasting_glucose,
            **_health_record_trend_metadata(record),
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
            **_health_record_trend_metadata(record),
        }
        for record in health_records
        if record.systolic_bp is not None or record.diastolic_bp is not None
    ]


def _build_weight_series(health_records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "date": record.measured_at.date().isoformat(),
            "value": float(record.weight_kg),
            **_health_record_trend_metadata(record),
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
        logs = []
        for log in await challenge_service.list_challenge_logs(user_challenge.id):
            if log.completed_at is None:
                continue
            completed_date = log.completed_at.astimezone(config.TIMEZONE).date()
            if _in_range(completed_date, date_from, date_to):
                logs.append((log, completed_date))
        if not logs:
            continue
        if logs:
            completed_count = sum(1 for log, _ in logs if log.is_completed)
            rate = round(completed_count / len(logs) * 100, 2)
        else:
            rate = 0.0
        challenge_rates.append(
            {
                "date": logs[-1][1].isoformat(),
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
