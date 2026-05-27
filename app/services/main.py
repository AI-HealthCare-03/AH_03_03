from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from app.models.users import User
from app.services import analysis as analysis_service
from app.services import challenges as challenge_service
from app.services import diets as diet_service
from app.services import health as health_service
from app.services import medications as medication_service
from app.services import notifications as notification_service


async def get_public_main() -> dict[str, Any]:
    return {
        "service_title": "AI Health Manager",
        "service_description": "건강정보와 생활습관 기록을 바탕으로 만성질환 위험 관리와 실천 챌린지를 돕는 서비스입니다.",
        "health_highlights": [
            {"title": "혈당 관리", "value": "공복혈당 추적", "status": "login_required"},
            {"title": "체중 관리", "value": "BMI/허리둘레 확인", "status": "login_required"},
            {"title": "지질 관리", "value": "콜레스테롤 지표 확인", "status": "login_required"},
        ],
        "challenge_highlights": [
            {"title": "식후 10분 걷기", "category": "EXERCISE"},
            {"title": "물 6잔 마시기", "category": "HABIT"},
            {"title": "야식 줄이기", "category": "DIET"},
        ],
        "locked_features": [
            {"name": "건강 위험도 분석", "reason": "로그인 후 이용 가능"},
            {"name": "개인 대시보드", "reason": "로그인 후 이용 가능"},
            {"name": "맞춤 챌린지", "reason": "로그인 후 이용 가능"},
        ],
        "cta_buttons": [
            {"label": "회원가입", "url": "/signup"},
            {"label": "로그인", "url": "/login"},
        ],
    }


async def get_login_main_summary(user: User) -> dict[str, Any]:
    latest_health = await health_service.get_latest_health_record(user.id)
    analysis_results = await analysis_service.list_analysis_results(user.id, limit=1)
    user_challenges = await challenge_service.list_user_challenges(user.id, limit=5)
    active_challenges = await challenge_service.list_active_challenges(limit=5)
    unread_notifications = await notification_service.list_unread_notifications(user.id, limit=1000)
    recent_health_records = await health_service.list_health_records(user.id, limit=3)
    recent_diet_records = await diet_service.list_diet_records(user.id, limit=3)
    active_medications = await medication_service.list_medications(user.id, is_active=True, limit=5)
    recent_medication_records = await medication_service.list_medication_records(user_id=user.id, limit=3)

    latest_analysis = analysis_results[0] if analysis_results else None
    today_tasks = _build_today_tasks(user_challenges, active_medications)

    return {
        "user_profile_summary": {
            "id": user.id,
            "name": user.name,
            "nickname": user.nickname,
            "profile_image_url": user.profile_image_url,
        },
        "latest_health_summary": _to_response_payload(latest_health),
        "latest_analysis_summary": _to_response_payload(latest_analysis),
        "active_challenge_summary": {
            "active_challenge_count": len(active_challenges),
            "my_challenge_count": len(user_challenges),
            "my_challenges": _to_response_payload(user_challenges),
        },
        "dashboard_summary": {
            "has_health_record": latest_health is not None,
            "has_analysis_result": latest_analysis is not None,
            "active_medication_count": len(active_medications),
            "recent_diet_count": len(recent_diet_records),
        },
        "today_tasks": today_tasks,
        "notification_summary": {
            "unread_count": len(unread_notifications),
            "items": _to_response_payload(unread_notifications[:5]),
        },
        "recent_records": {
            "health_records": _to_response_payload(recent_health_records),
            "diet_records": _to_response_payload(recent_diet_records),
            "medication_records": _to_response_payload(recent_medication_records),
        },
        "ai_comment": _build_ai_comment(latest_health, latest_analysis),
    }


def _build_today_tasks(user_challenges: list[Any], active_medications: list[Any]) -> list[dict[str, Any]]:
    tasks = []
    for challenge in user_challenges[:3]:
        tasks.append(
            {
                "task_type": "CHALLENGE",
                "title": "오늘의 챌린지 수행",
                "target_id": challenge.id,
                "is_completed": False,
            }
        )
    for medication in active_medications[:3]:
        tasks.append(
            {
                "task_type": "MEDICATION",
                "title": f"{medication.name} 복용 기록",
                "target_id": medication.id,
                "is_completed": False,
            }
        )
    return tasks


def _build_ai_comment(latest_health: Any, latest_analysis: Any) -> str:
    if latest_analysis is not None:
        return (
            "최근 분석 결과를 바탕으로 생활습관을 꾸준히 기록해보세요. 맞춤 코멘트는 분석 결과 범위 안에서 제공합니다."
        )
    if latest_health is not None:
        return "건강정보가 저장되었습니다. 분석을 실행하면 맞춤 요약을 확인할 수 있습니다."
    return "건강정보를 입력하면 맞춤 관리 요약과 챌린지를 확인할 수 있습니다."


def _to_response_payload(value: Any) -> Any:
    """메인 요약은 여러 도메인의 Tortoise 모델을 모으므로 응답 직전에 JSON 안전 형태로 낮춘다."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list | tuple):
        return [_to_response_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_response_payload(item) for key, item in value.items()}

    model_meta = getattr(value, "_meta", None)
    field_names = getattr(model_meta, "db_fields", None)
    if field_names:
        return {
            field_name: _to_response_payload(getattr(value, field_name, None))
            for field_name in field_names
            if hasattr(value, field_name)
        }

    return str(value)
