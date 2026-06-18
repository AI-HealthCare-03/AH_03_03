import re
from collections import Counter
from datetime import date, datetime, time, timedelta
from math import ceil
from types import SimpleNamespace

from fastapi import HTTPException
from starlette import status

from app.core import config
from app.dtos.challenges import (
    ChallengeCreateRequest,
    ChallengeLogCreateRequest,
    ChallengeRecommendationCreateRequest,
    UserChallengeCreateRequest,
    UserChallengeUpdateRequest,
)
from app.models.challenges import (
    Challenge,
    ChallengeCategory,
    ChallengeLog,
    ChallengeRecommendation,
    ChallengeStatus,
    UserChallenge,
    UserChallengeStatus,
)
from app.repositories import challenge_repository

DEFAULT_CHALLENGE_DURATION_DAYS = 7
CHALLENGE_COMPLETION_THRESHOLD = 0.8
RECOMMENDATION_FALLBACK_REASON_BY_CATEGORY = {
    ChallengeCategory.BLOOD_PRESSURE.value: "혈압 관리 관점에서 부담 없이 시작하기 좋은 챌린지입니다.",
    ChallengeCategory.BLOOD_GLUCOSE.value: "혈당 관리 관점에서 식사와 생활 리듬을 점검하기 좋은 챌린지입니다.",
    ChallengeCategory.DIET.value: "식단 관리 관점에서 작은 실천을 이어가기 좋은 챌린지입니다.",
    ChallengeCategory.WEIGHT.value: "체중 관리 관점에서 꾸준히 실천하기 좋은 챌린지입니다.",
    ChallengeCategory.EXERCISE.value: "활동량을 조금씩 늘리는 데 도움이 될 수 있는 챌린지입니다.",
    ChallengeCategory.MONITORING.value: "건강 상태를 꾸준히 기록하고 돌아보는 데 도움이 되는 챌린지입니다.",
}


def _get_daily_goal_count(challenge: Challenge | None) -> int:
    if challenge is None:
        return 1
    metric = str(getattr(challenge, "target_metric", None) or "").lower()
    if not any(token in metric for token in ("count", "times", "횟수", "회")):
        return 1
    matched = re.search(r"\d+", str(getattr(challenge, "target_value", None) or ""))
    if not matched:
        return 1
    return max(1, min(int(matched.group()), 10))


def _count_completed_days(log_dates: list[date], daily_goal_count: int) -> int:
    counts = Counter(log_dates)
    return sum(1 for count in counts.values() if count >= daily_goal_count)


def _is_completed_day(completed_count: int, total_count: int) -> bool:
    return total_count > 0 and completed_count >= total_count


def _get_duration_days(challenge: Challenge | None) -> int:
    duration_days = getattr(challenge, "duration_days", None) if challenge is not None else None
    if isinstance(duration_days, int) and duration_days > 0:
        return duration_days
    return DEFAULT_CHALLENGE_DURATION_DAYS


def _get_required_days(total_days: int) -> int:
    return max(1, ceil(total_days * CHALLENGE_COMPLETION_THRESHOLD))


def _is_rejoinable_user_challenge(user_challenge: UserChallenge) -> bool:
    return user_challenge.status == UserChallengeStatus.CANCELED or user_challenge.canceled_at is not None


def _now() -> datetime:
    return datetime.now(config.TIMEZONE)


def _to_kst_date(value: datetime | None) -> date | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.date()
    return value.astimezone(config.TIMEZONE).date()


def _kst_day_range(target_date: date) -> tuple[datetime, datetime]:
    started_at = datetime.combine(target_date, time.min, tzinfo=config.TIMEZONE)
    return started_at, started_at + timedelta(days=1)


def _default_expected_done_at(started_at: datetime, duration_days: int = DEFAULT_CHALLENGE_DURATION_DAYS) -> datetime:
    return started_at + timedelta(days=duration_days)


def _attach_user_challenge_dates(user_challenge: UserChallenge) -> None:
    user_challenge.started_date = _to_kst_date(user_challenge.started_at)
    user_challenge.expected_done_date = _to_kst_date(user_challenge.expected_done_at)
    user_challenge.completed_date = _to_kst_date(user_challenge.completed_at)
    user_challenge.is_completed = user_challenge.completed_at is not None


def _attach_challenge_log_dates(log: ChallengeLog) -> None:
    log.completed_date = _to_kst_date(log.completed_at)


def _status_value(value: object) -> str:
    return str(getattr(value, "value", value) or "")


def _is_active_challenge_object(challenge: object | None) -> bool:
    if challenge is None:
        return False
    return _status_value(getattr(challenge, "status", None)) == ChallengeStatus.ACTIVE.value


def _recommendation_points_to_active_challenge(recommendation: object) -> bool:
    challenge = getattr(recommendation, "challenge", None)
    return challenge is None or _is_active_challenge_object(challenge)


def _fallback_recommendation_reason(challenge: object) -> str:
    category = _status_value(getattr(challenge, "category", None))
    return RECOMMENDATION_FALLBACK_REASON_BY_CATEGORY.get(
        category,
        "건강관리 루틴을 가볍게 시작하는 데 도움이 될 수 있는 챌린지입니다.",
    )


def _dedupe_challenge_recommendations(recommendations: list[object]) -> list[object]:
    seen_challenge_ids: set[int] = set()
    deduped: list[object] = []
    for recommendation in recommendations:
        challenge_id = getattr(recommendation, "challenge_id", None)
        if challenge_id is None or challenge_id in seen_challenge_ids:
            continue
        seen_challenge_ids.add(challenge_id)
        deduped.append(recommendation)
    return deduped


def _synthetic_challenge_recommendation(
    *,
    user_id: int,
    challenge: object,
    index: int,
) -> SimpleNamespace:
    now = _now()
    return SimpleNamespace(
        id=-(index + 1),
        user_id=user_id,
        analysis_result_id=None,
        challenge_id=challenge.id,
        reason=_fallback_recommendation_reason(challenge),
        priority=0,
        is_selected=False,
        created_at=now,
        updated_at=now,
        challenge=challenge,
    )


async def _with_user_challenge_progress(user_challenge: UserChallenge | None) -> UserChallenge | None:
    if user_challenge is None:
        return None

    today = _now().date()
    challenge = await Challenge.get_or_none(id=user_challenge.challenge_id)
    daily_goal_count = _get_daily_goal_count(challenge)
    completed_log_dates = [
        completed_date
        for completed_date in [
            _to_kst_date(log.completed_at)
            for log in await ChallengeLog.filter(user_challenge_id=user_challenge.id, is_completed=True).only(
                "completed_at"
            )
        ]
        if completed_date is not None
    ]
    completed_days = _count_completed_days(completed_log_dates, daily_goal_count)
    today_completed_count = sum(1 for completed_date in completed_log_dates if completed_date == today)
    today_completed = today_completed_count >= daily_goal_count
    duration_days = _get_duration_days(challenge)
    required_days = _get_required_days(duration_days)
    expected_done_at = user_challenge.expected_done_at or _default_expected_done_at(
        user_challenge.started_at,
        duration_days,
    )
    completion_rate = round((completed_days / duration_days) * 100, 1) if duration_days > 0 else 0.0
    has_met_completion_condition = completed_days >= required_days
    is_finalized = bool(user_challenge.completed_at) or expected_done_at <= _now()
    is_final_completed = bool(user_challenge.completed_at) or (is_finalized and has_met_completion_condition)
    progress = max(0, min(round(completion_rate), 100))
    if is_final_completed:
        progress = 100

    _attach_user_challenge_dates(user_challenge)
    user_challenge.expected_done_at = expected_done_at
    user_challenge.expected_done_date = _to_kst_date(expected_done_at)
    user_challenge.end_date = _to_kst_date(expected_done_at)
    user_challenge.is_completed = is_final_completed
    user_challenge.completed_days = completed_days
    user_challenge.total_days = duration_days
    user_challenge.required_days = required_days
    user_challenge.completion_rate = completion_rate
    user_challenge.has_met_completion_condition = has_met_completion_condition
    user_challenge.is_finalized = is_finalized
    user_challenge.progress = progress
    user_challenge.today_completed = today_completed
    user_challenge.today_completed_count = today_completed_count
    user_challenge.daily_goal_count = daily_goal_count
    user_challenge.duration_days = duration_days
    return user_challenge


async def _with_user_challenges_progress(user_challenges: list[UserChallenge]) -> list[UserChallenge]:
    return [
        item
        for item in [await _with_user_challenge_progress(user_challenge) for user_challenge in user_challenges]
        if item
    ]


async def _sync_challenge_reminders_for_user(user_id: int) -> None:
    from app.services import notifications as notification_service

    await notification_service.sync_challenge_reminder_schedule_for_user(user_id)


async def list_active_challenges(
    *,
    category: str | None = None,
    challenge_type: str | None = None,
    target_disease: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Challenge]:
    return await challenge_repository.list_active_challenges_filtered(
        category=category,
        challenge_type=challenge_type,
        target_disease=target_disease,
        limit=limit,
        offset=offset,
    )


async def get_challenge(challenge_id: int) -> Challenge | None:
    return await challenge_repository.get_challenge_by_id(challenge_id)


async def create_challenge(request: ChallengeCreateRequest) -> Challenge:
    return await challenge_repository.create_challenge(request.model_dump())


async def join_challenge(
    user_id: int, challenge_id: int, request: UserChallengeCreateRequest | None = None
) -> UserChallenge:
    existing = await challenge_repository.get_user_challenge_by_user_and_challenge(user_id, challenge_id)
    data = request.model_dump(exclude={"challenge_id"}, exclude_none=True) if request is not None else {}
    data.setdefault("started_at", _now())
    data.setdefault("expected_done_at", _default_expected_done_at(data["started_at"]))
    if existing is not None:
        if _is_rejoinable_user_challenge(existing):
            rejoined = await _with_user_challenge_progress(
                await challenge_repository.update_user_challenge(
                    existing.id,
                    {
                        "status": UserChallengeStatus.JOINED,
                        "started_at": data["started_at"],
                        "expected_done_at": data["expected_done_at"],
                        "completed_at": None,
                        "canceled_at": None,
                    },
                )
            )
            await _sync_challenge_reminders_for_user(user_id)
            return rejoined
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="이미 참여한 챌린지입니다.")

    joined = await _with_user_challenge_progress(
        await challenge_repository.create_user_challenge(user_id, challenge_id, data)
    )
    await _sync_challenge_reminders_for_user(user_id)
    return joined


async def get_user_challenge(user_challenge_id: int) -> UserChallenge | None:
    return await _with_user_challenge_progress(await challenge_repository.get_user_challenge_by_id(user_challenge_id))


async def list_user_challenges(user_id: int, limit: int = 20, offset: int = 0) -> list[UserChallenge]:
    return await _with_user_challenges_progress(
        await challenge_repository.list_user_challenges(user_id, limit=limit, offset=offset)
    )


async def count_active_user_challenges(user_id: int) -> int:
    return await challenge_repository.count_user_challenges_by_status(user_id, [UserChallengeStatus.JOINED])


async def update_user_challenge(user_challenge_id: int, request: UserChallengeUpdateRequest) -> UserChallenge | None:
    data = request.model_dump(exclude_unset=True)
    return await _with_user_challenge_progress(
        await challenge_repository.update_user_challenge(user_challenge_id, data)
    )


async def create_challenge_log(user_challenge_id: int, request: ChallengeLogCreateRequest) -> ChallengeLog:
    user_challenge = await challenge_repository.get_user_challenge_by_id(user_challenge_id)
    if user_challenge is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="사용자 챌린지를 찾을 수 없습니다.")

    challenge = await Challenge.get_or_none(id=user_challenge.challenge_id)
    daily_goal_count = _get_daily_goal_count(challenge)
    existing_count = await challenge_repository.count_challenge_logs_by_date(user_challenge_id, request.log_date)
    if existing_count >= daily_goal_count:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="해당 날짜의 챌린지 로그가 하루 목표 횟수에 도달했습니다.",
        )

    data = request.model_dump(exclude_none=True)
    if request.is_completed and data.get("completed_at") is None:
        data["completed_at"] = _now()
    if not request.is_completed:
        data.pop("completed_at", None)
    log = await challenge_repository.create_challenge_log(user_challenge_id, data)
    _attach_challenge_log_dates(log)
    return log


async def list_challenge_logs(user_challenge_id: int) -> list[ChallengeLog]:
    logs = await challenge_repository.list_challenge_logs(user_challenge_id)
    for log in logs:
        _attach_challenge_log_dates(log)
    return logs


async def complete_today_challenge(user_challenge_id: int) -> ChallengeLog:
    now = _now()
    today = now.date()
    today_started_at, tomorrow_started_at = _kst_day_range(today)
    user_challenge = await challenge_repository.get_user_challenge_by_id(user_challenge_id)
    challenge = await Challenge.get_or_none(id=user_challenge.challenge_id) if user_challenge else None
    daily_goal_count = _get_daily_goal_count(challenge)
    completed_today = await ChallengeLog.filter(
        user_challenge_id=user_challenge_id,
        is_completed=True,
        completed_at__gte=today_started_at,
        completed_at__lt=tomorrow_started_at,
    ).count()
    if completed_today >= daily_goal_count:
        existing_log = (
            await ChallengeLog.filter(
                user_challenge_id=user_challenge_id,
                is_completed=True,
                completed_at__gte=today_started_at,
                completed_at__lt=tomorrow_started_at,
            )
            .order_by("-completed_at")
            .first()
        )
        if existing_log is not None:
            _attach_challenge_log_dates(existing_log)
            return existing_log

    log = await challenge_repository.create_challenge_log(
        user_challenge_id,
        {
            "log_date": today,
            "is_completed": True,
            "completed_at": now,
            "memo": f"오늘 챌린지 수행 {completed_today + 1}/{daily_goal_count}",
        },
    )
    _attach_challenge_log_dates(log)
    try:
        from app.services import service_jobs

        await service_jobs.enqueue_family_notification_create(
            alert_type="challenge_completed",
            user_challenge_id=user_challenge_id,
        )
    except Exception:  # noqa: BLE001
        # 가족 알림은 부가 기능이므로 챌린지 완료 기록 자체를 실패시키지 않는다.
        pass
    return log


async def get_challenge_calendar(user_id: int, target_date: date) -> dict:
    started_at, ended_before = _kst_day_range(target_date)
    started_challenges = await challenge_repository.list_user_challenges_started_between(
        user_id=user_id,
        started_at=started_at,
        ended_before=ended_before,
    )
    completed_logs = await challenge_repository.list_challenge_logs_completed_between(
        user_id=user_id,
        completed_at=started_at,
        ended_before=ended_before,
    )
    challenge_ids = {item.challenge_id for item in started_challenges}
    for log in completed_logs:
        challenge_ids.add(log.user_challenge.challenge_id)
    challenges = {item.id: item for item in await Challenge.filter(id__in=challenge_ids)} if challenge_ids else {}
    completed_logs_by_user_challenge_id: dict[int, list[ChallengeLog]] = {}
    for log in completed_logs:
        if _is_rejoinable_user_challenge(log.user_challenge):
            continue
        completed_logs_by_user_challenge_id.setdefault(log.user_challenge_id, []).append(log)

    items: list[dict] = []
    seen_user_challenge_ids: set[int] = set()
    total_count = 0
    completed_count = 0
    for user_challenge in started_challenges:
        if _is_rejoinable_user_challenge(user_challenge):
            continue
        _attach_user_challenge_dates(user_challenge)
        item_completed_logs = completed_logs_by_user_challenge_id.get(user_challenge.id, [])
        for completed_log in item_completed_logs:
            _attach_challenge_log_dates(completed_log)
        challenge = challenges.get(user_challenge.challenge_id)
        item_total_count = _get_daily_goal_count(challenge)
        item_completed_count = min(len(item_completed_logs), item_total_count)
        item_is_completed = _is_completed_day(item_completed_count, item_total_count)
        total_count += item_total_count
        completed_count += item_completed_count
        latest_completed_log = item_completed_logs[-1] if item_completed_logs else None
        if item_is_completed or user_challenge.completed_at is not None:
            status_label = "COMPLETED"
        elif user_challenge.canceled_at is not None:
            status_label = "CANCELED"
        elif user_challenge.expected_done_at is not None and user_challenge.expected_done_at < _now():
            status_label = "EXPIRED"
        else:
            status_label = "IN_PROGRESS"
        items.append(
            {
                "challenge_id": user_challenge.challenge_id,
                "user_challenge_id": user_challenge.id,
                "challenge_log_id": latest_completed_log.id if latest_completed_log else None,
                "title": challenge.title if challenge else None,
                "status": status_label,
                "total_count": item_total_count,
                "completed_count": item_completed_count,
                "is_completed": item_is_completed,
                "started_at": user_challenge.started_at,
                "expected_done_at": user_challenge.expected_done_at,
                "due_at": user_challenge.expected_done_at,
                "completed_at": latest_completed_log.completed_at
                if latest_completed_log
                else user_challenge.completed_at,
                "started_date": user_challenge.started_date,
                "expected_done_date": user_challenge.expected_done_date,
                "due_date": user_challenge.expected_done_date,
                "completed_date": latest_completed_log.completed_date
                if latest_completed_log
                else user_challenge.completed_date,
            }
        )
        seen_user_challenge_ids.add(user_challenge.id)

    for user_challenge_id, item_completed_logs in completed_logs_by_user_challenge_id.items():
        for log in item_completed_logs:
            _attach_challenge_log_dates(log)
        latest_completed_log = item_completed_logs[-1]
        user_challenge = latest_completed_log.user_challenge
        if _is_rejoinable_user_challenge(user_challenge):
            continue
        _attach_user_challenge_dates(user_challenge)
        challenge = challenges.get(user_challenge.challenge_id)
        if user_challenge.id in seen_user_challenge_ids:
            continue
        item_total_count = _get_daily_goal_count(challenge)
        item_completed_count = min(len(item_completed_logs), item_total_count)
        item_is_completed = _is_completed_day(item_completed_count, item_total_count)
        total_count += item_total_count
        completed_count += item_completed_count
        items.append(
            {
                "challenge_id": user_challenge.challenge_id,
                "user_challenge_id": user_challenge.id,
                "challenge_log_id": latest_completed_log.id,
                "title": challenge.title if challenge else None,
                "status": "COMPLETED" if item_is_completed else "IN_PROGRESS",
                "total_count": item_total_count,
                "completed_count": item_completed_count,
                "is_completed": item_is_completed,
                "started_at": user_challenge.started_at,
                "expected_done_at": user_challenge.expected_done_at,
                "due_at": user_challenge.expected_done_at,
                "completed_at": latest_completed_log.completed_at,
                "started_date": user_challenge.started_date,
                "expected_done_date": user_challenge.expected_done_date,
                "due_date": user_challenge.expected_done_date,
                "completed_date": latest_completed_log.completed_date,
            }
        )

    items.sort(key=lambda item: (item["completed_at"] or item["started_at"] or started_at, item["user_challenge_id"]))
    return {
        "date": target_date,
        "total_count": total_count,
        "completed_count": completed_count,
        "is_completed": _is_completed_day(completed_count, total_count),
        "items": items,
    }


async def give_up_challenge(user_challenge_id: int) -> UserChallenge | None:
    updated = await _with_user_challenge_progress(
        await challenge_repository.update_user_challenge(
            user_challenge_id,
            {
                "status": UserChallengeStatus.CANCELED,
                "canceled_at": _now(),
            },
        )
    )
    if updated is not None:
        await _sync_challenge_reminders_for_user(int(updated.user_id))
    return updated


async def create_challenge_recommendation(
    user_id: int,
    analysis_result_id: int,
    challenge_id: int,
    request: ChallengeRecommendationCreateRequest,
) -> ChallengeRecommendation:
    challenge = await challenge_repository.get_challenge_by_id(challenge_id)
    if not _is_active_challenge_object(challenge):
        fallback_challenges = await list_active_challenges(limit=1)
        if not fallback_challenges:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="추천 가능한 active 챌린지가 없습니다.")
        challenge = fallback_challenges[0]
        challenge_id = challenge.id
    data = request.model_dump(exclude={"analysis_result_id", "challenge_id"})
    if not data.get("reason"):
        data["reason"] = _fallback_recommendation_reason(challenge)
    return await challenge_repository.create_challenge_recommendation(
        user_id=user_id,
        analysis_result_id=analysis_result_id,
        challenge_id=challenge_id,
        data=data,
    )


async def list_challenge_recommendations(
    user_id: int, limit: int = 20, offset: int = 0
) -> list[ChallengeRecommendation]:
    if limit <= 0:
        return []

    fetch_limit = max(limit * 3, limit + 20)
    recommendations = await challenge_repository.list_challenge_recommendations(
        user_id,
        limit=fetch_limit,
        offset=offset,
    )
    active_recommendations = [
        recommendation
        for recommendation in recommendations
        if _recommendation_points_to_active_challenge(recommendation)
    ]
    deduped = _dedupe_challenge_recommendations(active_recommendations)

    if len(deduped) < limit and offset == 0:
        existing_ids = {item.challenge_id for item in deduped}
        active_challenges = await list_active_challenges(limit=100)
        fallback_challenges = [
            challenge
            for challenge in active_challenges
            if challenge.id not in existing_ids and _is_active_challenge_object(challenge)
        ]
        needed = limit - len(deduped)
        deduped.extend(
            _synthetic_challenge_recommendation(user_id=user_id, challenge=challenge, index=index)
            for index, challenge in enumerate(fallback_challenges[:needed])
        )

    return deduped[:limit]
