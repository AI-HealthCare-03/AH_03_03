import re
from collections import Counter
from datetime import date, datetime, time, timedelta

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
from app.models.challenges import Challenge, ChallengeLog, ChallengeRecommendation, UserChallenge, UserChallengeStatus
from app.repositories import challenge_repository


def _get_daily_goal_count(challenge: Challenge | None) -> int:
    if challenge is None:
        return 1
    metric = str(challenge.target_metric or "").lower()
    if not any(token in metric for token in ("count", "times", "횟수", "회")):
        return 1
    matched = re.search(r"\d+", str(challenge.target_value or ""))
    if not matched:
        return 1
    return max(1, min(int(matched.group()), 10))


def _count_completed_days(log_dates: list[date], daily_goal_count: int) -> int:
    counts = Counter(log_dates)
    return sum(1 for count in counts.values() if count >= daily_goal_count)


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


def _default_expected_done_at(started_at: datetime) -> datetime:
    return started_at + timedelta(days=1)


def _attach_user_challenge_dates(user_challenge: UserChallenge) -> None:
    user_challenge.started_date = _to_kst_date(user_challenge.started_at)
    user_challenge.expected_done_date = _to_kst_date(user_challenge.expected_done_at)
    user_challenge.completed_date = _to_kst_date(user_challenge.completed_at)
    user_challenge.is_completed = user_challenge.completed_at is not None


def _attach_challenge_log_dates(log: ChallengeLog) -> None:
    log.completed_date = _to_kst_date(log.completed_at)


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
    duration_days = challenge.duration_days if challenge is not None else None
    progress = 0
    if duration_days and duration_days > 0:
        progress = max(0, min(round((completed_days / duration_days) * 100), 100))
    if user_challenge.completed_at is not None:
        progress = 100

    _attach_user_challenge_dates(user_challenge)
    user_challenge.completed_days = completed_days
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
    if existing is not None:
        # Rejoining a canceled challenge needs a product policy and DB constraint pass.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="이미 참여한 챌린지입니다.")

    data = request.model_dump(exclude={"challenge_id"}, exclude_none=True) if request is not None else {}
    data.setdefault("started_at", _now())
    data.setdefault("expected_done_at", _default_expected_done_at(data["started_at"]))
    return await _with_user_challenge_progress(
        await challenge_repository.create_user_challenge(user_id, challenge_id, data)
    )


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
    completed_log_by_user_challenge_id = {log.user_challenge_id: log for log in completed_logs}

    items: list[dict] = []
    seen_user_challenge_ids: set[int] = set()
    for user_challenge in started_challenges:
        _attach_user_challenge_dates(user_challenge)
        completed_log = completed_log_by_user_challenge_id.get(user_challenge.id)
        if completed_log is not None:
            _attach_challenge_log_dates(completed_log)
        challenge = challenges.get(user_challenge.challenge_id)
        if completed_log is not None or user_challenge.completed_at is not None:
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
                "challenge_log_id": completed_log.id if completed_log else None,
                "title": challenge.title if challenge else None,
                "status": status_label,
                "started_at": user_challenge.started_at,
                "expected_done_at": user_challenge.expected_done_at,
                "due_at": user_challenge.expected_done_at,
                "completed_at": completed_log.completed_at if completed_log else user_challenge.completed_at,
                "started_date": user_challenge.started_date,
                "expected_done_date": user_challenge.expected_done_date,
                "due_date": user_challenge.expected_done_date,
                "completed_date": completed_log.completed_date if completed_log else user_challenge.completed_date,
            }
        )
        seen_user_challenge_ids.add(user_challenge.id)

    for log in completed_logs:
        _attach_challenge_log_dates(log)
        user_challenge = log.user_challenge
        _attach_user_challenge_dates(user_challenge)
        challenge = challenges.get(user_challenge.challenge_id)
        if user_challenge.id in seen_user_challenge_ids:
            continue
        items.append(
            {
                "challenge_id": user_challenge.challenge_id,
                "user_challenge_id": user_challenge.id,
                "challenge_log_id": log.id,
                "title": challenge.title if challenge else None,
                "status": "COMPLETED",
                "started_at": user_challenge.started_at,
                "expected_done_at": user_challenge.expected_done_at,
                "due_at": user_challenge.expected_done_at,
                "completed_at": log.completed_at,
                "started_date": user_challenge.started_date,
                "expected_done_date": user_challenge.expected_done_date,
                "due_date": user_challenge.expected_done_date,
                "completed_date": log.completed_date,
            }
        )

    items.sort(key=lambda item: (item["completed_at"] or item["started_at"] or started_at, item["user_challenge_id"]))
    return {"date": target_date, "items": items}


async def give_up_challenge(user_challenge_id: int) -> UserChallenge | None:
    return await _with_user_challenge_progress(
        await challenge_repository.update_user_challenge(
            user_challenge_id,
            {
                "status": UserChallengeStatus.CANCELED,
                "canceled_at": _now(),
            },
        )
    )


async def create_challenge_recommendation(
    user_id: int,
    analysis_result_id: int,
    challenge_id: int,
    request: ChallengeRecommendationCreateRequest,
) -> ChallengeRecommendation:
    data = request.model_dump(exclude={"analysis_result_id", "challenge_id"})
    return await challenge_repository.create_challenge_recommendation(
        user_id=user_id,
        analysis_result_id=analysis_result_id,
        challenge_id=challenge_id,
        data=data,
    )


async def list_challenge_recommendations(
    user_id: int, limit: int = 20, offset: int = 0
) -> list[ChallengeRecommendation]:
    return await challenge_repository.list_challenge_recommendations(user_id, limit=limit, offset=offset)
