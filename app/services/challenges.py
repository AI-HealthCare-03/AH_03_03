import re
from collections import Counter
from datetime import date, datetime

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


async def _with_user_challenge_progress(user_challenge: UserChallenge | None) -> UserChallenge | None:
    if user_challenge is None:
        return None

    today = datetime.now(config.TIMEZONE).date()
    challenge = await Challenge.get_or_none(id=user_challenge.challenge_id)
    daily_goal_count = _get_daily_goal_count(challenge)
    completed_log_dates = [
        log.log_date
        for log in await ChallengeLog.filter(user_challenge_id=user_challenge.id, is_completed=True).only("log_date")
    ]
    completed_days = _count_completed_days(completed_log_dates, daily_goal_count)
    today_completed_count = sum(1 for log_date in completed_log_dates if log_date == today)
    today_completed = today_completed_count >= daily_goal_count
    duration_days = challenge.duration_days if challenge is not None else None
    progress = 0
    if duration_days and duration_days > 0:
        progress = max(0, min(round((completed_days / duration_days) * 100), 100))
    if user_challenge.status == UserChallengeStatus.COMPLETED:
        progress = 100

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

    data = request.model_dump(exclude={"challenge_id"}) if request is not None else {}
    data.setdefault("started_at", datetime.now(config.TIMEZONE))
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
    return await challenge_repository.create_challenge_log(user_challenge_id, request.model_dump())


async def list_challenge_logs(user_challenge_id: int) -> list[ChallengeLog]:
    return await challenge_repository.list_challenge_logs(user_challenge_id)


async def complete_today_challenge(user_challenge_id: int) -> ChallengeLog:
    today = datetime.now(config.TIMEZONE).date()
    user_challenge = await challenge_repository.get_user_challenge_by_id(user_challenge_id)
    challenge = await Challenge.get_or_none(id=user_challenge.challenge_id) if user_challenge else None
    daily_goal_count = _get_daily_goal_count(challenge)
    completed_today = await ChallengeLog.filter(
        user_challenge_id=user_challenge_id,
        log_date=today,
        is_completed=True,
    ).count()
    if completed_today >= daily_goal_count:
        existing_log = await challenge_repository.get_challenge_log_by_date(user_challenge_id, today)
        if existing_log is not None:
            return existing_log

    return await challenge_repository.create_challenge_log(
        user_challenge_id,
        {
            "log_date": today,
            "is_completed": True,
            "memo": f"오늘 챌린지 수행 {completed_today + 1}/{daily_goal_count}",
        },
    )


async def give_up_challenge(user_challenge_id: int) -> UserChallenge | None:
    return await _with_user_challenge_progress(
        await challenge_repository.update_user_challenge(
            user_challenge_id,
            {
                "status": UserChallengeStatus.CANCELED,
                "canceled_at": datetime.now(config.TIMEZONE),
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
