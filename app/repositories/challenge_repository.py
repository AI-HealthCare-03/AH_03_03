from datetime import date
from typing import Any

from app.models.challenges import Challenge, ChallengeLog, ChallengeRecommendation, ChallengeStatus, UserChallenge


async def list_active_challenges(limit: int = 50, offset: int = 0) -> list[Challenge]:
    return await Challenge.filter(status=ChallengeStatus.ACTIVE).order_by("-created_at").offset(offset).limit(limit)


async def get_challenge_by_id(challenge_id: int) -> Challenge | None:
    return await Challenge.get_or_none(id=challenge_id)


async def create_challenge(data: dict[str, Any]) -> Challenge:
    return await Challenge.create(**data)


async def create_user_challenge(user_id: int, challenge_id: int, data: dict[str, Any] | None = None) -> UserChallenge:
    payload = data or {}
    return await UserChallenge.create(user_id=user_id, challenge_id=challenge_id, **payload)


async def get_user_challenge_by_id(user_challenge_id: int) -> UserChallenge | None:
    return await UserChallenge.get_or_none(id=user_challenge_id)


async def list_user_challenges(user_id: int, limit: int = 20, offset: int = 0) -> list[UserChallenge]:
    return await UserChallenge.filter(user_id=user_id).order_by("-created_at").offset(offset).limit(limit)


async def update_user_challenge(user_challenge_id: int, data: dict[str, Any]) -> UserChallenge | None:
    user_challenge = await get_user_challenge_by_id(user_challenge_id)
    if user_challenge is None:
        return None

    for key, value in data.items():
        setattr(user_challenge, key, value)
    await user_challenge.save(update_fields=list(data.keys()) if data else None)
    return user_challenge


async def create_challenge_log(user_challenge_id: int, data: dict[str, Any]) -> ChallengeLog:
    return await ChallengeLog.create(user_challenge_id=user_challenge_id, **data)


async def list_challenge_logs(user_challenge_id: int) -> list[ChallengeLog]:
    return await ChallengeLog.filter(user_challenge_id=user_challenge_id).order_by("-log_date")


async def get_challenge_log_by_date(user_challenge_id: int, log_date: date) -> ChallengeLog | None:
    return await ChallengeLog.get_or_none(user_challenge_id=user_challenge_id, log_date=log_date)


async def update_challenge_log(challenge_log_id: int, data: dict[str, Any]) -> ChallengeLog | None:
    challenge_log = await ChallengeLog.get_or_none(id=challenge_log_id)
    if challenge_log is None:
        return None

    for key, value in data.items():
        setattr(challenge_log, key, value)
    await challenge_log.save(update_fields=list(data.keys()) if data else None)
    return challenge_log


async def create_challenge_recommendation(
    user_id: int, analysis_result_id: int, challenge_id: int, data: dict[str, Any]
) -> ChallengeRecommendation:
    return await ChallengeRecommendation.create(
        user_id=user_id,
        analysis_result_id=analysis_result_id,
        challenge_id=challenge_id,
        **data,
    )


async def list_challenge_recommendations(
    user_id: int, limit: int = 20, offset: int = 0
) -> list[ChallengeRecommendation]:
    return await ChallengeRecommendation.filter(user_id=user_id).order_by("-created_at").offset(offset).limit(limit)
