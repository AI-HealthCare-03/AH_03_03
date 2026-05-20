from datetime import datetime
from typing import Any

from app.core import config
from app.dtos.challenges import (
    ChallengeCreateRequest,
    ChallengeLogCreateRequest,
    ChallengeRecommendationCreateRequest,
    UserChallengeCreateRequest,
)
from app.models.challenges import Challenge, ChallengeLog, ChallengeRecommendation, UserChallenge, UserChallengeStatus
from app.repositories import challenge_repository


async def list_active_challenges(limit: int = 50, offset: int = 0) -> list[Challenge]:
    return await challenge_repository.list_active_challenges(limit=limit, offset=offset)


async def get_challenge(challenge_id: int) -> Challenge | None:
    return await challenge_repository.get_challenge_by_id(challenge_id)


async def create_challenge(request: ChallengeCreateRequest) -> Challenge:
    return await challenge_repository.create_challenge(request.model_dump())


async def join_challenge(
    user_id: int, challenge_id: int, request: UserChallengeCreateRequest | None = None
) -> UserChallenge:
    data = request.model_dump(exclude={"challenge_id"}) if request is not None else None
    return await challenge_repository.create_user_challenge(user_id, challenge_id, data)


async def get_user_challenge(user_challenge_id: int) -> UserChallenge | None:
    return await challenge_repository.get_user_challenge_by_id(user_challenge_id)


async def list_user_challenges(user_id: int, limit: int = 20, offset: int = 0) -> list[UserChallenge]:
    return await challenge_repository.list_user_challenges(user_id, limit=limit, offset=offset)


async def update_user_challenge(user_challenge_id: int, data: dict[str, Any]) -> UserChallenge | None:
    return await challenge_repository.update_user_challenge(user_challenge_id, data)


async def create_challenge_log(user_challenge_id: int, request: ChallengeLogCreateRequest) -> ChallengeLog:
    return await challenge_repository.create_challenge_log(user_challenge_id, request.model_dump())


async def list_challenge_logs(user_challenge_id: int) -> list[ChallengeLog]:
    return await challenge_repository.list_challenge_logs(user_challenge_id)


async def complete_today_challenge(user_challenge_id: int) -> ChallengeLog:
    today = datetime.now(config.TIMEZONE).date()
    existing_log = await challenge_repository.get_challenge_log_by_date(user_challenge_id, today)
    if existing_log is not None:
        updated = await challenge_repository.update_challenge_log(
            existing_log.id,
            {"is_completed": True, "memo": existing_log.memo or "오늘 챌린지 완료"},
        )
        return updated or existing_log

    return await challenge_repository.create_challenge_log(
        user_challenge_id,
        {
            "log_date": today,
            "is_completed": True,
            "memo": "오늘 챌린지 완료",
        },
    )


async def give_up_challenge(user_challenge_id: int) -> UserChallenge | None:
    return await challenge_repository.update_user_challenge(
        user_challenge_id,
        {
            "status": UserChallengeStatus.CANCELED,
            "canceled_at": datetime.now(config.TIMEZONE),
        },
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
