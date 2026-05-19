from typing import Annotated, Any

from fastapi import APIRouter, Depends, status

from app.dependencies.security import get_request_user
from app.dtos.challenges import (
    ChallengeCreateRequest,
    ChallengeLogCreateRequest,
    ChallengeLogResponse,
    ChallengeRecommendationCreateRequest,
    ChallengeRecommendationResponse,
    ChallengeResponse,
    UserChallengeCreateRequest,
    UserChallengeResponse,
)
from app.models.users import User
from app.services import challenges as challenge_service

challenge_router = APIRouter(prefix="/challenges", tags=["challenges"])


@challenge_router.get("", response_model=list[ChallengeResponse])
async def list_active_challenges(limit: int = 50, offset: int = 0):
    return await challenge_service.list_active_challenges(limit=limit, offset=offset)


@challenge_router.post("", response_model=ChallengeResponse, status_code=status.HTTP_201_CREATED)
async def create_challenge(request: ChallengeCreateRequest):
    return await challenge_service.create_challenge(request)


@challenge_router.get("/my", response_model=list[UserChallengeResponse])
async def list_user_challenges(user: Annotated[User, Depends(get_request_user)], limit: int = 20, offset: int = 0):
    return await challenge_service.list_user_challenges(user.id, limit=limit, offset=offset)


@challenge_router.patch("/my/{user_challenge_id}", response_model=UserChallengeResponse | None)
async def update_user_challenge(user_challenge_id: int, data: dict[str, Any]):
    return await challenge_service.update_user_challenge(user_challenge_id, data)


@challenge_router.post(
    "/my/{user_challenge_id}/logs", response_model=ChallengeLogResponse, status_code=status.HTTP_201_CREATED
)
async def create_challenge_log(user_challenge_id: int, request: ChallengeLogCreateRequest):
    return await challenge_service.create_challenge_log(user_challenge_id, request)


@challenge_router.get("/my/{user_challenge_id}/logs", response_model=list[ChallengeLogResponse])
async def list_challenge_logs(user_challenge_id: int):
    return await challenge_service.list_challenge_logs(user_challenge_id)


@challenge_router.post(
    "/recommendations", response_model=ChallengeRecommendationResponse, status_code=status.HTTP_201_CREATED
)
async def create_challenge_recommendation(
    request: ChallengeRecommendationCreateRequest, user: Annotated[User, Depends(get_request_user)]
):
    return await challenge_service.create_challenge_recommendation(
        user_id=user.id,
        analysis_result_id=request.analysis_result_id,
        challenge_id=request.challenge_id,
        request=request,
    )


@challenge_router.get("/recommendations", response_model=list[ChallengeRecommendationResponse])
async def list_challenge_recommendations(
    user: Annotated[User, Depends(get_request_user)], limit: int = 20, offset: int = 0
):
    return await challenge_service.list_challenge_recommendations(user.id, limit=limit, offset=offset)


@challenge_router.get("/{challenge_id}", response_model=ChallengeResponse | None)
async def get_challenge(challenge_id: int):
    return await challenge_service.get_challenge(challenge_id)


@challenge_router.post(
    "/{challenge_id}/join", response_model=UserChallengeResponse, status_code=status.HTTP_201_CREATED
)
async def join_challenge(
    challenge_id: int,
    user: Annotated[User, Depends(get_request_user)],
    request: UserChallengeCreateRequest | None = None,
):
    return await challenge_service.join_challenge(user.id, challenge_id, request)
