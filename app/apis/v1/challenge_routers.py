from typing import Annotated, Any

from fastapi import APIRouter, Depends, status

from app.apis.v1.dependencies import ensure_admin_user, ensure_found, ensure_owner
from app.dependencies.security import get_request_user
from app.dtos.challenges import (
    ChallengeActionResponse,
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
from app.services import analysis as analysis_service
from app.services import challenges as challenge_service

challenge_router = APIRouter(prefix="/challenges", tags=["challenges"])


@challenge_router.get("", response_model=list[ChallengeResponse])
async def list_active_challenges(limit: int = 50, offset: int = 0):
    return await challenge_service.list_active_challenges(limit=limit, offset=offset)


@challenge_router.post("", response_model=ChallengeResponse, status_code=status.HTTP_201_CREATED)
async def create_challenge(request: ChallengeCreateRequest, user: Annotated[User, Depends(get_request_user)]):
    ensure_admin_user(user)
    return await challenge_service.create_challenge(request)


@challenge_router.get("/my", response_model=list[UserChallengeResponse])
async def list_user_challenges(user: Annotated[User, Depends(get_request_user)], limit: int = 20, offset: int = 0):
    return await challenge_service.list_user_challenges(user.id, limit=limit, offset=offset)


@challenge_router.patch("/my/{user_challenge_id}", response_model=UserChallengeResponse | None)
async def update_user_challenge(
    user_challenge_id: int,
    data: dict[str, Any],
    user: Annotated[User, Depends(get_request_user)],
):
    user_challenge = ensure_found(
        await challenge_service.get_user_challenge(user_challenge_id),
        "사용자 챌린지를 찾을 수 없습니다.",
    )
    ensure_owner(user_challenge.user_id, user)
    updated = await challenge_service.update_user_challenge(user_challenge_id, data)
    return ensure_found(updated, "사용자 챌린지를 찾을 수 없습니다.")


@challenge_router.post(
    "/my/{user_challenge_id}/logs", response_model=ChallengeLogResponse, status_code=status.HTTP_201_CREATED
)
async def create_challenge_log(
    user_challenge_id: int,
    request: ChallengeLogCreateRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    user_challenge = ensure_found(
        await challenge_service.get_user_challenge(user_challenge_id),
        "사용자 챌린지를 찾을 수 없습니다.",
    )
    ensure_owner(user_challenge.user_id, user)
    return await challenge_service.create_challenge_log(user_challenge_id, request)


@challenge_router.get("/my/{user_challenge_id}/logs", response_model=list[ChallengeLogResponse])
async def list_challenge_logs(user_challenge_id: int, user: Annotated[User, Depends(get_request_user)]):
    user_challenge = ensure_found(
        await challenge_service.get_user_challenge(user_challenge_id),
        "사용자 챌린지를 찾을 수 없습니다.",
    )
    ensure_owner(user_challenge.user_id, user)
    return await challenge_service.list_challenge_logs(user_challenge_id)


@challenge_router.post("/my/{user_challenge_id}/complete-today", response_model=ChallengeActionResponse)
async def complete_today_challenge(user_challenge_id: int, user: Annotated[User, Depends(get_request_user)]):
    user_challenge = ensure_found(
        await challenge_service.get_user_challenge(user_challenge_id),
        "사용자 챌린지를 찾을 수 없습니다.",
    )
    ensure_owner(user_challenge.user_id, user)
    result = await challenge_service.complete_today_challenge(user_challenge_id)
    return {"message": "오늘 챌린지를 완료 처리했습니다.", "result": result}


@challenge_router.patch("/my/{user_challenge_id}/give-up", response_model=ChallengeActionResponse)
async def give_up_challenge(user_challenge_id: int, user: Annotated[User, Depends(get_request_user)]):
    user_challenge = ensure_found(
        await challenge_service.get_user_challenge(user_challenge_id),
        "사용자 챌린지를 찾을 수 없습니다.",
    )
    ensure_owner(user_challenge.user_id, user)
    result = ensure_found(
        await challenge_service.give_up_challenge(user_challenge_id),
        "사용자 챌린지를 찾을 수 없습니다.",
    )
    return {"message": "챌린지를 포기 처리했습니다.", "result": result}


@challenge_router.post(
    "/recommendations", response_model=ChallengeRecommendationResponse, status_code=status.HTTP_201_CREATED
)
async def create_challenge_recommendation(
    request: ChallengeRecommendationCreateRequest, user: Annotated[User, Depends(get_request_user)]
):
    result = ensure_found(
        await analysis_service.get_analysis_result(request.analysis_result_id),
        "분석 결과를 찾을 수 없습니다.",
    )
    ensure_owner(result.user_id, user)
    ensure_found(await challenge_service.get_challenge(request.challenge_id), "챌린지를 찾을 수 없습니다.")
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


@challenge_router.get("/{challenge_id}", response_model=ChallengeResponse)
async def get_challenge(challenge_id: int):
    return ensure_found(await challenge_service.get_challenge(challenge_id), "챌린지를 찾을 수 없습니다.")


@challenge_router.post(
    "/{challenge_id}/join", response_model=UserChallengeResponse, status_code=status.HTTP_201_CREATED
)
async def join_challenge(
    challenge_id: int,
    user: Annotated[User, Depends(get_request_user)],
    request: UserChallengeCreateRequest | None = None,
):
    ensure_found(await challenge_service.get_challenge(challenge_id), "챌린지를 찾을 수 없습니다.")
    return await challenge_service.join_challenge(user.id, challenge_id, request)
