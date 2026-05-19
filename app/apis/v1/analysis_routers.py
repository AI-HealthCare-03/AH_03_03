from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.apis.v1.dependencies import ensure_found, ensure_owner
from app.dependencies.security import get_request_user
from app.dtos.analysis import (
    AnalysisResultCreateRequest,
    AnalysisResultFactorCreateRequest,
    AnalysisResultFactorResponse,
    AnalysisResultResponse,
    AnalysisSnapshotCreateRequest,
    AnalysisSnapshotResponse,
)
from app.models.users import User
from app.services import analysis as analysis_service

analysis_router = APIRouter(prefix="/analysis", tags=["analysis"])


@analysis_router.post("/results", response_model=AnalysisResultResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis_result(
    request: AnalysisResultCreateRequest, user: Annotated[User, Depends(get_request_user)]
):
    return await analysis_service.create_analysis_result(user.id, request)


@analysis_router.get("/results", response_model=list[AnalysisResultResponse])
async def list_analysis_results(user: Annotated[User, Depends(get_request_user)], limit: int = 20, offset: int = 0):
    return await analysis_service.list_analysis_results(user.id, limit=limit, offset=offset)


@analysis_router.get("/results/{result_id}", response_model=AnalysisResultResponse)
async def get_analysis_result(result_id: int, user: Annotated[User, Depends(get_request_user)]):
    result = ensure_found(await analysis_service.get_analysis_result(result_id), "분석 결과를 찾을 수 없습니다.")
    ensure_owner(result.user_id, user)
    return result


@analysis_router.get("/results/{result_id}/detail")
async def get_analysis_result_detail(result_id: int, user: Annotated[User, Depends(get_request_user)]):
    result = ensure_found(await analysis_service.get_analysis_result(result_id), "분석 결과를 찾을 수 없습니다.")
    ensure_owner(result.user_id, user)
    detail = await analysis_service.get_analysis_result_detail(result_id)
    return ensure_found(detail, "분석 결과를 찾을 수 없습니다.")


@analysis_router.post(
    "/results/{result_id}/factors",
    response_model=AnalysisResultFactorResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_analysis_factor(
    result_id: int,
    request: AnalysisResultFactorCreateRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    result = ensure_found(await analysis_service.get_analysis_result(result_id), "분석 결과를 찾을 수 없습니다.")
    ensure_owner(result.user_id, user)
    return await analysis_service.create_analysis_factor(result_id, request)


@analysis_router.get("/results/{result_id}/factors", response_model=list[AnalysisResultFactorResponse])
async def list_analysis_factors(result_id: int, user: Annotated[User, Depends(get_request_user)]):
    result = ensure_found(await analysis_service.get_analysis_result(result_id), "분석 결과를 찾을 수 없습니다.")
    ensure_owner(result.user_id, user)
    return await analysis_service.list_analysis_factors(result_id)


@analysis_router.post("/snapshots", response_model=AnalysisSnapshotResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis_snapshot(
    analysis_result_id: int,
    request: AnalysisSnapshotCreateRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    result = ensure_found(
        await analysis_service.get_analysis_result(analysis_result_id), "분석 결과를 찾을 수 없습니다."
    )
    ensure_owner(result.user_id, user)
    return await analysis_service.create_analysis_snapshot(analysis_result_id, request)


@analysis_router.get("/snapshots/{snapshot_id}", response_model=AnalysisSnapshotResponse)
async def get_analysis_snapshot(snapshot_id: int, user: Annotated[User, Depends(get_request_user)]):
    snapshot = ensure_found(
        await analysis_service.get_analysis_snapshot_by_id(snapshot_id), "분석 스냅샷을 찾을 수 없습니다."
    )
    await snapshot.fetch_related("analysis_result")
    ensure_owner(snapshot.analysis_result.user_id, user)
    return snapshot
