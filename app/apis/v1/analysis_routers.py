from typing import Annotated

from fastapi import APIRouter, Depends, status

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


@analysis_router.get("/results/{result_id}", response_model=AnalysisResultResponse | None)
async def get_analysis_result(result_id: int):
    return await analysis_service.get_analysis_result(result_id)


@analysis_router.get("/results/{result_id}/detail")
async def get_analysis_result_detail(result_id: int):
    return await analysis_service.get_analysis_result_detail(result_id)


@analysis_router.post(
    "/results/{result_id}/factors",
    response_model=AnalysisResultFactorResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_analysis_factor(result_id: int, request: AnalysisResultFactorCreateRequest):
    return await analysis_service.create_analysis_factor(result_id, request)


@analysis_router.get("/results/{result_id}/factors", response_model=list[AnalysisResultFactorResponse])
async def list_analysis_factors(result_id: int):
    return await analysis_service.list_analysis_factors(result_id)


@analysis_router.post("/snapshots", response_model=AnalysisSnapshotResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis_snapshot(analysis_result_id: int, request: AnalysisSnapshotCreateRequest):
    return await analysis_service.create_analysis_snapshot(analysis_result_id, request)


@analysis_router.get("/snapshots/{snapshot_id}", response_model=AnalysisSnapshotResponse | None)
async def get_analysis_snapshot(snapshot_id: int):
    return await analysis_service.get_analysis_snapshot_by_id(snapshot_id)
