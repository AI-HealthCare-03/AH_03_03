from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from app.apis.v1.dependencies import ensure_found, ensure_owner, get_request_user
from app.dtos.analysis import (
    AnalysisResultCreateRequest,
    AnalysisResultFactorCreateRequest,
    AnalysisResultFactorResponse,
    AnalysisResultResponse,
    AnalysisSnapshotCreateRequest,
    AnalysisSnapshotResponse,
    DummyAnalysisResultResponse,
    DummyAnalysisRunRequest,
)
from app.models.users import User
from app.services import analysis as analysis_service
from app.services import health as health_service
from app.services.sensitive_access_logs import safe_record_sensitive_access

analysis_router = APIRouter(prefix="/analysis", tags=["analysis"])


@analysis_router.post(
    "/dummy-run",
    response_model=list[DummyAnalysisResultResponse],
    status_code=status.HTTP_201_CREATED,
)
async def run_dummy_analysis(
    request: DummyAnalysisRunRequest,
    user: Annotated[User, Depends(get_request_user)],
):
    health_record = ensure_found(
        await health_service.get_health_record(request.health_record_id),
        "건강 기록을 찾을 수 없습니다.",
    )
    ensure_owner(health_record.user_id, user)
    return await analysis_service.run_dummy_analysis(user.id, health_record)


@analysis_router.post("/results", response_model=AnalysisResultResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis_result(
    request: AnalysisResultCreateRequest, user: Annotated[User, Depends(get_request_user)]
):
    return await analysis_service.create_analysis_result(user.id, request)


@analysis_router.get("/results", response_model=list[AnalysisResultResponse])
async def list_analysis_results(
    request: Request,
    user: Annotated[User, Depends(get_request_user)],
    limit: int = 20,
    offset: int = 0,
):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="ANALYSIS_RESULT",
        access_reason="analysis_results.list",
    )
    return await analysis_service.list_analysis_results(user.id, limit=limit, offset=offset)


@analysis_router.get("/results/latest", response_model=list[AnalysisResultResponse])
async def list_latest_analysis_results(request: Request, user: Annotated[User, Depends(get_request_user)]):
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=user.id,
        resource_type="ANALYSIS_RESULT",
        access_reason="analysis_results.latest",
    )
    return await analysis_service.list_latest_analysis_results(user.id)


@analysis_router.get("/results/{result_id}", response_model=AnalysisResultResponse)
async def get_analysis_result(result_id: int, request: Request, user: Annotated[User, Depends(get_request_user)]):
    result = ensure_found(await analysis_service.get_analysis_result(result_id), "분석 결과를 찾을 수 없습니다.")
    ensure_owner(result.user_id, user)
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=result.user_id,
        resource_type="ANALYSIS_RESULT",
        resource_id=result.id,
        access_reason="analysis_results.detail",
    )
    return result


@analysis_router.get("/results/{result_id}/detail")
async def get_analysis_result_detail(
    result_id: int, request: Request, user: Annotated[User, Depends(get_request_user)]
):
    result = ensure_found(await analysis_service.get_analysis_result(result_id), "분석 결과를 찾을 수 없습니다.")
    ensure_owner(result.user_id, user)
    await safe_record_sensitive_access(
        request=request,
        actor=user,
        target_user_id=result.user_id,
        resource_type="ANALYSIS_RESULT",
        resource_id=result.id,
        access_reason="analysis_results.detail_with_factors",
    )
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
