from typing import Any

from app.dtos.analysis import (
    AnalysisResultCreateRequest,
    AnalysisResultFactorCreateRequest,
    AnalysisSnapshotCreateRequest,
)
from app.models.analysis import AnalysisResult, AnalysisResultFactor, AnalysisSnapshot
from app.repositories import analysis_repository


async def create_analysis_result(user_id: int, request: AnalysisResultCreateRequest) -> AnalysisResult:
    data = request.model_dump(exclude={"health_record_id"})
    return await analysis_repository.create_analysis_result(
        user_id=user_id,
        health_record_id=request.health_record_id,
        exam_report_id=None,
        data=data,
    )


async def get_analysis_result(result_id: int) -> AnalysisResult | None:
    return await analysis_repository.get_analysis_result_by_id(result_id)


async def list_analysis_results(user_id: int, limit: int = 20, offset: int = 0) -> list[AnalysisResult]:
    return await analysis_repository.list_analysis_results_by_user(user_id, limit=limit, offset=offset)


async def create_analysis_factor(
    analysis_result_id: int, request: AnalysisResultFactorCreateRequest
) -> AnalysisResultFactor:
    return await analysis_repository.create_analysis_factor(analysis_result_id, request.model_dump())


async def create_analysis_factors(
    analysis_result_id: int, factors: list[AnalysisResultFactorCreateRequest]
) -> list[AnalysisResultFactor]:
    data = [factor.model_dump() for factor in factors]
    return await analysis_repository.create_analysis_factors(analysis_result_id, data)


async def list_analysis_factors(analysis_result_id: int) -> list[AnalysisResultFactor]:
    return await analysis_repository.list_analysis_factors_by_result(analysis_result_id)


async def create_analysis_snapshot(analysis_result_id: int, request: AnalysisSnapshotCreateRequest) -> AnalysisSnapshot:
    return await analysis_repository.create_analysis_snapshot(analysis_result_id, request.model_dump())


async def get_analysis_snapshot(analysis_result_id: int) -> AnalysisSnapshot | None:
    return await analysis_repository.get_analysis_snapshot_by_result(analysis_result_id)


async def get_analysis_snapshot_by_id(snapshot_id: int) -> AnalysisSnapshot | None:
    return await analysis_repository.get_analysis_snapshot_by_id(snapshot_id)


async def get_analysis_result_detail(result_id: int) -> dict[str, Any] | None:
    result = await get_analysis_result(result_id)
    if result is None:
        return None

    factors = await list_analysis_factors(result_id)
    snapshot = await get_analysis_snapshot(result_id)
    return {"result": result, "factors": factors, "snapshot": snapshot}
