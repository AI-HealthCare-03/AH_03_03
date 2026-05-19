from typing import Any

from app.models.analysis import AnalysisResult, AnalysisResultFactor, AnalysisSnapshot


async def create_analysis_result(
    user_id: int,
    health_record_id: int | None,
    exam_report_id: int | None,
    data: dict[str, Any],
) -> AnalysisResult:
    _ = exam_report_id
    return await AnalysisResult.create(user_id=user_id, health_record_id=health_record_id, **data)


async def get_analysis_result_by_id(result_id: int) -> AnalysisResult | None:
    return await AnalysisResult.get_or_none(id=result_id)


async def list_analysis_results_by_user(user_id: int, limit: int = 20, offset: int = 0) -> list[AnalysisResult]:
    return await AnalysisResult.filter(user_id=user_id).order_by("-analyzed_at").offset(offset).limit(limit)


async def create_analysis_factor(analysis_result_id: int, data: dict[str, Any]) -> AnalysisResultFactor:
    return await AnalysisResultFactor.create(analysis_result_id=analysis_result_id, **data)


async def create_analysis_factors(analysis_result_id: int, factors: list[dict[str, Any]]) -> list[AnalysisResultFactor]:
    objects = [AnalysisResultFactor(analysis_result_id=analysis_result_id, **factor) for factor in factors]
    if not objects:
        return []
    await AnalysisResultFactor.bulk_create(objects)
    return objects


async def list_analysis_factors_by_result(analysis_result_id: int) -> list[AnalysisResultFactor]:
    return await AnalysisResultFactor.filter(analysis_result_id=analysis_result_id).order_by("display_order", "id")


async def create_analysis_snapshot(analysis_result_id: int, data: dict[str, Any]) -> AnalysisSnapshot:
    return await AnalysisSnapshot.create(analysis_result_id=analysis_result_id, **data)


async def get_analysis_snapshot_by_result(analysis_result_id: int) -> AnalysisSnapshot | None:
    return await AnalysisSnapshot.filter(analysis_result_id=analysis_result_id).order_by("-created_at").first()


async def get_analysis_snapshot_by_id(snapshot_id: int) -> AnalysisSnapshot | None:
    return await AnalysisSnapshot.get_or_none(id=snapshot_id)
