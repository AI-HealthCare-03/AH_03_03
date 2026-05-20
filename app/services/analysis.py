from datetime import datetime
from decimal import Decimal
from typing import Any

from app.core import config
from app.dtos.analysis import (
    AnalysisResultCreateRequest,
    AnalysisResultFactorCreateRequest,
    AnalysisSnapshotCreateRequest,
)
from app.dtos.challenges import ChallengeRecommendationCreateRequest
from app.models.analysis import (
    AnalysisResult,
    AnalysisResultFactor,
    AnalysisSnapshot,
    AnalysisType,
    FactorDirection,
    RiskLevel,
)
from app.models.challenges import ChallengeCategory
from app.models.health import HealthRecord
from app.repositories import analysis_repository
from app.services import challenges as challenge_service


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


async def list_latest_analysis_results(user_id: int) -> list[AnalysisResult]:
    return await analysis_repository.list_analysis_results_by_user(user_id, limit=3, offset=0)


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


async def run_dummy_analysis(user_id: int, health_record: HealthRecord) -> list[dict[str, Any]]:
    results = []
    for analysis_type, score in _calculate_dummy_scores(health_record).items():
        risk_level = _risk_level(score)
        request = AnalysisResultCreateRequest(
            health_record_id=health_record.id,
            analysis_type=analysis_type,
            risk_score=score,
            risk_level=risk_level,
            summary=_guide_message(analysis_type, risk_level),
            model_name="dummy_rule_based",
            model_version="mvp-demo-v1",
            analyzed_at=datetime.now(config.TIMEZONE),
        )
        result = await create_analysis_result(user_id, request)
        factors = await create_analysis_factors(result.id, _dummy_factors(analysis_type, health_record, score))
        recommendation_ids = await _create_dummy_challenge_recommendations(user_id, result)
        results.append(
            {
                "analysis_result_id": result.id,
                "analysis_type": result.analysis_type,
                "risk_score": result.risk_score,
                "risk_level": result.risk_level,
                "guide_message": request.summary,
                "challenge_recommendation_ids": recommendation_ids,
                "factor_count": len(factors),
            }
        )
    return results


def _calculate_dummy_scores(record: HealthRecord) -> dict[AnalysisType, Decimal]:
    return {
        AnalysisType.DIABETES: _diabetes_score(record),
        AnalysisType.OBESITY: _obesity_score(record),
        AnalysisType.DYSLIPIDEMIA: _dyslipidemia_score(record),
    }


def _diabetes_score(record: HealthRecord) -> Decimal:
    score = Decimal("0.18")
    if record.fasting_glucose is not None:
        if record.fasting_glucose >= 126:
            score = max(score, Decimal("0.78"))
        elif record.fasting_glucose >= 100:
            score = max(score, Decimal("0.52"))
        else:
            score = max(score, Decimal("0.25"))
    if record.hba1c is not None:
        if record.hba1c >= Decimal("6.5"):
            score = max(score, Decimal("0.82"))
        elif record.hba1c >= Decimal("5.7"):
            score = max(score, Decimal("0.56"))
    return score


def _obesity_score(record: HealthRecord) -> Decimal:
    score = Decimal("0.18")
    if record.bmi is not None:
        if record.bmi >= Decimal("30"):
            score = max(score, Decimal("0.82"))
        elif record.bmi >= Decimal("25"):
            score = max(score, Decimal("0.68"))
        elif record.bmi >= Decimal("23"):
            score = max(score, Decimal("0.46"))
        else:
            score = max(score, Decimal("0.22"))
    if record.waist_cm is not None and record.waist_cm >= Decimal("90"):
        score = min(score + Decimal("0.10"), Decimal("0.95"))
    return score


def _dyslipidemia_score(record: HealthRecord) -> Decimal:
    score = Decimal("0.18")
    if record.ldl_cholesterol is not None:
        if record.ldl_cholesterol >= 160:
            score = max(score, Decimal("0.78"))
        elif record.ldl_cholesterol >= 130:
            score = max(score, Decimal("0.55"))
    if record.triglyceride is not None:
        if record.triglyceride >= 200:
            score = max(score, Decimal("0.72"))
        elif record.triglyceride >= 150:
            score = max(score, Decimal("0.52"))
    if record.hdl_cholesterol is not None and record.hdl_cholesterol < 40:
        score = max(score, Decimal("0.58"))
    if record.total_cholesterol is not None and record.total_cholesterol >= 240:
        score = max(score, Decimal("0.70"))
    return score


def _risk_level(score: Decimal) -> RiskLevel:
    if score >= Decimal("0.70"):
        return RiskLevel.HIGH
    if score >= Decimal("0.40"):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _guide_message(analysis_type: AnalysisType, risk_level: RiskLevel) -> str:
    disease_label = {
        AnalysisType.DIABETES: "당뇨",
        AnalysisType.OBESITY: "비만",
        AnalysisType.DYSLIPIDEMIA: "이상지질혈증",
    }[analysis_type]
    if risk_level == RiskLevel.HIGH:
        return f"{disease_label} 관련 지표가 높게 나타났습니다. 생활습관 기록과 전문가 상담을 함께 고려해 주세요."
    if risk_level == RiskLevel.MEDIUM:
        return f"{disease_label} 관련 관리가 필요한 구간입니다. 식단과 활동량을 꾸준히 기록해 보세요."
    return f"{disease_label} 관련 위험도는 낮은 편입니다. 현재의 건강 기록 습관을 유지해 보세요."


def _dummy_factors(
    analysis_type: AnalysisType, record: HealthRecord, score: Decimal
) -> list[AnalysisResultFactorCreateRequest]:
    common_factor = AnalysisResultFactorCreateRequest(
        factor_key="dummy_risk_score",
        factor_name="더미 위험도 점수",
        factor_value=str(score),
        contribution_score=score,
        direction=FactorDirection.NEUTRAL,
        display_order=0,
    )
    disease_factor = {
        AnalysisType.DIABETES: AnalysisResultFactorCreateRequest(
            factor_key="fasting_glucose",
            factor_name="공복혈당",
            factor_value=str(record.fasting_glucose) if record.fasting_glucose is not None else None,
            contribution_score=Decimal("0.40") if record.fasting_glucose is not None else Decimal("0.00"),
            direction=FactorDirection.POSITIVE,
            display_order=1,
        ),
        AnalysisType.OBESITY: AnalysisResultFactorCreateRequest(
            factor_key="bmi",
            factor_name="BMI",
            factor_value=str(record.bmi) if record.bmi is not None else None,
            contribution_score=Decimal("0.45") if record.bmi is not None else Decimal("0.00"),
            direction=FactorDirection.POSITIVE,
            display_order=1,
        ),
        AnalysisType.DYSLIPIDEMIA: AnalysisResultFactorCreateRequest(
            factor_key="ldl_cholesterol",
            factor_name="LDL 콜레스테롤",
            factor_value=str(record.ldl_cholesterol) if record.ldl_cholesterol is not None else None,
            contribution_score=Decimal("0.35") if record.ldl_cholesterol is not None else Decimal("0.00"),
            direction=FactorDirection.POSITIVE,
            display_order=1,
        ),
    }[analysis_type]
    return [common_factor, disease_factor]


async def _create_dummy_challenge_recommendations(user_id: int, result: AnalysisResult) -> list[int]:
    active_challenges = await challenge_service.list_active_challenges(limit=100)
    target_category = {
        AnalysisType.DIABETES: ChallengeCategory.BLOOD_GLUCOSE,
        AnalysisType.OBESITY: ChallengeCategory.WEIGHT,
        AnalysisType.DYSLIPIDEMIA: ChallengeCategory.DIET,
    }[result.analysis_type]
    challenge = next((item for item in active_challenges if item.category == target_category), None)
    if challenge is None and active_challenges:
        challenge = active_challenges[0]
    if challenge is None:
        return []

    recommendation = await challenge_service.create_challenge_recommendation(
        user_id=user_id,
        analysis_result_id=result.id,
        challenge_id=challenge.id,
        request=ChallengeRecommendationCreateRequest(
            challenge_id=challenge.id,
            analysis_result_id=result.id,
            reason=_guide_message(result.analysis_type, result.risk_level),
            priority=1,
            is_selected=False,
        ),
    )
    return [recommendation.id]
