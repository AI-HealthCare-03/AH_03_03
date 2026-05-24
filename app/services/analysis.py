import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from ai_worker.llm.explanation_service import generate_analysis_explanation
from ai_worker.llm.schemas import AnalysisExplanationInput, HealthRiskFactor
from app.core import config
from app.dtos.analysis import (
    AnalysisResultCreateRequest,
    AnalysisResultFactorCreateRequest,
    AnalysisSnapshotCreateRequest,
)
from app.dtos.challenges import ChallengeRecommendationCreateRequest
from app.models.analysis import (
    AnalysisMode,
    AnalysisResult,
    AnalysisResultFactor,
    AnalysisSnapshot,
    AnalysisType,
    FactorDirection,
    RiskLevel,
)
from app.models.challenges import ChallengeCategory
from app.models.health import HealthRecord
from app.models.users import User
from app.repositories import analysis_repository
from app.services import challenges as challenge_service
from app.services.health import (
    REQUIRED_BASIC_ANALYSIS_FIELDS,
    REQUIRED_PRECISION_ANALYSIS_FIELDS,
    REQUIRED_USER_ANALYSIS_FIELDS,
)

logger = logging.getLogger(__name__)


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
    return await analysis_repository.list_analysis_results_by_user(user_id, limit=4, offset=0)


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
    return {
        "result": result,
        "factors": factors,
        "snapshot": snapshot,
        "explanation": _analysis_explanation(result, factors),
    }


async def get_missing_fields_for_mode(user: User, health_record: HealthRecord, mode: AnalysisMode) -> list[str]:
    missing_fields = [
        label
        for field_name, label in REQUIRED_USER_ANALYSIS_FIELDS.items()
        if _is_missing_value(getattr(user, field_name))
    ]
    missing_fields.extend(
        label
        for field_name, label in REQUIRED_BASIC_ANALYSIS_FIELDS.items()
        if _is_missing_health_record_field(health_record, field_name)
    )
    if mode == AnalysisMode.PRECISION:
        missing_fields.extend(
            label
            for field_name, label in REQUIRED_PRECISION_ANALYSIS_FIELDS.items()
            if _is_missing_value(getattr(health_record, field_name))
        )
    return missing_fields


async def run_analysis(
    user_id: int, health_record: HealthRecord, mode: AnalysisMode = AnalysisMode.BASIC
) -> list[dict[str, Any]]:
    results = []
    user = await User.get_or_none(id=user_id)
    ml_predictions = _predict_ml_outputs(user, health_record) if mode == AnalysisMode.PRECISION else {}
    for analysis_type, fallback_score in _calculate_analysis_scores(health_record, mode, user).items():
        prediction = ml_predictions.get(analysis_type)
        score = Decimal(str(round(prediction.probability, 5))) if prediction is not None else fallback_score
        risk_level = _risk_level(score)
        model_name = prediction.model_name if prediction is not None else "rule_based"
        model_version = prediction.model_version if prediction is not None else f"web-{mode.value.lower()}-v1"
        request = AnalysisResultCreateRequest(
            health_record_id=health_record.id,
            analysis_type=analysis_type,
            analysis_mode=mode,
            risk_score=score,
            risk_level=risk_level,
            summary=_guide_message(analysis_type, risk_level, mode),
            model_name=model_name,
            model_version=model_version,
            analyzed_at=datetime.now(config.TIMEZONE),
        )
        result = await create_analysis_result(user_id, request)
        factors = await create_analysis_factors(result.id, _analysis_factors(analysis_type, health_record, score, mode))
        await create_analysis_snapshot(
            result.id,
            _analysis_snapshot_request(
                analysis_type=analysis_type,
                analysis_mode=mode,
                health_record=health_record,
                score=score,
                risk_level=risk_level,
                guide_message=request.summary or "",
                factors=factors,
                model_name=model_name,
                model_version=model_version,
                model_prediction=prediction.to_dict() if prediction is not None else None,
            ),
        )
        recommendation_ids = await _create_challenge_recommendations(user_id, result)
        results.append(
            {
                "analysis_result_id": result.id,
                "analysis_type": result.analysis_type,
                "analysis_mode": result.analysis_mode,
                "risk_score": result.risk_score,
                "risk_level": result.risk_level,
                "guide_message": request.summary,
                "explanation": _analysis_explanation(result, factors),
                "challenge_recommendation_ids": recommendation_ids,
                "factor_count": len(factors),
            }
        )
    return results


def _predict_ml_outputs(user: User | None, health_record: HealthRecord) -> dict[AnalysisType, Any]:
    if user is None:
        return {}
    try:
        from ai_worker.ml.inference.disease_risk_service import predict_chronic_disease_risks
    except Exception:
        logger.exception(
            "ML prediction import failed; falling back to rule-based precision analysis",
            extra={"health_record_id": health_record.id},
        )
        return {}

    try:
        raw_predictions = predict_chronic_disease_risks(user, health_record, diseases=["DM", "HTN", "DL"])
    except Exception:
        logger.exception(
            "ML prediction failed; falling back to rule-based precision analysis",
            extra={"user_id": user.id, "health_record_id": health_record.id},
        )
        return {}

    disease_map = {
        "DM": AnalysisType.DIABETES,
        "HTN": AnalysisType.HYPERTENSION,
        "DL": AnalysisType.DYSLIPIDEMIA,
    }
    return {
        disease_map[disease]: prediction for disease, prediction in raw_predictions.items() if disease in disease_map
    }


def _calculate_analysis_scores(
    record: HealthRecord, mode: AnalysisMode = AnalysisMode.BASIC, user: User | None = None
) -> dict[AnalysisType, Decimal]:
    if mode == AnalysisMode.PRECISION:
        return {
            AnalysisType.DIABETES: max(_basic_diabetes_score(record, user), _diabetes_score(record)),
            AnalysisType.OBESITY: max(_basic_obesity_score(record, user), _obesity_score(record)),
            AnalysisType.DYSLIPIDEMIA: max(_basic_dyslipidemia_score(record, user), _dyslipidemia_score(record)),
            AnalysisType.HYPERTENSION: max(_basic_hypertension_score(record, user), _hypertension_score(record)),
        }
    return {
        AnalysisType.DIABETES: _basic_diabetes_score(record, user),
        AnalysisType.OBESITY: _basic_obesity_score(record, user),
        AnalysisType.DYSLIPIDEMIA: _basic_dyslipidemia_score(record, user),
        AnalysisType.HYPERTENSION: _basic_hypertension_score(record, user),
    }


def _basic_diabetes_score(record: HealthRecord, user: User | None) -> Decimal:
    score = Decimal("0.20")
    score += _age_adjustment(user)
    if record.family_dm == "YES":
        score += Decimal("0.16")
    if record.bmi is not None and record.bmi >= Decimal("25"):
        score += Decimal("0.10")
    score += _lifestyle_adjustment(record)
    return min(score, Decimal("0.88"))


def _basic_obesity_score(record: HealthRecord, user: User | None) -> Decimal:
    _ = user
    score = Decimal("0.18")
    if record.bmi is not None:
        if record.bmi >= Decimal("30"):
            score = Decimal("0.78")
        elif record.bmi >= Decimal("25"):
            score = Decimal("0.64")
        elif record.bmi >= Decimal("23"):
            score = Decimal("0.46")
        else:
            score = Decimal("0.22")
    score += _lifestyle_adjustment(record)
    return min(score, Decimal("0.90"))


def _basic_dyslipidemia_score(record: HealthRecord, user: User | None) -> Decimal:
    score = Decimal("0.18")
    score += _age_adjustment(user)
    if record.family_dyslipidemia == "YES":
        score += Decimal("0.14")
    if record.bmi is not None and record.bmi >= Decimal("25"):
        score += Decimal("0.10")
    score += _lifestyle_adjustment(record)
    return min(score, Decimal("0.82"))


def _basic_hypertension_score(record: HealthRecord, user: User | None) -> Decimal:
    score = Decimal("0.20")
    score += _age_adjustment(user)
    if record.family_htn == "YES":
        score += Decimal("0.16")
    if record.bmi is not None and record.bmi >= Decimal("25"):
        score += Decimal("0.10")
    score += _lifestyle_adjustment(record)
    return min(score, Decimal("0.86"))


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


def _hypertension_score(record: HealthRecord) -> Decimal:
    is_high = (record.systolic_bp is not None and record.systolic_bp >= 140) or (
        record.diastolic_bp is not None and record.diastolic_bp >= 90
    )
    if is_high:
        return Decimal("0.78")

    is_medium = (record.systolic_bp is not None and record.systolic_bp >= 130) or (
        record.diastolic_bp is not None and record.diastolic_bp >= 80
    )
    if is_medium:
        return Decimal("0.52")

    return Decimal("0.18")


def _risk_level(score: Decimal) -> RiskLevel:
    if score >= Decimal("0.70"):
        return RiskLevel.HIGH
    if score >= Decimal("0.40"):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _guide_message(analysis_type: AnalysisType, risk_level: RiskLevel, mode: AnalysisMode = AnalysisMode.BASIC) -> str:
    disease_label = {
        AnalysisType.DIABETES: "당뇨",
        AnalysisType.OBESITY: "비만",
        AnalysisType.DYSLIPIDEMIA: "이상지질혈증",
        AnalysisType.HYPERTENSION: "고혈압",
    }[analysis_type]
    mode_label = "정밀" if mode == AnalysisMode.PRECISION else "간편"
    notice = f" 이 결과는 {mode_label} 분석 참고용 판정이며 의료 진단이 아닙니다."
    if risk_level == RiskLevel.HIGH:
        return f"{disease_label} 관련 위험도가 높게 나타났습니다. 생활습관 기록과 의료기관 상담을 함께 고려해 주세요.{notice}"
    if risk_level == RiskLevel.MEDIUM:
        return f"{disease_label} 관련 관리가 필요한 구간입니다. 식단과 활동량을 꾸준히 기록해 보세요.{notice}"
    return f"{disease_label} 관련 위험도는 낮은 편입니다. 현재의 건강 기록 습관을 유지해 보세요.{notice}"


def _analysis_explanation(result: AnalysisResult, factors: list[AnalysisResultFactor]) -> dict[str, Any]:
    explanation = generate_analysis_explanation(
        AnalysisExplanationInput(
            disease_type=result.analysis_type.value,
            risk_score=str(result.risk_score),
            risk_level=result.risk_level.value,
            model_name=result.model_name,
            model_version=result.model_version,
            factors=[
                HealthRiskFactor(
                    name=factor.factor_name,
                    value=factor.factor_value,
                    reason=factor.direction.value
                    if isinstance(factor.direction, FactorDirection)
                    else str(factor.direction),
                )
                for factor in factors
            ],
        )
    )
    return explanation.model_dump()


def _analysis_factors(
    analysis_type: AnalysisType, record: HealthRecord, score: Decimal, mode: AnalysisMode = AnalysisMode.BASIC
) -> list[AnalysisResultFactorCreateRequest]:
    common_factor = AnalysisResultFactorCreateRequest(
        factor_key="risk_score",
        factor_name="위험도 점수",
        factor_value=str(score),
        contribution_score=score,
        direction=FactorDirection.NEUTRAL,
        display_order=0,
    )
    basic_factor = _basic_factor(analysis_type, record)
    precision_factor = {
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
        AnalysisType.HYPERTENSION: AnalysisResultFactorCreateRequest(
            factor_key="blood_pressure",
            factor_name="혈압",
            factor_value=_blood_pressure_value(record),
            contribution_score=Decimal("0.42")
            if record.systolic_bp is not None or record.diastolic_bp is not None
            else Decimal("0.00"),
            direction=FactorDirection.POSITIVE,
            display_order=1,
        ),
    }[analysis_type]
    if mode == AnalysisMode.PRECISION:
        return [common_factor, precision_factor, basic_factor]
    return [common_factor, basic_factor]


def _basic_factor(analysis_type: AnalysisType, record: HealthRecord) -> AnalysisResultFactorCreateRequest:
    factor_key, factor_name, factor_value = {
        AnalysisType.DIABETES: ("family_dm", "당뇨병 가족력", record.family_dm),
        AnalysisType.OBESITY: ("bmi", "BMI", record.bmi),
        AnalysisType.DYSLIPIDEMIA: ("family_dyslipidemia", "이상지질혈증 가족력", record.family_dyslipidemia),
        AnalysisType.HYPERTENSION: ("family_htn", "고혈압 가족력", record.family_htn),
    }[analysis_type]
    return AnalysisResultFactorCreateRequest(
        factor_key=factor_key,
        factor_name=factor_name,
        factor_value=str(factor_value) if factor_value is not None else None,
        contribution_score=Decimal("0.24"),
        direction=FactorDirection.POSITIVE,
        display_order=2,
    )


def _analysis_snapshot_request(
    analysis_type: AnalysisType,
    analysis_mode: AnalysisMode,
    health_record: HealthRecord,
    score: Decimal,
    risk_level: RiskLevel,
    guide_message: str,
    factors: list[AnalysisResultFactor],
    model_name: str = "rule_based",
    model_version: str | None = None,
    model_prediction: dict[str, Any] | None = None,
) -> AnalysisSnapshotCreateRequest:
    model_version = model_version or f"web-{analysis_mode.value.lower()}-v1"
    input_features = {
        "height_cm": _to_json_value(health_record.height_cm),
        "weight_kg": _to_json_value(health_record.weight_kg),
        "bmi": _to_json_value(health_record.bmi),
        "occupation_code": health_record.occupation_code,
        "family_htn": health_record.family_htn,
        "family_dm": health_record.family_dm,
        "family_dyslipidemia": health_record.family_dyslipidemia,
        "smoking_status": health_record.smoking_status,
        "drinking_frequency": health_record.drinking_frequency,
        "drinking_amount": health_record.drinking_amount,
        "walking_days_per_week": health_record.walking_days_per_week,
        "strength_days_per_week": health_record.strength_days_per_week,
        "waist_cm": _to_json_value(health_record.waist_cm),
        "systolic_bp": health_record.systolic_bp,
        "diastolic_bp": health_record.diastolic_bp,
        "fasting_glucose": health_record.fasting_glucose,
        "hba1c": _to_json_value(health_record.hba1c),
        "total_cholesterol": health_record.total_cholesterol,
        "ldl_cholesterol": health_record.ldl_cholesterol,
        "hdl_cholesterol": health_record.hdl_cholesterol,
        "triglyceride": health_record.triglyceride,
    }
    shap_outputs = [
        {
            "factor_key": factor.factor_key,
            "factor_name": factor.factor_name,
            "factor_value": factor.factor_value,
            "contribution_score": _to_json_value(factor.contribution_score),
            "direction": factor.direction,
        }
        for factor in factors
    ]
    return AnalysisSnapshotCreateRequest(
        input_payload=_to_json_value(
            {
                "input_features": input_features,
                "analysis_type": analysis_type,
                "analysis_mode": analysis_mode,
            }
        ),
        output_payload=_to_json_value(
            {
                "model_outputs": {
                    analysis_type: {
                        "risk_score": _to_json_value(score),
                    }
                },
                "rule_outputs": {
                    "low_threshold": 0.40,
                    "high_threshold": 0.70,
                    "rule_engine": "rule_based",
                    "analysis_mode": analysis_mode,
                    "note": "참고용 룰 기반 분석이며 실제 의료 진단 또는 처방이 아닙니다.",
                    "hypertension_rule": {
                        "high": "systolic_bp >= 140 또는 diastolic_bp >= 90",
                        "medium": "systolic_bp >= 130 또는 diastolic_bp >= 80",
                        "low": "그 외",
                    }
                    if analysis_type == AnalysisType.HYPERTENSION
                    else None,
                },
                "final_outputs": {
                    "risk_level": risk_level,
                    "guide_message": guide_message,
                },
                "model_version_info": {
                    "model_name": model_name,
                    "model_version": model_version,
                },
            }
        ),
        shap_payload=_to_json_value(
            {
                "note": "룰 기반 분석의 주요 요인입니다.",
                "factors": shap_outputs,
            }
        ),
        model_payload=_to_json_value(
            {
                "model_name": model_name,
                "model_version": model_version,
                "analysis_mode": analysis_mode,
                "prediction": model_prediction,
            }
        ),
    )


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(_to_json_value(key)): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json_value(item) for item in value]
    return value


def _blood_pressure_value(record: HealthRecord) -> str | None:
    if record.systolic_bp is None and record.diastolic_bp is None:
        return None
    systolic = record.systolic_bp if record.systolic_bp is not None else "-"
    diastolic = record.diastolic_bp if record.diastolic_bp is not None else "-"
    return f"{systolic}/{diastolic}"


def _age_adjustment(user: User | None) -> Decimal:
    if user is None or user.birthday is None:
        return Decimal("0.00")
    age = datetime.now(config.TIMEZONE).date().year - user.birthday.year
    if age >= 60:
        return Decimal("0.14")
    if age >= 45:
        return Decimal("0.08")
    return Decimal("0.00")


def _lifestyle_adjustment(record: HealthRecord) -> Decimal:
    score = Decimal("0.00")
    if record.smoking_status == "CURRENT_SMOKER":
        score += Decimal("0.08")
    if record.drinking_frequency in {"WEEKLY_2_3", "WEEKLY_4_PLUS", "DAILY"}:
        score += Decimal("0.06")
    if record.drinking_amount in {"FIVE_TO_SIX", "SEVEN_PLUS", "HEAVY"}:
        score += Decimal("0.06")
    if record.walking_days_per_week is not None and record.walking_days_per_week <= 2:
        score += Decimal("0.05")
    if record.strength_days_per_week is not None and record.strength_days_per_week == 0:
        score += Decimal("0.04")
    return score


def _is_missing_value(value: object) -> bool:
    return value is None or value == ""


def _is_missing_health_record_field(record: HealthRecord, field_name: str) -> bool:
    if field_name == "bmi" and not _is_missing_value(record.height_cm) and not _is_missing_value(record.weight_kg):
        return False
    return _is_missing_value(getattr(record, field_name))


async def _create_challenge_recommendations(user_id: int, result: AnalysisResult) -> list[int]:
    active_challenges = await challenge_service.list_active_challenges(limit=100)
    target_category = {
        AnalysisType.DIABETES: ChallengeCategory.BLOOD_GLUCOSE,
        AnalysisType.OBESITY: ChallengeCategory.WEIGHT,
        AnalysisType.DYSLIPIDEMIA: ChallengeCategory.DIET,
        AnalysisType.HYPERTENSION: ChallengeCategory.BLOOD_PRESSURE,
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
            reason=_guide_message(result.analysis_type, result.risk_level, result.analysis_mode),
            priority=1,
            is_selected=False,
        ),
    )
    return [recommendation.id]
