import logging
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from ai_runtime.llm.explanation_service import (
    generate_analysis_explanation,
    generate_explanation_with_context,
    retrieve_health_context,
)
from ai_runtime.llm.schemas import AnalysisExplanationInput, HealthRiskFactor
from ai_runtime.ml.inference.feature_mapper import map_service_features
from ai_runtime.ml.inference.screening_predictor import load_screening_artifact
from ai_runtime.ml.inference.screening_risk_service import predict_screening_dual_stage_risk
from ai_runtime.ml.inference.x2_stage_mapper import X2ResultSource, map_x2_stage_to_risk_level
from app.core import config
from app.dtos.analysis import (
    AnalysisResultCreateRequest,
    AnalysisResultFactorCreateRequest,
    AnalysisResultResponse,
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
from app.services import exams as exam_service
from app.services.health import (
    REQUIRED_BASIC_ANALYSIS_FIELDS,
    REQUIRED_USER_ANALYSIS_FIELDS,
)

logger = logging.getLogger(__name__)
BASIC_SCREENING_DISEASE_CODES: dict[AnalysisType, str] = {
    AnalysisType.HYPERTENSION: "HTN",
    AnalysisType.DIABETES: "DM",
    AnalysisType.DYSLIPIDEMIA: "DL",
}
BASIC_ANALYSIS_TYPES = (
    AnalysisType.DIABETES,
    AnalysisType.OBESITY,
    AnalysisType.DYSLIPIDEMIA,
    AnalysisType.HYPERTENSION,
)
X2_ONLY_ANALYSIS_TYPES = (
    AnalysisType.ABDOMINAL_OBESITY,
    AnalysisType.FATTY_LIVER,
    AnalysisType.ANEMIA,
    AnalysisType.LIVER_FUNCTION,
    AnalysisType.KIDNEY_FUNCTION,
    AnalysisType.CHRONIC_KIDNEY_DISEASE,
)
X2_ANALYSIS_TYPES = (
    AnalysisType.HYPERTENSION,
    AnalysisType.DIABETES,
    AnalysisType.DYSLIPIDEMIA,
    AnalysisType.OBESITY,
    *X2_ONLY_ANALYSIS_TYPES,
)

RISK_LEVEL_LABELS: dict[RiskLevel, str] = {
    RiskLevel.LOW: "낮음",
    RiskLevel.ATTENTION: "관심 필요",
    RiskLevel.CAUTION: "주의",
    RiskLevel.HIGH_CAUTION: "높은 주의",
}

# 화면 표시용 관리 필요도 점수입니다. 질병 확률이나 진단 확률로 해석하면 안 됩니다.
RISK_LEVEL_DISPLAY_PERCENTS: dict[RiskLevel, int] = {
    RiskLevel.LOW: 25,
    RiskLevel.ATTENTION: 45,
    RiskLevel.CAUTION: 65,
    RiskLevel.HIGH_CAUTION: 80,
}


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
    return await analysis_repository.list_analysis_results_by_user(user_id, limit=len(AnalysisType), offset=0)


async def get_analysis_result_response(result: AnalysisResult) -> dict[str, Any]:
    snapshot = await get_analysis_snapshot(int(result.id))
    return _analysis_result_response(result, snapshot)


async def list_analysis_result_responses(results: list[AnalysisResult]) -> list[dict[str, Any]]:
    return [await get_analysis_result_response(result) for result in results]


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
        "result": _analysis_result_response(result, snapshot),
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
    return missing_fields


async def run_analysis(
    user_id: int, health_record: HealthRecord, mode: AnalysisMode = AnalysisMode.BASIC
) -> list[dict[str, Any]]:
    results = []
    user = await User.get_or_none(id=user_id)
    plans = await _build_analysis_plans(user=user, health_record=health_record, mode=mode)
    for plan in plans:
        analysis_type = plan["analysis_type"]
        score = plan["score"]
        risk_level = plan["risk_level"]
        model_name = plan["model_name"]
        model_version = plan["model_version"]
        screening_dual_stage = plan.get("screening_dual_stage")
        x2_rule = plan.get("x2_rule")
        precision_input = plan.get("precision_input")
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
        factors = await create_analysis_factors(
            result.id,
            _analysis_factors(analysis_type, health_record, score, mode, x2_rule=x2_rule),
        )
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
                screening_dual_stage=screening_dual_stage,
                x2_rule=x2_rule,
                precision_input=precision_input,
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
                "model_name": result.model_name,
                "model_version": result.model_version,
                "guide_message": request.summary,
                "explanation": _analysis_explanation(result, factors),
                "challenge_recommendation_ids": recommendation_ids,
                "factor_count": len(factors),
                **_risk_level_alias_fields(risk_level),
                **_x2_response_fields_from_payloads(
                    input_payload=precision_input,
                    final_outputs=_x2_rule_final_outputs(x2_rule),
                    x2_rule=x2_rule,
                ),
            }
        )
    return results


async def _build_analysis_plans(
    *,
    user: User | None,
    health_record: HealthRecord,
    mode: AnalysisMode,
) -> list[dict[str, Any]]:
    if mode == AnalysisMode.PRECISION:
        return await _build_precision_analysis_plans(user=user, health_record=health_record)
    return _build_basic_analysis_plans(user=user, health_record=health_record)


def _build_basic_analysis_plans(*, user: User | None, health_record: HealthRecord) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for analysis_type, score in _calculate_analysis_scores(health_record, AnalysisMode.BASIC, user).items():
        base_risk_level = _risk_level_for_analysis_score(analysis_type, score, health_record)
        screening_dual_stage = _predict_basic_screening_dual_stage(
            user=user,
            health_record=health_record,
            analysis_type=analysis_type,
            base_risk_level=base_risk_level,
            analysis_mode=AnalysisMode.BASIC,
        )
        risk_level = _resolve_final_risk_level(base_risk_level, screening_dual_stage)
        plans.append(
            {
                "analysis_type": analysis_type,
                "score": score,
                "risk_level": risk_level,
                "model_name": "rule_based",
                "model_version": "web-basic-v1",
                "screening_dual_stage": screening_dual_stage,
            }
        )
    return plans


async def _build_precision_analysis_plans(*, user: User | None, health_record: HealthRecord) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    base_plans = {
        plan["analysis_type"]: plan for plan in _build_basic_analysis_plans(user=user, health_record=health_record)
    }
    precision_input = await build_precision_analysis_input_payload(user=user, health_record=health_record)
    x2_features = precision_input["x2_input_payload"]

    for analysis_type in BASIC_ANALYSIS_TYPES:
        base_plan = base_plans[analysis_type]
        x2_result = map_x2_stage_to_risk_level(analysis_type.value, x2_features)
        x2_rule = x2_result.to_dict()
        if x2_result.result_source == X2ResultSource.X2_RULE.value and x2_result.risk_level is not None:
            risk_level = RiskLevel(x2_result.risk_level)
            plans.append(
                {
                    "analysis_type": analysis_type,
                    "score": _risk_score_for_level(risk_level),
                    "risk_level": risk_level,
                    "model_name": "x2_rule",
                    "model_version": "x2-rule-v1",
                    "x2_rule": x2_rule,
                    "precision_input": precision_input,
                }
            )
            continue

        x2_rule["result_source"] = "BASIC_FALLBACK"
        plans.append(
            {
                "analysis_type": analysis_type,
                "score": base_plan["score"],
                "risk_level": base_plan["risk_level"],
                "model_name": "rule_based",
                "model_version": "web-basic-fallback-v1",
                "screening_dual_stage": base_plan.get("screening_dual_stage"),
                "x2_rule": x2_rule,
                "precision_input": precision_input,
            }
        )

    for analysis_type in X2_ONLY_ANALYSIS_TYPES:
        x2_result = map_x2_stage_to_risk_level(analysis_type.value, x2_features)
        if x2_result.result_source != X2ResultSource.X2_RULE.value or x2_result.risk_level is None:
            continue
        risk_level = RiskLevel(x2_result.risk_level)
        plans.append(
            {
                "analysis_type": analysis_type,
                "score": _risk_score_for_level(risk_level),
                "risk_level": risk_level,
                "model_name": "x2_rule",
                "model_version": "x2-rule-v1",
                "x2_rule": x2_result.to_dict(),
                "precision_input": precision_input,
            }
        )

    return plans


def _predict_ml_outputs(user: User | None, health_record: HealthRecord) -> dict[AnalysisType, Any]:
    if user is None:
        return {}
    try:
        from ai_runtime.ml.inference.disease_risk_service import predict_chronic_disease_risks
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


def _predict_basic_screening_dual_stage(
    *,
    user: User | None,
    health_record: HealthRecord,
    analysis_type: AnalysisType,
    base_risk_level: RiskLevel,
    analysis_mode: AnalysisMode,
) -> dict[str, Any] | None:
    if analysis_mode != AnalysisMode.BASIC or user is None:
        return None

    disease_code = BASIC_SCREENING_DISEASE_CODES.get(analysis_type)
    if disease_code is None:
        return None

    try:
        artifact = load_screening_artifact(disease_code)
        mapping = map_service_features(user, health_record, artifact.feature_columns, strict=False)
        result = predict_screening_dual_stage_risk(
            disease_code=disease_code,
            features=mapping.features,
            base_risk_level=base_risk_level.value,
        )
    except Exception as exc:
        logger.exception(
            "BASIC screening dual-stage prediction failed; keeping rule-based result",
            extra={
                "analysis_type": analysis_type.value,
                "disease_code": disease_code,
                "health_record_id": health_record.id,
            },
        )
        return {
            "status": "fallback_basic_only",
            "disease_code": disease_code,
            "base_risk_level": base_risk_level.value,
            "fallback_reason": exc.__class__.__name__,
            "note": "screening dual-stage 실패로 기존 BASIC rule 결과를 유지합니다.",
        }

    return {
        "status": "applied",
        "disease_code": result.disease_code,
        "base_risk_level": result.base_risk_level,
        "base_high": result.base_high,
        "base_caution_or_above": result.base_caution_or_above,
        "screening_high": result.screening_high,
        "risk_level": result.risk_level,
        "service_band": result.service_band.value,
        "service_band_label": result.service_band_label,
        "service_band_percent": result.service_band_percent,
        "legacy_risk_level": None,
        "screening_missing_features": result.screening_missing_features,
        "screening_neutralized_features": result.screening_neutralized_features,
        "screening_model_count": result.screening_model_count,
        "feature_mapping_missing_sources": mapping.missing_required_sources,
        "feature_mapping_defaulted_features": mapping.defaulted_features,
        "feature_mapping_warnings": mapping.warnings,
        "note": "raw screening probability는 사용자 응답에 직접 노출하지 않습니다.",
    }


def _calculate_analysis_scores(
    record: HealthRecord, mode: AnalysisMode = AnalysisMode.BASIC, user: User | None = None
) -> dict[AnalysisType, Decimal]:
    if mode == AnalysisMode.PRECISION:
        # ML artifact가 없는 OBESITY는 정밀 모드에서도 룰 기반 검진값 보강으로 처리한다.
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
    bmi = _obesity_bmi_value(record)
    if bmi is not None:
        return _obesity_score_for_bmi(bmi)

    score = Decimal("0.18")
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
    bmi = _obesity_bmi_value(record)
    if bmi is None:
        return Decimal("0.18")
    return _obesity_score_for_bmi(bmi)


def _obesity_bmi_value(record: HealthRecord) -> Decimal | None:
    bmi = _to_decimal_value(getattr(record, "bmi", None))
    if bmi is not None:
        return bmi

    height_cm = _to_decimal_value(getattr(record, "height_cm", None))
    weight_kg = _to_decimal_value(getattr(record, "weight_kg", None))
    if height_cm is None or weight_kg is None or height_cm <= 0:
        return None

    height_m = height_cm / Decimal("100")
    return weight_kg / (height_m * height_m)


def _to_decimal_value(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _obesity_score_for_bmi(bmi: Decimal) -> Decimal:
    return _risk_score_for_level(_obesity_risk_level_for_bmi(bmi))


def _obesity_risk_level_for_bmi(bmi: Decimal) -> RiskLevel:
    if bmi >= Decimal("30"):
        return RiskLevel.HIGH_CAUTION
    if bmi >= Decimal("25"):
        return RiskLevel.CAUTION
    if bmi >= Decimal("23"):
        return RiskLevel.ATTENTION
    return RiskLevel.LOW


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
        return RiskLevel.HIGH_CAUTION
    if score >= Decimal("0.40"):
        return RiskLevel.CAUTION
    return RiskLevel.LOW


def _risk_level_for_analysis_score(analysis_type: AnalysisType, score: Decimal, record: HealthRecord) -> RiskLevel:
    if analysis_type == AnalysisType.OBESITY:
        bmi = _obesity_bmi_value(record)
        if bmi is not None:
            return _obesity_risk_level_for_bmi(bmi)
    return _risk_level(score)


def _risk_score_for_level(risk_level: RiskLevel) -> Decimal:
    return {
        RiskLevel.LOW: Decimal("0.25000"),
        RiskLevel.ATTENTION: Decimal("0.45000"),
        RiskLevel.CAUTION: Decimal("0.65000"),
        RiskLevel.HIGH_CAUTION: Decimal("0.80000"),
    }[risk_level]


def _x2_feature_payload(*, user: User | None, health_record: HealthRecord) -> dict[str, Any]:
    field_names = (
        "systolic_bp",
        "diastolic_bp",
        "fasting_glucose",
        "hba1c",
        "total_cholesterol",
        "ldl_cholesterol",
        "hdl_cholesterol",
        "triglyceride",
        "height_cm",
        "weight_kg",
        "bmi",
        "waist_cm",
        "ast",
        "alt",
        "gamma_gtp",
        "creatinine",
        "egfr",
        "urine_protein",
        "hemoglobin",
    )
    payload = {field_name: getattr(health_record, field_name, None) for field_name in field_names}
    payload["sex"] = (
        getattr(user, "gender", None) or getattr(health_record, "sex", None) or getattr(health_record, "gender", None)
    )
    return payload


X2_MEASUREMENT_ALIASES = {
    "systolicbp": "systolic_bp",
    "sbp": "systolic_bp",
    "수축기혈압": "systolic_bp",
    "수축기": "systolic_bp",
    "diastolicbp": "diastolic_bp",
    "dbp": "diastolic_bp",
    "이완기혈압": "diastolic_bp",
    "이완기": "diastolic_bp",
    "fastingglucose": "fasting_glucose",
    "glucose": "fasting_glucose",
    "glu": "fasting_glucose",
    "공복혈당": "fasting_glucose",
    "혈당": "fasting_glucose",
    "hba1c": "hba1c",
    "당화혈색소": "hba1c",
    "totalcholesterol": "total_cholesterol",
    "totalchol": "total_cholesterol",
    "tc": "total_cholesterol",
    "tcho": "total_cholesterol",
    "총콜레스테롤": "total_cholesterol",
    "ldl": "ldl_cholesterol",
    "ldlcholesterol": "ldl_cholesterol",
    "저밀도콜레스테롤": "ldl_cholesterol",
    "hdl": "hdl_cholesterol",
    "hdlcholesterol": "hdl_cholesterol",
    "고밀도콜레스테롤": "hdl_cholesterol",
    "triglyceride": "triglyceride",
    "tg": "triglyceride",
    "중성지방": "triglyceride",
    "height": "height_cm",
    "heightcm": "height_cm",
    "신장": "height_cm",
    "키": "height_cm",
    "weight": "weight_kg",
    "weightkg": "weight_kg",
    "체중": "weight_kg",
    "몸무게": "weight_kg",
    "bmi": "bmi",
    "체질량지수": "bmi",
    "waist": "waist_cm",
    "waistcm": "waist_cm",
    "허리둘레": "waist_cm",
    "복부둘레": "waist_cm",
    "hb": "hemoglobin",
    "hemoglobin": "hemoglobin",
    "혈색소": "hemoglobin",
    "헤모글로빈": "hemoglobin",
    "ast": "ast",
    "got": "ast",
    "sgot": "ast",
    "alt": "alt",
    "gpt": "alt",
    "sgpt": "alt",
    "gammagtp": "gamma_gtp",
    "gammagt": "gamma_gtp",
    "ggt": "gamma_gtp",
    "ggtp": "gamma_gtp",
    "γgtp": "gamma_gtp",
    "감마지티피": "gamma_gtp",
    "감마gtp": "gamma_gtp",
    "creatinine": "creatinine",
    "serumcreatinine": "creatinine",
    "크레아티닌": "creatinine",
    "혈청크레아티닌": "creatinine",
    "egfr": "egfr",
    "신사구체여과율": "egfr",
    "사구체여과율": "egfr",
    "urineprotein": "urine_protein",
    "proteinuria": "urine_protein",
    "요단백": "urine_protein",
}

X2_INT_FIELDS = {
    "systolic_bp",
    "diastolic_bp",
    "fasting_glucose",
    "total_cholesterol",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "triglyceride",
    "ast",
    "alt",
    "gamma_gtp",
}


async def build_precision_analysis_input_payload(
    *,
    user: User | None,
    health_record: HealthRecord,
) -> dict[str, Any]:
    x2_payload = _x2_feature_payload(user=user, health_record=health_record)
    field_sources = {
        field_name: "health_record_fallback" for field_name, value in x2_payload.items() if value is not None
    }
    selected_exam_report_id = None
    x2_measurement_source = "health_record_fallback"

    user_id = getattr(user, "id", None) or getattr(health_record, "user_id", None)
    if user_id is not None:
        exam_report, measurements = await exam_service.get_latest_confirmed_exam_measurements_for_analysis(int(user_id))
        if exam_report is not None:
            selected_exam_report_id = int(exam_report.id)
        measurement_payload, measurement_sources = _exam_measurements_to_x2_payload_with_sources(measurements)
        if measurement_payload:
            x2_payload.update(measurement_payload)
            field_sources.update(measurement_sources)
            x2_measurement_source = "exam_measurements"

    return {
        "selected_exam_report_id": selected_exam_report_id,
        "x1_input_payload": _health_record_input_payload(health_record),
        "x2_input_payload": x2_payload,
        "x2_measurement_source": x2_measurement_source,
        "x2_field_sources": field_sources,
    }


def exam_measurements_to_x2_payload(measurements: list[Any]) -> dict[str, Any]:
    payload, _sources = _exam_measurements_to_x2_payload_with_sources(measurements)
    return payload


def _exam_measurements_to_x2_payload_with_sources(measurements: list[Any]) -> tuple[dict[str, Any], dict[str, str]]:
    payload: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for measurement in measurements:
        canonical_key = _canonical_x2_measurement_key(measurement)
        if canonical_key is None:
            continue
        parsed_value = _parse_x2_measurement_value(canonical_key, getattr(measurement, "value", None))
        if parsed_value is None:
            continue
        payload[canonical_key] = parsed_value
        sources[canonical_key] = "exam_measurements"
    return payload, sources


def _canonical_x2_measurement_key(measurement: Any) -> str | None:
    candidates = (getattr(measurement, "measurement_key", None), getattr(measurement, "measurement_name", None))
    for candidate in candidates:
        normalized = _normalize_measurement_alias(candidate)
        if not normalized:
            continue
        canonical = X2_MEASUREMENT_ALIASES.get(normalized)
        if canonical is not None:
            return canonical
    return None


def _normalize_measurement_alias(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("γ", "γ").replace("–", "-").replace("—", "-")
    return re.sub(r"[\s_\-./()·:%]+", "", text)


def _parse_x2_measurement_value(field_name: str, value: object) -> Any:
    if value is None or str(value).strip() == "":
        return None
    if field_name == "urine_protein":
        return _normalize_urine_protein(value)
    parsed = exam_service.parse_exam_measurement_number(value)
    if parsed is None:
        return None
    if field_name in X2_INT_FIELDS:
        return int(parsed)
    return parsed


def _normalize_urine_protein(value: object) -> str | None:
    text = str(value).strip().lower()
    if not text:
        return None
    compact = re.sub(r"[\s()]+", "", text)
    if compact in {"-", "(-)", "negative", "neg", "음성", "정상", "normal"}:
        return "negative"
    if compact in {"trace", "±", "+-", "미량", "경계"}:
        return "trace"
    plus_match = re.search(r"(?:\+([1-4])|([1-4])\+|plus[_-]?([1-4]))", compact)
    if plus_match:
        level = next(group for group in plus_match.groups() if group is not None)
        return f"plus_{level}"
    if compact == "+":
        return "plus_1"
    return text


def _health_record_input_payload(health_record: HealthRecord) -> dict[str, Any]:
    return {
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


def _resolve_final_risk_level(base_risk_level: RiskLevel, screening_dual_stage: dict[str, Any] | None) -> RiskLevel:
    if not screening_dual_stage or screening_dual_stage.get("status") != "applied":
        return base_risk_level
    raw_risk_level = screening_dual_stage.get("risk_level") or screening_dual_stage.get("service_band")
    try:
        return RiskLevel(str(raw_risk_level))
    except ValueError:
        logger.warning(
            "Invalid screening risk level; keeping base risk level",
            extra={"risk_level": raw_risk_level, "base_risk_level": base_risk_level.value},
        )
        return base_risk_level


def _guide_message(analysis_type: AnalysisType, risk_level: RiskLevel, mode: AnalysisMode = AnalysisMode.BASIC) -> str:
    disease_label = {
        AnalysisType.HYPERTENSION: "고혈압",
        AnalysisType.DIABETES: "당뇨",
        AnalysisType.DYSLIPIDEMIA: "이상지질혈증",
        AnalysisType.OBESITY: "비만",
        AnalysisType.ABDOMINAL_OBESITY: "복부비만",
        AnalysisType.FATTY_LIVER: "지방간",
        AnalysisType.ANEMIA: "빈혈",
        AnalysisType.LIVER_FUNCTION: "간기능",
        AnalysisType.KIDNEY_FUNCTION: "신장기능",
        AnalysisType.CHRONIC_KIDNEY_DISEASE: "만성콩팥병",
    }.get(analysis_type, str(analysis_type))
    mode_label = "정밀" if mode == AnalysisMode.PRECISION else "간편"
    # 결과 문구는 위험도 안내에 머물러야 하며 진단/처방처럼 읽히면 안 된다.
    notice = f" 이 결과는 {mode_label} 분석 참고용 판정이며 의료 진단이 아닙니다."
    if risk_level == RiskLevel.HIGH_CAUTION:
        return f"{disease_label} 관련 높은 주의 단계입니다. 생활습관 기록과 의료기관 상담을 함께 고려해 주세요.{notice}"
    if risk_level == RiskLevel.CAUTION:
        return f"{disease_label} 관련 주의가 필요한 단계입니다. 식단과 활동량을 꾸준히 기록해 보세요.{notice}"
    if risk_level == RiskLevel.ATTENTION:
        return (
            f"{disease_label} 관련 관심이 필요한 단계입니다. 건강정보를 꾸준히 기록하며 변화를 확인해 보세요.{notice}"
        )
    return f"{disease_label} 관련 관리 필요도는 낮은 편입니다. 현재의 건강 기록 습관을 유지해 보세요.{notice}"


def _analysis_explanation(result: AnalysisResult, factors: list[AnalysisResultFactor]) -> dict[str, Any]:
    input_data = AnalysisExplanationInput(
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
    try:
        # keyword RAG는 참고 출처를 덧붙이는 PoC이며, 실패해도 분석 결과 저장을 막지 않는다.
        explanation = generate_explanation_with_context(
            input_data,
            contexts=retrieve_health_context(
                _analysis_reference_query(input_data),
                disease_type=result.analysis_type.value,
            ),
        )
    except Exception:
        logger.exception(
            "Analysis explanation generation failed; using base rule-based explanation",
            extra={"analysis_result_id": result.id, "analysis_type": result.analysis_type.value},
        )
        explanation = generate_analysis_explanation(input_data)
    return explanation.model_dump()


def _analysis_reference_query(input_data: AnalysisExplanationInput) -> str:
    factor_names = " ".join(factor.name for factor in input_data.factors)
    return f"{input_data.disease_type} {input_data.risk_level} {factor_names}".strip()


def _analysis_factors(
    analysis_type: AnalysisType,
    record: HealthRecord,
    score: Decimal,
    mode: AnalysisMode = AnalysisMode.BASIC,
    *,
    x2_rule: dict[str, Any] | None = None,
) -> list[AnalysisResultFactorCreateRequest]:
    common_factor = AnalysisResultFactorCreateRequest(
        factor_key="risk_score",
        factor_name="위험도 점수",
        factor_value=str(score),
        contribution_score=score,
        direction=FactorDirection.NEUTRAL,
        display_order=0,
    )
    x2_factor = _x2_factor(x2_rule)
    if analysis_type not in BASIC_ANALYSIS_TYPES:
        return [factor for factor in [common_factor, x2_factor] if factor is not None]

    basic_factor = _basic_factor(analysis_type, record)
    precision_factor_by_type = {
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
    }
    precision_factor = precision_factor_by_type[analysis_type]
    if mode == AnalysisMode.PRECISION:
        return [factor for factor in [common_factor, x2_factor, precision_factor, basic_factor] if factor is not None]
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


def _x2_factor(x2_rule: dict[str, Any] | None) -> AnalysisResultFactorCreateRequest | None:
    if not x2_rule:
        return None
    stage_label = x2_rule.get("x2_stage_label")
    source = x2_rule.get("result_source")
    missing_fields = x2_rule.get("x2_missing_fields") or []
    if stage_label:
        factor_value = str(stage_label)
    elif missing_fields:
        factor_value = f"부족 항목: {', '.join(str(field) for field in missing_fields)}"
    else:
        factor_value = str(source or "")
    return AnalysisResultFactorCreateRequest(
        factor_key="x2_rule_stage",
        factor_name="검진 수치 기반 단계",
        factor_value=factor_value,
        contribution_score=None,
        direction=FactorDirection.NEUTRAL,
        display_order=1,
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
    screening_dual_stage: dict[str, Any] | None = None,
    x2_rule: dict[str, Any] | None = None,
    precision_input: dict[str, Any] | None = None,
) -> AnalysisSnapshotCreateRequest:
    model_version = model_version or f"web-{analysis_mode.value.lower()}-v1"
    input_features = _health_record_input_payload(health_record)
    selected_exam_report_id = precision_input.get("selected_exam_report_id") if precision_input else None
    x1_input_payload = precision_input.get("x1_input_payload") if precision_input else input_features
    x2_input_payload = (
        precision_input.get("x2_input_payload")
        if precision_input
        else _x2_feature_payload(
            user=None,
            health_record=health_record,
        )
    )
    x2_measurement_source = (
        precision_input.get("x2_measurement_source") if precision_input else "health_record_fallback"
    )
    x2_field_sources = precision_input.get("x2_field_sources") if precision_input else {}
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
    final_outputs = (
        {
            "risk_level": risk_level,
            "guide_message": guide_message,
            "service_band": risk_level,
            "service_band_label": _risk_level_label(risk_level),
            "service_band_percent": _risk_level_display_percent(risk_level),
            "legacy_risk_level": None,
        }
        | _screening_dual_stage_final_outputs(screening_dual_stage)
        | _x2_rule_final_outputs(x2_rule)
    )
    return AnalysisSnapshotCreateRequest(
        input_payload=_to_json_value(
            {
                "input_features": input_features,
                "analysis_type": analysis_type,
                "analysis_mode": analysis_mode,
                "selected_exam_report_id": selected_exam_report_id,
                "x1_input_payload": x1_input_payload,
                "x2_input_payload": x2_input_payload,
                "x2_measurement_source": x2_measurement_source,
                "x2_field_sources": x2_field_sources,
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
                    "result_source": x2_rule.get("result_source") if x2_rule else "BASIC",
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
                "final_outputs": final_outputs,
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
                "screening_dual_stage": screening_dual_stage,
                "x2_rule": x2_rule,
            }
        ),
    )


def _screening_dual_stage_final_outputs(screening_dual_stage: dict[str, Any] | None) -> dict[str, Any]:
    if not screening_dual_stage:
        return {}

    status = screening_dual_stage.get("status")
    if status != "applied":
        return {
            "screening_dual_stage_status": status,
            "screening_dual_stage_fallback_reason": screening_dual_stage.get("fallback_reason"),
        }

    return {
        "screening_dual_stage_status": status,
    }


def _x2_rule_final_outputs(x2_rule: dict[str, Any] | None) -> dict[str, Any]:
    if not x2_rule:
        return {"result_source": "BASIC"}
    return {
        "result_source": x2_rule.get("result_source"),
        "x2_stage_code": x2_rule.get("x2_stage_code"),
        "x2_stage_label": x2_rule.get("x2_stage_label"),
        "x2_available": x2_rule.get("x2_available"),
        "x2_missing_fields": x2_rule.get("x2_missing_fields") or [],
    }


def _analysis_result_response(result: AnalysisResult, snapshot: AnalysisSnapshot | None = None) -> dict[str, Any]:
    payload = AnalysisResultResponse.model_validate(result).model_dump()
    payload.update(_risk_level_alias_fields(result.risk_level))
    payload.update(_x2_response_fields_from_snapshot(snapshot))
    return payload


def _x2_response_field_defaults() -> dict[str, Any]:
    return {
        "result_source": None,
        "x2_stage_code": None,
        "x2_stage_label": None,
        "x2_available": None,
        "x2_missing_fields": None,
        "selected_exam_report_id": None,
        "x2_measurement_source": None,
    }


def _x2_response_fields_from_snapshot(snapshot: AnalysisSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return _x2_response_field_defaults()

    input_payload = _as_mapping(getattr(snapshot, "input_payload", None))
    output_payload = _as_mapping(getattr(snapshot, "output_payload", None))
    model_payload = _as_mapping(getattr(snapshot, "model_payload", None))
    final_outputs = _as_mapping(output_payload.get("final_outputs"))
    x2_rule = _as_mapping(model_payload.get("x2_rule"))
    return _x2_response_fields_from_payloads(
        input_payload=input_payload,
        final_outputs=final_outputs,
        x2_rule=x2_rule,
    )


def _x2_response_fields_from_payloads(
    *,
    input_payload: dict[str, Any] | None = None,
    final_outputs: dict[str, Any] | None = None,
    x2_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_payload = _as_mapping(input_payload)
    final_outputs = _as_mapping(final_outputs)
    x2_rule = _as_mapping(x2_rule)

    def pick_x2_value(field_name: str) -> Any:
        if field_name in final_outputs:
            return final_outputs.get(field_name)
        return x2_rule.get(field_name)

    fields = _x2_response_field_defaults()
    fields.update(
        {
            "result_source": _optional_string(pick_x2_value("result_source")),
            "x2_stage_code": _optional_string(pick_x2_value("x2_stage_code")),
            "x2_stage_label": _optional_string(pick_x2_value("x2_stage_label")),
            "x2_available": _optional_bool(pick_x2_value("x2_available")),
            "x2_missing_fields": _optional_string_list(pick_x2_value("x2_missing_fields")),
            "selected_exam_report_id": _optional_int(input_payload.get("selected_exam_report_id")),
            "x2_measurement_source": _optional_string(input_payload.get("x2_measurement_source")),
        }
    )
    return fields


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_string(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _risk_level_alias_fields(risk_level: RiskLevel | str) -> dict[str, Any]:
    normalized = _coerce_risk_level(risk_level)
    return {
        "service_band": normalized.value,
        "service_band_label": _risk_level_label(normalized),
        "service_band_percent": _risk_level_display_percent(normalized),
        "legacy_risk_level": None,
    }


def _risk_level_label(risk_level: RiskLevel) -> str:
    return RISK_LEVEL_LABELS[risk_level]


def _risk_level_display_percent(risk_level: RiskLevel) -> int:
    return RISK_LEVEL_DISPLAY_PERCENTS[risk_level]


def _coerce_risk_level(risk_level: RiskLevel | str) -> RiskLevel:
    if isinstance(risk_level, RiskLevel):
        return risk_level
    legacy_map = {
        "LOW": RiskLevel.LOW,
        "MEDIUM": RiskLevel.CAUTION,
        "HIGH": RiskLevel.HIGH_CAUTION,
    }
    value = str(risk_level)
    return legacy_map.get(value, RiskLevel(value))


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
    }.get(result.analysis_type)
    if target_category is None:
        return []
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
