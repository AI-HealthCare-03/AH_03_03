import logging
from typing import Any

from ai_worker.cv.food.fallback_policy import select_food_detection_candidate
from ai_worker.cv.food.nutrition.scoring.disease_food_scorer import DiseaseFoodScorer
from ai_worker.cv.food.nutrition.scoring.schemas import DISEASE_CODES, DiseaseFoodScoreRecord
from ai_worker.llm.explanation_service import generate_diet_score_explanation
from ai_worker.llm.schemas import DietScoreExplanationInput
from app.core import config
from app.dtos.diets import (
    DietAnalyzeRequest,
    DietPhotoResultCreateRequest,
    DietRecordCreateRequest,
    DietRecordUpdateRequest,
)
from app.models.diets import DietPhotoResult, DietRecord
from app.repositories import diet_repository

SCORING_SOURCE = "nutrition_rule_table"
FOOD_DETECTION_SOURCE = "rule_based_food_detection"
SCORING_FALLBACK_SOURCE = "nutrition_rule_table_unavailable"
logger = logging.getLogger(__name__)


async def create_diet_record(user_id: int, request: DietRecordCreateRequest) -> DietRecord:
    return await diet_repository.create_diet_record(user_id, request.model_dump())


async def get_diet_record(diet_record_id: int) -> DietRecord | None:
    return await diet_repository.get_diet_record_by_id(diet_record_id)


async def list_diet_records(
    user_id: int, analysis_method: str | None = None, limit: int = 20, offset: int = 0
) -> list[DietRecord]:
    return await diet_repository.list_diet_records_by_user(
        user_id=user_id,
        analysis_method=analysis_method,
        limit=limit,
        offset=offset,
    )


async def update_diet_record(diet_record_id: int, request: DietRecordUpdateRequest) -> DietRecord | None:
    return await diet_repository.update_diet_record(diet_record_id, request.model_dump(exclude_unset=True))


async def delete_diet_record(diet_record_id: int) -> int:
    return await diet_repository.delete_diet_record(diet_record_id)


async def create_diet_photo_result(diet_record_id: int, request: DietPhotoResultCreateRequest) -> DietPhotoResult:
    return await diet_repository.create_diet_photo_result(diet_record_id, request.model_dump())


async def get_diet_photo_result(photo_result_id: int) -> DietPhotoResult | None:
    return await diet_repository.get_diet_photo_result_by_id(photo_result_id)


async def list_diet_photo_results(diet_record_id: int, limit: int = 20, offset: int = 0) -> list[DietPhotoResult]:
    return await diet_repository.list_diet_photo_results(diet_record_id, limit=limit, offset=offset)


async def run_diet_analysis(user_id: int, request: DietAnalyzeRequest) -> dict[str, object]:
    case = _select_rule_based_diet_case(request)
    # 자체 CV/GPT Vision은 아직 기본 경로가 아니므로, 시연은 규칙 기반 음식 후보로 안정성을 우선한다.
    food_candidate = select_food_detection_candidate(
        cv_result=None,
        rule_based_foods=case["detected_foods"],
        gpt_vision_fallback_enabled=config.GPT_VISION_FALLBACK_ENABLED,
        confidence_threshold=config.FOOD_CV_CONFIDENCE_THRESHOLD,
    )
    detected_foods = food_candidate.to_scorer_foods()
    scoring_result = _safe_build_nutrition_scoring_result(detected_foods)
    explanation = _safe_generate_diet_explanation(scoring_result["disease_scores"])
    nutrition_summary = {
        **case["nutrition_summary"],
        "disease_scores": scoring_result["disease_scores"],
        "scoring_source": scoring_result["scoring_source"],
        "explanation": explanation,
    }
    diet_record = await create_diet_record(
        user_id,
        DietRecordCreateRequest(
            meal_type=request.meal_type,
            meal_time=request.meal_time,
            description=request.description or "간편 식단 분석 기록",
            image_path=request.image_path,
            detected_foods=detected_foods,
            nutrition_summary=nutrition_summary,
            diet_score=case["diet_score"],
            diet_feedback=case["diet_feedback"],
            analysis_method="IMAGE_ANALYSIS",
            memo=request.memo,
        ),
    )
    photo_result = await create_diet_photo_result(
        diet_record.id,
        DietPhotoResultCreateRequest(
            detected_foods=detected_foods,
            confidence_payload={
                "method": food_candidate.provider,
                "average_confidence": food_candidate.confidence,
                "needs_review": food_candidate.needs_review,
                "fallback_reason": food_candidate.fallback_reason,
                "scoring_source": scoring_result["scoring_source"],
                "gpt_vision_called": False,
            },
            raw_output={
                "source": food_candidate.provider,
                "case": case["case_name"],
                "foods": detected_foods,
                "provider_result": {
                    "provider": food_candidate.provider,
                    "confidence": food_candidate.confidence,
                    "needs_review": food_candidate.needs_review,
                    "fallback_reason": food_candidate.fallback_reason,
                    "raw_output": food_candidate.raw_output,
                },
                "disease_scores": scoring_result["disease_scores"],
                "food_score_details": scoring_result["food_score_details"],
                "scoring_source": scoring_result["scoring_source"],
                "explanation": explanation,
            },
            is_dummy=False,
        ),
    )
    return {
        "message": "식단 분석이 완료되었습니다.",
        "diet_record": diet_record,
        "photo_result": photo_result,
        "detected_foods": detected_foods,
        "nutrition_summary": nutrition_summary,
        "diet_score": case["diet_score"],
        "diet_feedback": case["diet_feedback"],
        "disease_scores": scoring_result["disease_scores"],
        "food_score_details": scoring_result["food_score_details"],
        "scoring_source": scoring_result["scoring_source"],
        "explanation": explanation,
        "warnings": case["warnings"],
        "recommended_actions": case["recommended_actions"],
    }


def build_nutrition_scoring_result(
    detected_foods: list[dict[str, Any]],
    scorer: DiseaseFoodScorer | None = None,
) -> dict[str, Any]:
    scorer = scorer or DiseaseFoodScorer()
    runtime_scores = scorer.load_runtime_scores()
    details = [_build_food_score_detail(food, runtime_scores) for food in detected_foods]
    matched_scores = [detail["scores"] for detail in details if detail["scores"] is not None]
    return {
        "detected_foods": [_food_name(food) for food in detected_foods],
        "disease_scores": _average_disease_scores(matched_scores),
        "food_score_details": details,
        "scoring_source": SCORING_SOURCE,
    }


def _safe_build_nutrition_scoring_result(detected_foods: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        return build_nutrition_scoring_result(detected_foods)
    except Exception:
        # 점수 테이블 로드 실패가 식단 기록 자체를 막지 않도록 null 점수 payload로 낮춘다.
        logger.exception("Diet nutrition scoring failed; continuing with unavailable score payload")
        return {
            "detected_foods": [_food_name(food) for food in detected_foods],
            "disease_scores": {code: None for code in DISEASE_CODES},
            "food_score_details": [
                {
                    "food_name": _food_name(food),
                    "matched_food_name": None,
                    "scores": None,
                    "match_status": "scoring_unavailable",
                }
                for food in detected_foods
            ],
            "scoring_source": SCORING_FALLBACK_SOURCE,
        }


def _safe_generate_diet_explanation(disease_scores: dict[str, float | int | None]) -> dict[str, Any]:
    try:
        return generate_diet_score_explanation(DietScoreExplanationInput(disease_scores=disease_scores)).model_dump()
    except Exception:
        # 설명 생성은 보조 기능이므로 실패 시에도 안전 문구가 포함된 기본 설명을 반환한다.
        logger.exception("Diet score explanation failed; using base safety explanation")
        return {
            "summary": "식단 점수 설명을 생성하지 못했습니다.",
            "caution": "식단 분석 결과는 건강관리 참고용으로만 확인해 주세요.",
            "recommended_action": "음식명과 섭취량을 확인한 뒤 다시 분석해 보세요.",
            "safety_notice": "이 설명은 진단이 아니며, 건강관리 참고용입니다. 정확한 진단과 치료는 의료진 상담이 필요합니다.",
            "source": "rule_based_explanation_fallback",
            "reference_summary": None,
            "reference_sources": [],
        }


def _build_food_score_detail(food: dict[str, Any], runtime_scores: list[DiseaseFoodScoreRecord]) -> dict[str, Any]:
    food_name = _food_name(food)
    matched = _match_food_score(food_name, runtime_scores)
    if matched is None:
        return {
            "food_name": food_name,
            "matched_food_name": None,
            "scores": None,
            "match_status": "unmatched",
        }
    return {
        "food_name": food_name,
        "matched_food_name": matched.food_name,
        "scores": _record_disease_scores(matched),
        "match_status": "matched",
    }


def _food_name(food: dict[str, Any]) -> str:
    return str(food.get("name") or food.get("food_name") or "").strip()


def _match_food_score(food_name: str, runtime_scores: list[DiseaseFoodScoreRecord]) -> DiseaseFoodScoreRecord | None:
    normalized_name = _normalize_food_name(food_name)
    if not normalized_name:
        return None
    for record in runtime_scores:
        if _normalize_food_name(record.food_name) == normalized_name:
            return record
    for record in runtime_scores:
        record_name = _normalize_food_name(record.food_name)
        if normalized_name in record_name or record_name in normalized_name:
            return record
    return None


def _normalize_food_name(value: str) -> str:
    return value.replace(" ", "").lower()


def _record_disease_scores(record: DiseaseFoodScoreRecord) -> dict[str, float]:
    return {
        "DM": record.dm_score,
        "HTN": record.htn_score,
        "DL": record.dl_score,
        "OBE": record.obe_score,
        "ANEM": record.anem_score,
    }


def _average_disease_scores(scores: list[dict[str, float] | None]) -> dict[str, float | None]:
    if not scores:
        return {code: None for code in DISEASE_CODES}
    return {
        code: round(sum(score[code] for score in scores if score is not None) / len(scores), 1)
        for code in DISEASE_CODES
    }


def _select_rule_based_diet_case(request: DietAnalyzeRequest) -> dict[str, Any]:
    text = f"{request.meal_type or ''} {request.description or ''}".lower()
    if "breakfast" in text or "아침" in text:
        return _diet_case(
            "breakfast",
            [
                {"name": "오트밀", "confidence": 0.91},
                {"name": "삶은 달걀", "confidence": 0.88},
                {"name": "블루베리", "confidence": 0.82},
            ],
            {"calories": 430, "carbohydrate_g": 48, "protein_g": 22, "fat_g": 14, "sodium_mg": 420},
            88.0,
            "아침 식사로 균형이 좋습니다. 단백질 구성을 유지해 보세요.",
            [],
            ["물 한 컵 함께 마시기", "오전 간식은 견과류 소량 선택"],
            0.87,
        )
    if "lunch" in text or "점심" in text:
        return _diet_case(
            "lunch",
            [
                {"name": "현미밥", "confidence": 0.92},
                {"name": "닭가슴살", "confidence": 0.88},
                {"name": "샐러드", "confidence": 0.84},
            ],
            {"calories": 620, "carbohydrate_g": 72, "protein_g": 38, "fat_g": 18, "sodium_mg": 780},
            82.5,
            "단백질과 채소 구성이 좋은 편입니다. 나트륨은 조금만 더 낮춰보세요.",
            ["소스류 나트륨 확인 필요"],
            ["드레싱은 절반만 사용", "식후 10분 산책"],
            0.88,
        )
    if "dinner" in text or "저녁" in text:
        return _diet_case(
            "dinner",
            [
                {"name": "잡곡밥", "confidence": 0.86},
                {"name": "된장찌개", "confidence": 0.81},
                {"name": "생선구이", "confidence": 0.87},
            ],
            {"calories": 710, "carbohydrate_g": 80, "protein_g": 34, "fat_g": 24, "sodium_mg": 1280},
            74.0,
            "저녁으로는 나트륨이 다소 높을 수 있습니다. 국물 섭취를 줄여보세요.",
            ["나트륨 주의"],
            ["국물은 절반 이하로 섭취", "야식 피하기"],
            0.85,
        )
    if "snack" in text or "간식" in text:
        return _diet_case(
            "snack",
            [{"name": "카페라떼", "confidence": 0.83}, {"name": "쿠키", "confidence": 0.79}],
            {"calories": 360, "carbohydrate_g": 48, "protein_g": 8, "fat_g": 14, "sodium_mg": 210},
            58.0,
            "당류와 정제 탄수화물이 많은 간식입니다. 대체 간식을 고려해 보세요.",
            ["당류 섭취 주의"],
            ["무가당 음료 선택", "견과류나 요거트로 대체"],
            0.81,
        )
    return _diet_case(
        "default",
        [{"name": "일반식", "confidence": 0.76}, {"name": "채소반찬", "confidence": 0.72}],
        {"calories": 650, "carbohydrate_g": 78, "protein_g": 25, "fat_g": 22, "sodium_mg": 900},
        70.0,
        "식단 정보가 부족해 일반식 기준으로 분석했습니다.",
        ["정확한 음식명 입력 시 더 나은 분석 가능"],
        ["음식 설명 자세히 입력", "채소와 단백질 비율 확인"],
        0.74,
    )


def _diet_case(
    case_name: str,
    detected_foods: list[dict[str, Any]],
    nutrition_summary: dict[str, Any],
    diet_score: float,
    diet_feedback: str,
    warnings: list[str],
    recommended_actions: list[str],
    average_confidence: float,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "detected_foods": detected_foods,
        "nutrition_summary": nutrition_summary,
        "diet_score": diet_score,
        "diet_feedback": diet_feedback,
        "warnings": warnings,
        "recommended_actions": recommended_actions,
        "average_confidence": average_confidence,
    }
