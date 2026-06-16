import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_runtime.cv.food.matcher import FoodMatchResult, LocalFallbackFoodDbMatcher, match_food_name
from ai_runtime.cv.food.normalization import normalize_food_name
from ai_runtime.cv.food.nutrition.providers import MfdsFoodDbMatcher
from ai_runtime.cv.food.nutrition.scoring.disease_food_scorer import DiseaseFoodScorer
from ai_runtime.cv.food.nutrition.scoring.schemas import DISEASE_CODES, DiseaseFoodScoreRecord
from ai_runtime.cv.food.pipeline import FoodAnalysisPipelineConfig, run_food_analysis_pipeline
from ai_runtime.cv.providers.gpt_vision import VisionClient
from ai_runtime.llm.explanation_service import generate_diet_score_explanation
from ai_runtime.llm.schemas import DietScoreExplanationInput
from app.core import config
from app.dtos.diets import (
    DietAnalyzeRequest,
    DietAnalyzeResponse,
    DietPhotoResultCreateRequest,
    DietRecordCreateRequest,
    DietRecordResponse,
    DietRecordUpdateRequest,
)
from app.models.diets import DietPhotoResult, DietRecord
from app.repositories import diet_repository
from app.services.storage import get_storage_service, normalize_storage_key

SCORING_SOURCE = "nutrition_rule_table"
FOOD_DETECTION_SOURCE = "rule_based_food_detection"
SCORING_FALLBACK_SOURCE = "nutrition_rule_table_unavailable"
DIET_ANALYSIS_SERVICE_UNAVAILABLE = "diet_analysis_service_unavailable"
DIET_ANALYSIS_UPLOAD_DIR = "diet_analysis"
DIET_ANALYSIS_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
logger = logging.getLogger(__name__)


async def create_diet_record(user_id: int, request: DietRecordCreateRequest) -> DietRecord:
    return await diet_repository.create_diet_record(user_id, request.model_dump())


async def get_diet_record(diet_record_id: int) -> DietRecord | None:
    return await diet_repository.get_diet_record_by_id(diet_record_id)


def build_diet_record_response(record: DietRecord) -> DietRecordResponse:
    return DietRecordResponse(
        id=int(record.id),
        user_id=int(record.user_id),
        meal_type=getattr(record, "meal_type", None),
        meal_time=getattr(record, "meal_time", None),
        description=getattr(record, "description", None),
        image_path=getattr(record, "image_path", None),
        detected_foods=getattr(record, "detected_foods", None),
        nutrition_summary=getattr(record, "nutrition_summary", None),
        diet_score=getattr(record, "diet_score", None),
        diet_feedback=getattr(record, "diet_feedback", None),
        analysis_method=getattr(record, "analysis_method", None),
        is_user_corrected=bool(getattr(record, "is_user_corrected", False)),
        memo=getattr(record, "memo", None),
        image_url=diet_record_image_url(record),
        created_at=getattr(record, "created_at", None) or datetime.now(UTC),
        updated_at=getattr(record, "updated_at", None) or datetime.now(UTC),
    )


def diet_record_image_url(record: DietRecord) -> str | None:
    image_key = _diet_record_image_key(record)
    if not image_key:
        return None
    try:
        if not get_storage_service().exists(image_key):
            return None
    except (OSError, ValueError):
        return None
    return f"/api/v1/diets/{int(record.id)}/image"


def read_diet_record_image(record: DietRecord) -> tuple[bytes, str, str] | None:
    image_key = _diet_record_image_key(record)
    if not image_key:
        return None
    storage = get_storage_service()
    try:
        if not storage.exists(image_key):
            return None
        return (
            storage.read_bytes(image_key),
            _media_type_from_upload_path(image_key) or "application/octet-stream",
            Path(image_key).name,
        )
    except (OSError, ValueError):
        return None


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


async def run_diet_analysis(
    user_id: int,
    request: DietAnalyzeRequest,
    image_bytes: bytes | None = None,
    image_media_type: str | None = None,
) -> dict[str, object]:
    ensure_diet_analysis_available()
    demo_fallback_enabled = _diet_demo_fallback_enabled()
    case = _select_rule_based_diet_case(request) if demo_fallback_enabled else None
    food_analysis = await run_food_analysis_pipeline(
        rule_based_foods=case["detected_foods"] if case is not None else [],
        image_bytes=image_bytes,
        image_media_type=image_media_type,
        config=FoodAnalysisPipelineConfig(
            provider=str(config.DIET_VISION_PROVIDER or "rule_based"),
            gpt_vision_enabled=bool(config.DIET_GPT_VISION_ENABLED),
            gpt_vision_fallback_enabled=bool(config.GPT_VISION_FALLBACK_ENABLED),
            confidence_threshold=float(config.FOOD_CV_CONFIDENCE_THRESHOLD),
            openai_api_key=config.OPENAI_API_KEY,
            gpt_vision_model=config.DIET_GPT_VISION_MODEL,
            vision_client_cls=VisionClient,
            food_db_matcher=_build_food_db_matcher(),
        ),
    )
    food_candidate = food_analysis.food_candidate
    detected_foods = food_analysis.detected_foods
    fallback_used = food_analysis.fallback_used
    provider_message = food_analysis.provider_message
    if food_candidate.provider == FOOD_DETECTION_SOURCE and not demo_fallback_enabled:
        logger.warning(
            "Diet analysis rejected rule-based fallback without DIET_DEMO_FALLBACK_ENABLED",
            extra={"provider_message": provider_message},
        )
        raise ValueError(DIET_ANALYSIS_SERVICE_UNAVAILABLE)
    scoring_result = _safe_build_nutrition_scoring_result(detected_foods)
    explanation = _safe_generate_diet_explanation(scoring_result["disease_scores"])
    nutrition_summary = {
        **(case["nutrition_summary"] if case is not None else {}),
        "disease_scores": scoring_result["disease_scores"],
        "scoring_source": scoring_result["scoring_source"],
        "explanation": explanation,
    }
    diet_feedback = _diet_feedback(case, food_candidate.needs_review)
    diet_record = await create_diet_record(
        user_id,
        DietRecordCreateRequest(
            meal_type=request.meal_type,
            meal_time=request.meal_time,
            description=request.description or "간편 식단 분석 기록",
            image_path=request.image_path,
            detected_foods=detected_foods,
            nutrition_summary=nutrition_summary,
            diet_score=None,
            diet_feedback=diet_feedback,
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
                "gpt_vision_called": food_analysis.provider_candidate is not None,
                "fallback_used": fallback_used,
                "provider_message": provider_message,
            },
            raw_output={
                "source": food_candidate.provider,
                "vision_provider": food_candidate.provider,
                "fallback_used": fallback_used,
                "provider_message": provider_message,
                **({"demo_fallback_case": case["case_name"]} if case is not None else {}),
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
        "diet_record": build_diet_record_response(diet_record),
        "photo_result": photo_result,
        "detected_foods": detected_foods,
        "nutrition_summary": nutrition_summary,
        "diet_score": None,
        "diet_feedback": diet_feedback,
        "disease_scores": scoring_result["disease_scores"],
        "food_score_details": scoring_result["food_score_details"],
        "scoring_source": scoring_result["scoring_source"],
        "vision_provider": food_candidate.provider,
        "fallback_used": fallback_used,
        "raw_output": photo_result.raw_output,
        "explanation": explanation,
        "warnings": case["warnings"] if case is not None else [],
        "recommended_actions": case["recommended_actions"] if case is not None else [],
    }


def ensure_diet_analysis_available() -> None:
    if _diet_demo_fallback_enabled() or _gpt_vision_diet_provider_available():
        return
    raise ValueError(DIET_ANALYSIS_SERVICE_UNAVAILABLE)


def _diet_demo_fallback_enabled() -> bool:
    return bool(getattr(config, "DIET_DEMO_FALLBACK_ENABLED", False))


def _gpt_vision_diet_provider_available() -> bool:
    provider = str(config.DIET_VISION_PROVIDER or "").strip().lower()
    api_key = str(config.OPENAI_API_KEY or "").strip()
    return provider == "gpt_vision" and bool(config.DIET_GPT_VISION_ENABLED) and bool(api_key)


def _diet_feedback(case: dict[str, Any] | None, needs_review: bool) -> str:
    if case is not None:
        return str(case["diet_feedback"])
    if needs_review:
        return "음식 후보와 영양성분 후보를 확인해 주세요."
    return "음식 후보와 영양성분 후보를 기준으로 식단 관리 포인트를 확인해 주세요."


def store_diet_analysis_upload(
    *,
    user_id: int,
    image_bytes: bytes,
    image_media_type: str | None,
    filename: str | None,
) -> dict[str, str]:
    suffix = _safe_image_suffix(filename, image_media_type)
    upload_key = _build_diet_analysis_upload_key(user_id=user_id, suffix=suffix)
    stored_key = get_storage_service().save_bytes(
        image_bytes,
        upload_key,
        content_type=image_media_type or DIET_ANALYSIS_MEDIA_TYPES.get(suffix),
    )
    return {
        "upload_path": stored_key,
        "image_media_type": image_media_type or _media_type_from_upload_path(stored_key),
        "image_filename": filename or Path(stored_key).name,
    }


async def run_diet_analysis_from_job(job_id: int) -> DietAnalyzeResponse:
    from app.services import async_jobs as async_job_service

    job = await async_job_service.get_job(job_id)
    if job is None:
        raise ValueError("diet_analysis_job_not_found")

    payload = job.request_payload or {}
    user_id = _payload_int(payload, "user_id")
    if user_id is None:
        raise ValueError("diet_analysis_user_id_missing")

    upload_path = str(payload.get("upload_path") or "")
    request = DietAnalyzeRequest(
        meal_type=str(payload.get("meal_type") or "") or None,
        meal_time=payload.get("meal_time") or None,
        description=str(payload.get("description") or "") or None,
        image_path=str(upload_path or payload.get("image_path") or "") or None,
        memo=str(payload.get("memo") or "") or None,
    )
    response = await run_diet_analysis(
        user_id,
        request,
        image_bytes=_read_diet_analysis_bytes(upload_path),
        image_media_type=str(payload.get("image_media_type") or "") or _media_type_from_upload_path(upload_path),
    )
    return DietAnalyzeResponse.model_validate(response)


def _build_diet_analysis_upload_key(*, user_id: int, suffix: str) -> str:
    return normalize_storage_key(f"diet-analysis/{user_id}/{uuid4().hex}/source{suffix}")


def _safe_image_suffix(filename: str | None, media_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in DIET_ANALYSIS_MEDIA_TYPES:
        return suffix
    if media_type == "image/png":
        return ".png"
    if media_type == "image/webp":
        return ".webp"
    return ".jpg"


def _read_diet_analysis_bytes(path_value: str) -> bytes | None:
    if not path_value:
        return None
    try:
        storage = get_storage_service()
        if storage.exists(path_value):
            return storage.read_bytes(path_value)
    except ValueError:
        pass

    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise ValueError("diet_analysis_upload_missing")
    return path.read_bytes()


def _build_food_db_matcher() -> MfdsFoodDbMatcher | None:
    if not config.DIET_MFDS_ENABLED:
        return None
    service_key = (config.MFDS_SERVICE_KEY or "").strip()
    encoded_service_key = (config.MFDS_SERVICE_KEY_ENCODED or "").strip()
    if not service_key and not encoded_service_key:
        logger.warning("DIET_MFDS_ENABLED=true but MFDS service key is missing; using local food matcher")
        return None
    return MfdsFoodDbMatcher(
        service_key=service_key or encoded_service_key,
        encoded_service_key=encoded_service_key or None,
        timeout_seconds=float(config.DIET_MFDS_TIMEOUT_SECONDS),
        max_candidates=int(config.DIET_MFDS_MAX_CANDIDATES),
        fallback_matcher=LocalFallbackFoodDbMatcher(),
    )


def _diet_record_image_key(record: DietRecord) -> str | None:
    image_path = str(getattr(record, "image_path", "") or "").strip()
    if not image_path or not _looks_like_diet_analysis_storage_key(image_path):
        return None
    try:
        return normalize_storage_key(image_path)
    except ValueError:
        return None


def _looks_like_diet_analysis_storage_key(value: str) -> bool:
    normalized = value.replace("\\", "/").strip("/")
    return normalized.startswith("diet-analysis/") or "/diet-analysis/" in normalized


def _media_type_from_upload_path(path_value: str) -> str | None:
    if not path_value:
        return None
    return DIET_ANALYSIS_MEDIA_TYPES.get(Path(path_value).suffix.lower())


def _payload_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_nutrition_scoring_result(
    detected_foods: list[dict[str, Any]],
    scorer: DiseaseFoodScorer | None = None,
) -> dict[str, Any]:
    scorer = scorer or DiseaseFoodScorer()
    runtime_scores = scorer.load_runtime_scores()
    details = [_build_food_score_detail(food, runtime_scores) for food in detected_foods]
    matched_scores = [detail["scores"] for detail in details if detail["scores"] is not None]
    return {
        "detected_foods": [str(detail["food_name"]) for detail in details if detail["food_name"]],
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
                    "original_name": _food_match(food).original_name,
                    "query_name": _food_match(food).query_name,
                    "matched_food_name": None,
                    "matched_food_code": _food_match(food).matched_food_code,
                    "match_source": _food_match(food).match_source,
                    "match_confidence": _food_match(food).match_confidence,
                    "needs_user_confirmation": _food_match(food).needs_user_confirmation,
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
    match = _food_match(food)
    use_query_name_as_display = _is_mfds_match(match)
    lookup_name = match.query_name if use_query_name_as_display else match.matched_food_name or match.query_name
    matched = _match_food_score(lookup_name, runtime_scores) if lookup_name else None
    matched_food_name = match.matched_food_name or (matched.food_name if matched is not None else None)
    needs_user_confirmation = match.needs_user_confirmation and matched is None
    match_source = (
        match.match_source if match.matched_food_name else SCORING_SOURCE if matched is not None else match.match_source
    )
    match_confidence = match.match_confidence if match.matched_food_name else 0.7 if matched is not None else None
    base_detail = {
        "food_name": match.query_name if use_query_name_as_display else matched_food_name or match.query_name,
        "original_name": match.original_name,
        "query_name": match.query_name,
        "matched_food_name": matched_food_name,
        "matched_food_code": match.matched_food_code,
        "match_source": match_source,
        "match_confidence": match_confidence,
        "needs_user_confirmation": needs_user_confirmation,
    }
    if matched is None:
        return {
            **base_detail,
            "scores": None,
            "match_status": "needs_user_confirmation" if needs_user_confirmation else "unmatched",
        }
    return {
        **base_detail,
        "scores": _record_disease_scores(matched),
        "match_status": "matched",
    }


def _food_name(food: dict[str, Any]) -> str:
    match = _food_match(food)
    if _is_mfds_match(match):
        return str(match.query_name or match.original_name)
    return str(match.matched_food_name or match.query_name or match.original_name)


def _is_mfds_match(match: FoodMatchResult) -> bool:
    return str(match.match_source or "").startswith("mfds_")


def _food_match(food: dict[str, Any]) -> FoodMatchResult:
    original_name = str(food.get("original_name") or food.get("name") or food.get("food_name") or "").strip()
    matched_food_name = _optional_string(food.get("matched_food_name"))
    query_name = _optional_string(food.get("query_name")) or original_name
    if any(key in food for key in ("query_name", "matched_food_name", "matched_food_code", "match_source")):
        return FoodMatchResult(
            original_name=original_name,
            query_name=query_name,
            matched_food_name=matched_food_name,
            matched_food_code=_optional_string(food.get("matched_food_code")),
            match_source=_optional_string(food.get("match_source")) or "provided_payload",
            match_confidence=_optional_float(food.get("match_confidence")),
            needs_user_confirmation=bool(food.get("needs_user_confirmation", matched_food_name is None)),
        )
    return match_food_name(original_name)


def _match_food_score(food_name: str, runtime_scores: list[DiseaseFoodScoreRecord]) -> DiseaseFoodScoreRecord | None:
    normalized_name = _normalize_food_name(food_name)
    if not normalized_name:
        return None
    for record in runtime_scores:
        if _normalize_food_name(record.food_name) == normalized_name:
            return record
    if len(normalized_name) < 3:
        return None
    for record in runtime_scores:
        record_name = _normalize_food_name(record.food_name)
        if normalized_name in record_name or record_name in normalized_name:
            return record
    return None


def _normalize_food_name(value: str) -> str:
    return normalize_food_name(value)


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
            "단백질과 채소 구성이 적절한 편입니다. 나트륨은 조금만 더 낮춰보세요.",
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
            "당류와 정제 탄수화물이 많은 간식입니다. 대체 간식을 고려해 보세요.",
            ["당류 섭취 주의"],
            ["무가당 음료 선택", "견과류나 요거트로 대체"],
            0.81,
        )
    return _diet_case(
        "default",
        [{"name": "일반식", "confidence": 0.76}, {"name": "채소반찬", "confidence": 0.72}],
        {"calories": 650, "carbohydrate_g": 78, "protein_g": 25, "fat_g": 22, "sodium_mg": 900},
        "식단 정보가 부족해 일반식 기준으로 분석했습니다.",
        ["정확한 음식명 입력 시 더 나은 분석 가능"],
        ["음식 설명 자세히 입력", "채소와 단백질 비율 확인"],
        0.74,
    )


def _diet_case(
    case_name: str,
    detected_foods: list[dict[str, Any]],
    nutrition_summary: dict[str, Any],
    diet_feedback: str,
    warnings: list[str],
    recommended_actions: list[str],
    average_confidence: float,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "detected_foods": detected_foods,
        "nutrition_summary": nutrition_summary,
        "diet_feedback": diet_feedback,
        "warnings": warnings,
        "recommended_actions": recommended_actions,
        "average_confidence": average_confidence,
    }
