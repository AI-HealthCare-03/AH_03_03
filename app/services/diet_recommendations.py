import asyncio
import re
import time
from collections.abc import Iterable
from typing import Any

from fastapi import HTTPException
from starlette import status

from ai_runtime.llm.diet_recommendation_rewriter import rewrite_diet_rag_comment
from ai_runtime.llm.llm_client import record_langfuse_event
from ai_runtime.llm.rag.diet_sources import DIET_RAG_QUERY_TEMPLATES, ISSUE_TO_RAG_CODES, enabled_rag_codes
from ai_runtime.llm.rag.embeddings import get_embedding_provider
from ai_runtime.llm.rag.keyword_retriever import KeywordRagMatch, retrieve_keyword_rag_matches
from ai_runtime.llm.rag.retriever import RetrievedDocument
from ai_runtime.llm.rag.vector_retriever import VectorRagRetriever
from app.core import config
from app.core.providers import has_openai_config
from app.models.analysis import AnalysisType, RiskLevel
from app.repositories import analysis_repository, health_repository
from app.services import challenges as challenge_service
from app.services import diets as diet_service

SAFETY_NOTICE = "이 내용은 의료적 판단이 아닌 생활관리 참고 정보입니다. 실제 섭취량이 확정되지 않아 영양 평가는 참고용으로 봐 주세요."

MAX_FINDINGS = 5
MAX_RECOMMENDED_CHALLENGES = 3
MAX_RAG_EVIDENCE_SOURCES = 5
GENERAL_RAG_CODES = {"DIET_NUTRITION", "DIET_CAUTION", "DIET_FAQ"}
RAG_FALLBACK_SUMMARY = (
    "이번 식단은 기본 추천을 먼저 참고해 보세요. 기록을 이어가면 더 잘 맞는 생활관리 포인트를 살펴볼 수 있습니다."
)
DIET_RAG_STRATEGY_KEYWORD_ONLY = "keyword_only"
DIET_RAG_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK = "keyword_first_vector_fallback"
DIET_RAG_STRATEGY_VECTOR_DISABLED = "vector_disabled"
SUPPORTED_DIET_RAG_STRATEGIES = {
    DIET_RAG_STRATEGY_KEYWORD_ONLY,
    DIET_RAG_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK,
    DIET_RAG_STRATEGY_VECTOR_DISABLED,
}
DIET_RAG_MIN_KEYWORD_RESULTS = 1

NUTRIENT_ALIASES = {
    "calories_kcal": ("calories_kcal", "calories", "kcal", "energy", "energy_kcal", "열량"),
    "carbohydrate_g": ("carbohydrate_g", "carbohydrate", "carbs", "carb_g", "탄수화물"),
    "protein_g": ("protein_g", "protein", "단백질"),
    "fat_g": ("fat_g", "fat", "지방"),
    "sodium_mg": ("sodium_mg", "sodium", "나트륨"),
    "sugar_g": ("sugar_g", "sugar", "sugars", "당류", "당"),
    "saturated_fat_g": ("saturated_fat_g", "saturated_fat", "sat_fat", "포화지방"),
    "fiber_g": ("fiber_g", "fiber", "dietary_fiber", "식이섬유"),
    "iron_mg": ("iron_mg", "iron", "철"),
    "calcium_mg": ("calcium_mg", "calcium", "칼슘"),
    "potassium_mg": ("potassium_mg", "potassium", "칼륨"),
}

DIET_CHALLENGE_RULES = {
    "sodium_high": ["염분 빼볼까염 챌린지", "식사일지 작성 챌린지"],
    "sugar_high": ["추가설탕 안녕이당 챌린지", "탄수화물 체인지 챌린지"],
    "carbohydrate_high": ["탄수화물 체인지 챌린지", "밥최고NO 채고밥YES 챌린지"],
    "fat_high": ["기름기 쫙빼기 챌린지", "건강식탁 챌린지"],
    "calorie_high": ["Goodbye 야식 챌린지", "Goodbye 폭식 챌린지", "2020 식사 챌린지"],
    "protein_support": ["건강식탁 챌린지", "철분 반찬 추가 챌린지"],
    "fiber_support": ["식이섬유 먹어유 챌린지", "건강식탁 챌린지"],
    "iron_support": ["철분 반찬 추가 챌린지", "비타민C 함께 먹기 챌린지", "철분 흡수 방해 식품 줄이기 챌린지"],
    "hydration_support": ["기상 직후 물 한 컵 챌린지", "한 시간마다 한 모금 물 마시기 챌린지"],
    "late_night_or_irregular": ["Goodbye 야식 챌린지", "일정한 삼시세끼 챌린지"],
    "alcohol_liver_support": ["30일, 금주 챌린지", "30일, 하루 두 잔 절주 챌린지", "30일 폭음 피하기 챌린지"],
    "kidney_caution": ["식사일지 작성 챌린지"],
    "balanced_support": ["건강식탁 챌린지", "식사일지 작성 챌린지"],
}

ISSUE_DEFINITIONS = {
    "sodium_high": {
        "type": "excess_candidate",
        "nutrient": "sodium_mg",
        "label": "나트륨 주의",
        "message": "이번 식단에는 나트륨이 높은 후보가 있어요. 국물이나 짠 소스는 조금 덜어내는 것부터 시작해 보세요.",
        "reason": "짠맛을 줄이는 식습관을 가볍게 시작하는 데 도움이 될 수 있습니다.",
    },
    "carbohydrate_high": {
        "type": "excess_candidate",
        "nutrient": "carbohydrate_g",
        "label": "탄수화물 주의",
        "message": "탄수화물 비중이 높은 후보가 보여요. 다음 식사는 반찬 구성을 곁들여 균형을 맞춰보세요.",
        "reason": "탄수화물 선택을 천천히 조절하는 습관에 도움이 될 수 있습니다.",
    },
    "sugar_high": {
        "type": "excess_candidate",
        "nutrient": "sugar_g",
        "label": "당류 주의",
        "message": "당류가 높은 후보가 포함됐을 수 있어요. 단 음료나 후식은 양을 줄여보면 좋습니다.",
        "reason": "당류를 줄이는 작은 실천과 연결됩니다.",
    },
    "fat_high": {
        "type": "excess_candidate",
        "nutrient": "fat_g",
        "label": "지방 주의",
        "message": "기름진 음식 후보가 보여요. 튀김보다 구이, 삶은 음식처럼 담백한 선택을 참고해 보세요.",
        "reason": "기름기를 줄이고 균형을 맞추는 습관에 도움이 될 수 있습니다.",
    },
    "calorie_high": {
        "type": "excess_candidate",
        "nutrient": "calories_kcal",
        "label": "열량 주의",
        "message": "열량이 높은 후보가 포함됐을 수 있어요. 다음 끼니는 채소와 단백질 반찬을 곁들여 균형을 맞춰보세요.",
        "reason": "과식과 야식을 줄이는 생활관리 습관에 도움이 될 수 있습니다.",
    },
    "protein_support": {
        "type": "support_candidate",
        "nutrient": "protein_g",
        "label": "단백질 보완",
        "message": "단백질 반찬을 조금 더 보완하면 식사 균형에 도움이 될 수 있습니다.",
        "reason": "두부, 계란, 생선처럼 부담 없는 반찬을 더하는 실천과 연결됩니다.",
    },
    "fiber_support": {
        "type": "support_candidate",
        "nutrient": "fiber_g",
        "label": "식이섬유 보완",
        "message": "식이섬유를 조금 더 보완하면 좋습니다. 평소 식사에서 부담 없이 더할 수 있는 반찬부터 살펴보세요.",
        "reason": "채소와 잡곡을 자연스럽게 늘리는 습관에 도움이 될 수 있습니다.",
    },
    "iron_support": {
        "type": "support_candidate",
        "nutrient": "iron_mg",
        "label": "철분 보완",
        "message": "철분과 비타민C가 있는 반찬을 함께 챙겨보면 좋습니다.",
        "reason": "철분이 있는 반찬을 식사에 더하는 실천과 연결됩니다.",
    },
    "late_night_or_irregular": {
        "type": "habit_candidate",
        "nutrient": None,
        "label": "식사 시간 주의",
        "message": "식사 시간이 늦거나 불규칙했다면 내일은 비슷한 시간에 한 끼를 챙기는 것부터 시작해 보세요.",
        "reason": "야식과 불규칙한 식사를 줄이는 생활관리 습관에 도움이 될 수 있습니다.",
    },
    "alcohol_liver_support": {
        "type": "habit_candidate",
        "nutrient": None,
        "label": "간 건강 식습관",
        "message": "간 건강 관리 관점에서는 음주, 당류, 기름진 안주를 함께 살펴보면 좋습니다.",
        "reason": "절주와 간 건강 생활관리 습관에 도움이 될 수 있습니다.",
    },
    "kidney_caution": {
        "type": "medical_caution",
        "nutrient": None,
        "label": "신장 관련 주의",
        "message": "신장 관련 수치가 걱정된다면 제한식을 단정하기보다 식사일지를 남기고 의료진과 상의해 보세요.",
        "reason": "상담 전 식사 기록을 준비하는 습관에 도움이 될 수 있습니다.",
    },
    "balanced_support": {
        "type": "support_candidate",
        "nutrient": None,
        "label": "균형 유지",
        "message": "이번 식단은 식사 기록과 균형 유지 관점에서 참고해 주세요. 지금처럼 기록을 이어가면 패턴을 보기 쉬워집니다.",
        "reason": "균형식과 꾸준한 식사 기록 습관에 도움이 될 수 있습니다.",
    },
}

DISEASE_RULES = {
    "HTN": {
        "analysis_types": {AnalysisType.HYPERTENSION},
        "label": "혈압 관리",
        "message": "혈압 관리가 필요한 경우 짠맛을 줄이는 식습관부터 참고해 보세요.",
        "issues": ("sodium_high",),
    },
    "DM": {
        "analysis_types": {AnalysisType.DIABETES},
        "label": "혈당 관리",
        "message": "혈당 관리가 필요한 경우 탄수화물과 당류 선택을 천천히 살펴보세요.",
        "issues": ("carbohydrate_high", "sugar_high"),
    },
    "DL": {
        "analysis_types": {AnalysisType.DYSLIPIDEMIA},
        "label": "콜레스테롤·중성지방 관리",
        "message": "지질 관리 관점에서는 기름진 음식은 줄이고 식이섬유를 보완하는 방향을 참고해 보세요.",
        "issues": ("fat_high", "fiber_support"),
    },
    "OBE": {
        "analysis_types": {AnalysisType.OBESITY, AnalysisType.ABDOMINAL_OBESITY},
        "label": "체중 관리",
        "message": "체중 관리 관점에서는 열량이 높은 음식과 야식 습관을 함께 살펴보세요.",
        "issues": ("calorie_high", "late_night_or_irregular"),
    },
    "ANEM": {
        "analysis_types": {AnalysisType.ANEMIA},
        "label": "빈혈 관리",
        "message": "빈혈 관리 관점에서는 철분과 단백질 반찬을 보완하는 식사를 참고해 보세요.",
        "issues": ("iron_support", "protein_support"),
    },
    "FL": {
        "analysis_types": {AnalysisType.FATTY_LIVER, AnalysisType.LIVER_FUNCTION},
        "label": "간 건강 관리",
        "message": "간 건강 관리 관점에서는 음주, 당류, 기름진 음식과 늦은 식사를 함께 살펴보세요.",
        "issues": ("alcohol_liver_support", "sugar_high", "fat_high", "calorie_high"),
    },
    "CKD": {
        "analysis_types": {AnalysisType.KIDNEY_FUNCTION, AnalysisType.CHRONIC_KIDNEY_DISEASE},
        "label": "신장 관리",
        "message": "신장 관련 식사는 개인 수치에 따라 달라질 수 있어 식사일지를 바탕으로 의료진과 상의해 보세요.",
        "issues": ("kidney_caution",),
    },
}

FOOD_RECOMMENDATIONS = {
    "sodium_high": (("채소 반찬", "단백질 반찬"), ("국물 많은 음식", "짠 소스")),
    "carbohydrate_high": (("잡곡밥", "채소 반찬", "단백질 반찬"), ("흰밥이 많은 한 그릇 음식", "달콤한 소스")),
    "sugar_high": (("무가당 음료", "채소 반찬"), ("단 음료", "달콤한 후식")),
    "fat_high": (("채소 반찬", "담백한 단백질 반찬"), ("튀김류", "기름진 소스")),
    "calorie_high": (("채소 반찬", "단백질 반찬"), ("야식", "과식하기 쉬운 메뉴")),
    "protein_support": (("단백질 반찬", "두부·계란·생선류"), ()),
    "fiber_support": (("채소 반찬", "잡곡밥", "식이섬유가 있는 반찬"), ()),
    "iron_support": (("철분이 있는 반찬", "비타민C가 있는 과일"), ("식사 직후 진한 차·커피는 줄여보면 좋습니다",)),
    "late_night_or_irregular": (("규칙적인 식사", "가벼운 저녁 구성"), ("늦은 야식",)),
    "alcohol_liver_support": (("물", "담백한 단백질 반찬"), ("음주", "기름진 안주")),
    "kidney_caution": (("식사일지", "조리법 기록", "상담 전 식단 기록"), ("개인 수치 확인 전 보충제 섭취",)),
    "balanced_support": (("채소 반찬", "단백질 반찬", "잡곡밥"), ()),
}

CKD_CONSERVATIVE_RECOMMENDATIONS = (
    "식사일지",
    "조리법 기록",
    "국물 줄이기",
    "짠 소스 덜어내기",
    "상담 전 식단 기록",
)
CKD_GENERAL_FOOD_EXCLUSIONS = {
    "잡곡밥",
    "해조류 반찬",
    "채소 반찬",
    "단백질 반찬",
    "담백한 단백질 반찬",
    "두부·계란·생선류",
    "고단백 식품",
    "고단백 보충제",
}
CKD_BLOCKED_CHALLENGE_TITLES = {"식이섬유 먹어유 챌린지", "건강식탁 챌린지"}
CKD_FINDING_MESSAGE = (
    "신장 관련 식사는 개인 수치에 따라 달라질 수 있어요. "
    "식사일지를 남기고 검사 수치를 바탕으로 의료진과 함께 조정해 보세요."
)
CKD_NUTRIENT_SUPPORT_MESSAGE = (
    "식이섬유나 단백질 보완도 개인 수치에 따라 달라질 수 있어 식사 기록을 바탕으로 상담해 보세요."
)

RAG_COMMENT_TEMPLATES = {
    "HTN": (
        "혈압 관리가 필요한 경우 저염 식습관을 먼저 참고해 보세요. "
        "이번 식단에서 국물이나 짠 소스는 조금 덜어내는 것부터 시작해 볼 수 있습니다."
    ),
    "DM": (
        "혈당 관리가 필요한 경우 탄수화물과 당류 섭취 패턴을 함께 살펴보면 좋습니다. "
        "실제 섭취량이 확정되지 않아 참고용으로 봐 주세요."
    ),
    "DL": ("지질 관리 관점에서는 기름진 음식은 조금 줄이고 식이섬유가 있는 식품을 보완해 보세요."),
    "OBE": "체중 관리 관점에서는 열량이 높은 후보와 야식 패턴을 함께 살펴보면 좋습니다.",
    "CKD": (
        "신장 관련 식사는 개인 상태에 따라 달라질 수 있어 의료진과 상의해 보세요. "
        "식사일지를 기록해 상담 시 참고자료로 활용해 보세요."
    ),
}


async def get_diet_health_recommendations(user_id: int, diet_record_id: int) -> dict[str, Any]:
    record = await diet_service.get_diet_record(diet_record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="식단 기록을 찾을 수 없습니다.")
    if int(record.user_id) != int(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="해당 리소스에 접근할 권한이 없습니다.")

    health_record = await health_repository.get_latest_health_record_by_user(user_id)
    analysis_results = await analysis_repository.list_analysis_results_by_user(user_id, limit=20)
    active_challenges = await challenge_service.list_active_challenges(limit=500)
    return await build_diet_health_recommendation_response_async(
        diet_record=record,
        analysis_results=analysis_results,
        health_record=health_record,
        active_challenges=active_challenges,
    )


def build_diet_health_recommendation_response(
    *,
    diet_record: Any,
    analysis_results: Iterable[Any],
    health_record: Any | None,
    active_challenges: Iterable[Any],
) -> dict[str, Any]:
    response, issue_keys, rag_disease_codes = _build_diet_health_recommendation_base(
        diet_record=diet_record,
        analysis_results=analysis_results,
        health_record=health_record,
        active_challenges=active_challenges,
    )
    rag_comment = _safe_build_rag_comment(issue_keys=issue_keys, rag_disease_codes=rag_disease_codes)
    return _finalize_diet_health_recommendation_response(response=response, rag_comment=rag_comment)


def build_diet_health_recommendation_context_response(
    *,
    diet_record: Any,
    analysis_results: Iterable[Any],
    health_record: Any | None,
    active_challenges: Iterable[Any],
) -> dict[str, Any]:
    response, issue_keys, rag_disease_codes = _build_diet_health_recommendation_base(
        diet_record=diet_record,
        analysis_results=analysis_results,
        health_record=health_record,
        active_challenges=active_challenges,
    )
    response["rag_comment"] = _safe_build_rag_comment(issue_keys=issue_keys, rag_disease_codes=rag_disease_codes)
    return response


async def build_diet_health_recommendation_response_async(
    *,
    diet_record: Any,
    analysis_results: Iterable[Any],
    health_record: Any | None,
    active_challenges: Iterable[Any],
    vector_retriever: Any | None = None,
) -> dict[str, Any]:
    response, issue_keys, rag_disease_codes = _build_diet_health_recommendation_base(
        diet_record=diet_record,
        analysis_results=analysis_results,
        health_record=health_record,
        active_challenges=active_challenges,
    )
    rag_comment = await _safe_build_rag_comment_async(
        issue_keys=issue_keys,
        rag_disease_codes=rag_disease_codes,
        vector_retriever=vector_retriever,
    )
    return await _finalize_diet_health_recommendation_response_async(response=response, rag_comment=rag_comment)


def _build_diet_health_recommendation_base(
    *,
    diet_record: Any,
    analysis_results: Iterable[Any],
    health_record: Any | None,
    active_challenges: Iterable[Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    analysis_results = list(analysis_results)
    detected_foods = _detected_foods_as_list(getattr(diet_record, "detected_foods", None))
    nutrients_by_food = [_extract_food_nutrition(food) for food in detected_foods]
    disease_codes = _disease_codes_from_context(analysis_results, health_record)
    issue_keys = _rank_issue_keys(
        nutrition_issues=_nutrition_issue_keys(nutrients_by_food, diet_record),
        disease_codes=disease_codes,
    )
    if not issue_keys and not disease_codes:
        issue_keys = ["balanced_support"]

    basis_by_issue = _basis_by_issue(nutrients_by_food, issue_keys)
    nutrition_findings = [
        _finding_payload(issue_key, basis_by_issue.get(issue_key), disease_codes=disease_codes)
        for issue_key in issue_keys
        if issue_key in ISSUE_DEFINITIONS
    ]
    disease_context = [_disease_context_payload(code) for code in disease_codes if code in DISEASE_RULES]
    recommended_foods, caution_foods = _food_messages(issue_keys, disease_codes=disease_codes)
    recommended_challenges = _recommended_challenges(issue_keys, active_challenges, disease_codes=disease_codes)
    rag_disease_codes = _rag_disease_codes_from_analysis_results(analysis_results)
    response = {
        "diet_record_id": int(diet_record.id),
        "nutrition_findings": nutrition_findings,
        "disease_context": disease_context,
        "recommended_foods": recommended_foods,
        "caution_foods": caution_foods,
        "recommended_challenges": recommended_challenges,
        "safety_notice": SAFETY_NOTICE,
    }
    return response, issue_keys, rag_disease_codes


def _finalize_diet_health_recommendation_response(
    *, response: dict[str, Any], rag_comment: dict[str, Any]
) -> dict[str, Any]:
    response["rag_comment"] = rag_comment
    response["rag_comment"] = rewrite_diet_rag_comment(
        rag_comment=rag_comment,
        recommendation_payload=response,
        use_real_llm=config.DIET_RECOMMENDATION_LLM_REWRITE_ENABLED and has_openai_config(config),
    )
    return response


async def _finalize_diet_health_recommendation_response_async(
    *, response: dict[str, Any], rag_comment: dict[str, Any]
) -> dict[str, Any]:
    response["rag_comment"] = rag_comment
    use_real_llm = config.DIET_RECOMMENDATION_LLM_REWRITE_ENABLED and has_openai_config(config)
    if use_real_llm:
        response["rag_comment"] = await asyncio.to_thread(
            rewrite_diet_rag_comment,
            rag_comment=rag_comment,
            recommendation_payload=response,
            use_real_llm=True,
        )
    else:
        response["rag_comment"] = rewrite_diet_rag_comment(
            rag_comment=rag_comment,
            recommendation_payload=response,
            use_real_llm=False,
        )
    return response


def _detected_foods_as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("foods", "detected_foods", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [value]
    return []


def _extract_food_nutrition(food: dict[str, Any]) -> dict[str, Any]:
    metadata = food.get("match_metadata") if isinstance(food.get("match_metadata"), dict) else {}
    nutrition = metadata.get("nutrition") if isinstance(metadata.get("nutrition"), dict) else food.get("nutrition")
    if not isinstance(nutrition, dict):
        nutrition = {}
    values = {canonical: _first_float(nutrition, aliases) for canonical, aliases in NUTRIENT_ALIASES.items()}
    return {
        "food_name": _food_display_name(food),
        "values": {key: value for key, value in values.items() if value is not None},
        "basis": _nutrition_basis(nutrition),
    }


def _first_float(source: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    normalized = {_normalize_key(key): value for key, value in source.items()}
    for alias in aliases:
        value = normalized.get(_normalize_key(alias))
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _normalize_key(value: object) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", "", str(value).lower())


def _to_float(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if value is None:
        return None
    matched = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not matched:
        return None
    try:
        return float(matched.group(0))
    except ValueError:
        return None


def _nutrition_basis(nutrition: dict[str, Any]) -> str:
    basis_label = str(nutrition.get("basis_label") or "").strip()
    if basis_label:
        return basis_label
    amount = nutrition.get("basis_amount")
    unit = str(nutrition.get("basis_unit") or "").strip()
    if amount not in {None, ""} and unit:
        return f"{amount}{unit} 기준"
    for key in ("serving_size", "serving_reference", "food_weight"):
        value = str(nutrition.get(key) or "").strip()
        if value:
            return value
    return "식품영양성분 데이터 기준 후보"


def _food_display_name(food: dict[str, Any]) -> str:
    return str(food.get("original_name") or food.get("query_name") or food.get("name") or "").strip()


def _nutrition_issue_keys(nutrients_by_food: list[dict[str, Any]], diet_record: Any) -> list[str]:
    values = [item["values"] for item in nutrients_by_food if item.get("values")]
    issues: list[str] = []
    if any(value.get("sodium_mg", 0) >= 600 for value in values):
        issues.append("sodium_high")
    if any(value.get("carbohydrate_g", 0) >= 55 for value in values):
        issues.append("carbohydrate_high")
    if any(value.get("sugar_g", 0) >= 15 for value in values):
        issues.append("sugar_high")
    if any(value.get("fat_g", 0) >= 20 or value.get("saturated_fat_g", 0) >= 8 for value in values):
        issues.append("fat_high")
    if any(value.get("calories_kcal", 0) >= 500 for value in values):
        issues.append("calorie_high")
    if values and max((value.get("fiber_g", 0) for value in values), default=0) < 3:
        issues.append("fiber_support")
    if _looks_late_or_irregular(diet_record):
        issues.append("late_night_or_irregular")
    if _looks_like_alcohol(nutrients_by_food, diet_record):
        issues.append("alcohol_liver_support")
    return _dedupe(issues)


def _looks_late_or_irregular(diet_record: Any) -> bool:
    text = " ".join(str(getattr(diet_record, key, "") or "") for key in ("meal_type", "description", "memo"))
    return bool(re.search(r"야식|늦은|불규칙|late|night", text, flags=re.IGNORECASE))


def _looks_like_alcohol(nutrients_by_food: list[dict[str, Any]], diet_record: Any) -> bool:
    names = " ".join(str(item.get("food_name") or "") for item in nutrients_by_food)
    text = " ".join(
        [names, str(getattr(diet_record, "description", "") or ""), str(getattr(diet_record, "memo", "") or "")]
    )
    return bool(re.search(r"술|소주|맥주|와인|막걸리|음주|폭음|alcohol|beer|wine", text, flags=re.IGNORECASE))


def _disease_codes_from_context(analysis_results: Iterable[Any], health_record: Any | None) -> list[str]:
    codes: list[str] = []
    latest_by_type: dict[AnalysisType, Any] = {}
    for result in analysis_results:
        analysis_type = _analysis_type(getattr(result, "analysis_type", None))
        if analysis_type is not None and analysis_type not in latest_by_type:
            latest_by_type[analysis_type] = result

    for code, rule in DISEASE_RULES.items():
        if any(_is_relevant_result(latest_by_type.get(analysis_type)) for analysis_type in rule["analysis_types"]):
            codes.append(code)

    if health_record is not None:
        if bool(getattr(health_record, "has_hypertension", False)):
            codes.append("HTN")
        if bool(getattr(health_record, "has_diabetes", False)):
            codes.append("DM")
        if bool(getattr(health_record, "has_dyslipidemia", False)):
            codes.append("DL")
        if bool(getattr(health_record, "has_obesity", False)):
            codes.append("OBE")
    return _dedupe(codes)


def _analysis_type(value: Any) -> AnalysisType | None:
    try:
        return value if isinstance(value, AnalysisType) else AnalysisType(str(value))
    except ValueError:
        return None


def _is_relevant_result(result: Any | None) -> bool:
    if result is None:
        return False
    risk = str(getattr(result, "risk_level", "") or "").upper()
    return risk not in {"", RiskLevel.LOW.value}


def _rag_disease_codes_from_analysis_results(analysis_results: Iterable[Any]) -> list[str]:
    latest_by_type: dict[AnalysisType, Any] = {}
    for result in analysis_results:
        analysis_type = _analysis_type(getattr(result, "analysis_type", None))
        if analysis_type is not None and analysis_type not in latest_by_type:
            latest_by_type[analysis_type] = result

    codes: list[str] = []
    for code, rule in DISEASE_RULES.items():
        if any(_is_rag_relevant_result(latest_by_type.get(analysis_type)) for analysis_type in rule["analysis_types"]):
            codes.append(code)
    return _dedupe(codes)


def _is_rag_relevant_result(result: Any | None) -> bool:
    if result is None:
        return False
    risk = _normalize_risk_level(getattr(result, "risk_level", None))
    if not risk:
        return False
    low_or_normal = {"LOW", "NORMAL", "GOOD", "OK", "NONE", "정상", "양호", "낮음", "낮은위험"}
    return risk not in low_or_normal


def _rank_issue_keys(*, nutrition_issues: list[str], disease_codes: list[str]) -> list[str]:
    ranked: list[str] = []
    for disease_code in disease_codes:
        for issue_key in DISEASE_RULES.get(disease_code, {}).get("issues", ()):
            if issue_key in nutrition_issues or issue_key in {"protein_support", "iron_support", "kidney_caution"}:
                ranked.append(issue_key)
    ranked.extend(nutrition_issues)
    return _dedupe(ranked)[:MAX_FINDINGS]


def _basis_by_issue(nutrients_by_food: list[dict[str, Any]], issue_keys: list[str]) -> dict[str, str]:
    basis: dict[str, str] = {}
    for issue_key in issue_keys:
        nutrient = ISSUE_DEFINITIONS.get(issue_key, {}).get("nutrient")
        for food in nutrients_by_food:
            if nutrient is None or nutrient in food.get("values", {}):
                basis[issue_key] = str(food.get("basis") or "식품영양성분 데이터 기준 후보")
                break
    return basis


def _finding_payload(issue_key: str, basis: str | None, *, disease_codes: list[str] | None = None) -> dict[str, Any]:
    definition = ISSUE_DEFINITIONS[issue_key]
    message = str(definition["message"])
    if "CKD" in set(disease_codes or []):
        message = (
            CKD_NUTRIENT_SUPPORT_MESSAGE if issue_key in {"fiber_support", "protein_support"} else CKD_FINDING_MESSAGE
        )
    return {
        "type": definition["type"],
        "issue_key": issue_key,
        "nutrient": definition["nutrient"],
        "label": definition["label"],
        "message": message,
        "basis": basis or "식품영양성분 데이터 기준 후보",
    }


def _disease_context_payload(code: str) -> dict[str, str]:
    rule = DISEASE_RULES[code]
    return {
        "disease_code": code,
        "label": rule["label"],
        "message": rule["message"],
    }


def _food_messages(issue_keys: list[str], *, disease_codes: list[str] | None = None) -> tuple[list[str], list[str]]:
    recommended: list[str] = []
    caution: list[str] = []
    for issue_key in issue_keys:
        foods, cautions = FOOD_RECOMMENDATIONS.get(issue_key, ((), ()))
        recommended.extend(_as_text_items(foods))
        caution.extend(_as_text_items(cautions))
    if "CKD" in set(disease_codes or []):
        recommended = [item for item in recommended if item not in CKD_GENERAL_FOOD_EXCLUSIONS and "고단백" not in item]
        recommended = list(CKD_CONSERVATIVE_RECOMMENDATIONS) + recommended
    if "HTN" in set(disease_codes or []) or "sodium_high" in issue_keys:
        recommended = ["국물 적은 반찬" if item == "해조류 반찬" else item for item in recommended]
    return _dedupe(recommended)[:5], _dedupe(caution)[:5]


def _as_text_items(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return []


def _recommended_challenges(
    issue_keys: list[str],
    active_challenges: Iterable[Any],
    *,
    disease_codes: list[str] | None = None,
) -> list[dict[str, Any]]:
    active_by_title = {str(challenge.title): challenge for challenge in active_challenges}
    candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_titles: set[str] = set()
    for issue_key in _challenge_issue_keys(issue_keys=issue_keys, disease_codes=disease_codes or []):
        for title in DIET_CHALLENGE_RULES.get(issue_key, ()):
            challenge = active_by_title.get(title)
            if challenge is None:
                continue
            challenge_id = int(challenge.id)
            challenge_title = str(challenge.title)
            if "CKD" in set(disease_codes or []) and challenge_title in CKD_BLOCKED_CHALLENGE_TITLES:
                continue
            if challenge_id in seen_ids or challenge_title in seen_titles:
                continue
            seen_ids.add(challenge_id)
            seen_titles.add(challenge_title)
            candidates.append(
                {
                    "challenge_id": challenge_id,
                    "title": challenge_title,
                    "reason": _challenge_reason(issue_key=issue_key, title=challenge_title),
                    "purpose": _challenge_purpose(issue_key=issue_key, title=challenge_title),
                }
            )
    diversified: list[dict[str, Any]] = []
    seen_purposes: set[str] = set()
    seen_reasons: set[str] = set()
    for candidate in candidates:
        if candidate["purpose"] in seen_purposes or candidate["reason"] in seen_reasons:
            continue
        diversified.append(candidate)
        seen_purposes.add(candidate["purpose"])
        seen_reasons.add(candidate["reason"])
        if len(diversified) >= MAX_RECOMMENDED_CHALLENGES:
            return [_public_challenge_payload(item) for item in diversified]
    for candidate in candidates:
        if candidate in diversified or candidate["reason"] in seen_reasons:
            continue
        diversified.append(candidate)
        seen_reasons.add(candidate["reason"])
        if len(diversified) >= MAX_RECOMMENDED_CHALLENGES:
            break
    return [_public_challenge_payload(item) for item in diversified]


def _challenge_issue_keys(*, issue_keys: list[str], disease_codes: list[str]) -> list[str]:
    disease_primary_issues = {
        "HTN": ("sodium_high",),
        "DM": ("carbohydrate_high",),
        "DL": ("fat_high", "fiber_support"),
        "OBE": ("calorie_high",),
        "ANEM": ("iron_support",),
        "FL": ("alcohol_liver_support", "fat_high"),
        "CKD": ("kidney_caution",),
    }
    prioritized: list[str] = []
    for disease_code in disease_codes:
        prioritized.extend(disease_primary_issues.get(disease_code, ()))
    prioritized.extend(issue_keys)
    return _dedupe(prioritized)


def _challenge_purpose(*, issue_key: str, title: str) -> str:
    if "식사일지" in title or "삼시세끼" in title:
        return "meal_record"
    if "염분" in title:
        return "sodium"
    if "설탕" in title or "탄수화물" in title or "채고밥" in title:
        return "sugar_carb"
    if "야식" in title or "폭식" in title or "2020" in title:
        return "calorie_routine"
    if "기름" in title:
        return "fat"
    if "식이섬유" in title or "건강식탁" in title:
        return "fiber_balance"
    if "철분" in title or "비타민C" in title:
        return "iron_support"
    if "금주" in title or "절주" in title or "폭음" in title:
        return "alcohol_liver"
    return issue_key


def _challenge_reason(*, issue_key: str, title: str) -> str:
    if "식사일지" in title:
        return "식사와 조리법을 기록하면 나에게 맞는 관리 포인트를 상담할 때 정리하기 좋습니다."
    if "염분" in title:
        return "국물과 짠 소스를 조금 덜어내는 연습에 도움이 될 수 있습니다."
    if "설탕" in title:
        return "단 음료나 후식을 줄이는 작은 실천으로 시작해 볼 수 있습니다."
    if "탄수화물" in title or "채고밥" in title:
        return "밥과 면의 양을 살펴보고 채소나 단백질 반찬을 곁들이는 데 도움이 될 수 있습니다."
    if "야식" in title:
        return "늦은 식사 패턴을 줄이고 저녁 구성을 가볍게 정리하는 데 도움이 될 수 있습니다."
    if "폭식" in title or "2020" in title:
        return "식사량과 식사 시간을 천천히 조절하는 습관과 연결됩니다."
    if "기름" in title:
        return "튀김이나 기름진 소스를 줄이고 담백한 조리법을 고르는 데 도움이 될 수 있습니다."
    if "식이섬유" in title or "건강식탁" in title:
        return "식사 균형을 살피고 부족하기 쉬운 반찬을 보완하는 데 도움이 될 수 있습니다."
    if "철분" in title:
        return "철분이 있는 반찬을 식사에 자연스럽게 더하는 연습과 연결됩니다."
    if "비타민C" in title:
        return "철분이 있는 반찬과 함께 과일을 곁들이는 식사 구성을 참고해 볼 수 있습니다."
    if "금주" in title or "절주" in title or "폭음" in title:
        return "음주 빈도와 양을 기록하고 줄여보는 생활관리 실천과 연결됩니다."
    return str(
        ISSUE_DEFINITIONS.get(issue_key, {}).get("reason")
        or "식단 관리 습관을 가볍게 시작하는 데 도움이 될 수 있습니다."
    )


def _public_challenge_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "challenge_id": candidate["challenge_id"],
        "title": candidate["title"],
        "reason": candidate["reason"],
    }


def _safe_build_rag_comment(*, issue_keys: list[str], rag_disease_codes: list[str]) -> dict[str, Any]:
    try:
        return _build_rag_comment(issue_keys=issue_keys, rag_disease_codes=rag_disease_codes)
    except Exception:
        return {
            "enabled": False,
            "fallback_used": True,
            "summary": RAG_FALLBACK_SUMMARY,
            "disease_comments": [],
            "evidence_sources": [],
            "safety_notice": SAFETY_NOTICE,
        }


async def _safe_build_rag_comment_async(
    *,
    issue_keys: list[str],
    rag_disease_codes: list[str],
    vector_retriever: Any | None = None,
) -> dict[str, Any]:
    try:
        return await _build_rag_comment_async(
            issue_keys=issue_keys,
            rag_disease_codes=rag_disease_codes,
            vector_retriever=vector_retriever,
        )
    except Exception:
        return {
            "enabled": False,
            "fallback_used": True,
            "summary": RAG_FALLBACK_SUMMARY,
            "disease_comments": [],
            "evidence_sources": [],
            "safety_notice": SAFETY_NOTICE,
        }


def _build_rag_comment(*, issue_keys: list[str], rag_disease_codes: list[str]) -> dict[str, Any]:
    if not config.RAG_ENABLED:
        return {
            "enabled": False,
            "fallback_used": True,
            "summary": RAG_FALLBACK_SUMMARY,
            "disease_comments": [],
            "evidence_sources": [],
            "safety_notice": SAFETY_NOTICE,
        }
    target_codes = _resolve_rag_target_codes(issue_keys=issue_keys, rag_disease_codes=rag_disease_codes)
    if not target_codes:
        return {
            "enabled": False,
            "fallback_used": True,
            "summary": "이번 식단은 기본 추천을 먼저 참고해 보세요. 식사 기록이 쌓이면 더 잘 맞는 생활관리 포인트를 살펴볼 수 있습니다.",
            "disease_comments": [],
            "evidence_sources": [],
            "safety_notice": SAFETY_NOTICE,
        }

    matches = _retrieve_diet_rag_matches(issue_keys=issue_keys, target_codes=target_codes)
    evidence_sources = _evidence_sources(matches)
    return _build_rag_comment_from_evidence(
        rag_disease_codes=rag_disease_codes,
        target_codes=target_codes,
        evidence_sources=evidence_sources,
    )


async def _build_rag_comment_async(
    *,
    issue_keys: list[str],
    rag_disease_codes: list[str],
    vector_retriever: Any | None = None,
) -> dict[str, Any]:
    if not config.RAG_ENABLED:
        return {
            "enabled": False,
            "fallback_used": True,
            "summary": RAG_FALLBACK_SUMMARY,
            "disease_comments": [],
            "evidence_sources": [],
            "safety_notice": SAFETY_NOTICE,
        }
    target_codes = _resolve_rag_target_codes(issue_keys=issue_keys, rag_disease_codes=rag_disease_codes)
    if not target_codes:
        return {
            "enabled": False,
            "fallback_used": True,
            "summary": "이번 식단은 기본 추천을 먼저 참고해 보세요. 식사 기록이 쌓이면 더 잘 맞는 생활관리 포인트를 살펴볼 수 있습니다.",
            "disease_comments": [],
            "evidence_sources": [],
            "safety_notice": SAFETY_NOTICE,
        }

    started_at = time.monotonic()
    strategy = _normalized_diet_rag_strategy()
    matches = _retrieve_diet_rag_matches(issue_keys=issue_keys, target_codes=target_codes)
    evidence_sources = _evidence_sources(matches)
    vector_documents: list[RetrievedDocument] = []
    fallback_reason: str | None = None
    fallback_used = False

    if _should_try_vector_fallback(strategy=strategy, keyword_count=len(matches)):
        vector_retriever = vector_retriever or _build_diet_vector_retriever()
        if vector_retriever is None:
            fallback_reason = "vector_unavailable"
        else:
            fallback_used = True
            try:
                vector_documents = await _retrieve_diet_vector_documents(
                    issue_keys=issue_keys,
                    target_codes=target_codes,
                    vector_retriever=vector_retriever,
                )
            except Exception as exc:  # noqa: BLE001 - RAG fallback must not break diet recommendations.
                fallback_reason = f"vector_failed:{type(exc).__name__}"
                vector_documents = []
            if vector_documents:
                evidence_sources = _evidence_sources_from_documents(vector_documents, target_codes=target_codes)
                fallback_reason = (
                    "keyword_empty_vector_used" if len(matches) == 0 else "keyword_insufficient_vector_used"
                )
            elif fallback_reason is None:
                fallback_reason = "vector_no_result"
    else:
        fallback_reason = _strategy_skip_reason(strategy=strategy, keyword_count=len(matches))

    _trace_diet_rag_strategy(
        strategy=strategy,
        keyword_returned_count=len(matches),
        vector_documents=vector_documents,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        target_codes=target_codes,
        issue_keys=issue_keys,
        latency_ms=round((time.monotonic() - started_at) * 1000, 3),
    )
    return _build_rag_comment_from_evidence(
        rag_disease_codes=rag_disease_codes,
        target_codes=target_codes,
        evidence_sources=evidence_sources,
    )


def _build_rag_comment_from_evidence(
    *,
    rag_disease_codes: list[str],
    target_codes: list[str],
    evidence_sources: list[dict[str, str]],
) -> dict[str, Any]:
    if not evidence_sources:
        return {
            "enabled": True,
            "fallback_used": True,
            "summary": RAG_FALLBACK_SUMMARY,
            "disease_comments": [],
            "evidence_sources": [],
            "safety_notice": SAFETY_NOTICE,
        }

    disease_comments = _rag_disease_comments(
        rag_disease_codes=rag_disease_codes,
        target_codes=target_codes,
        evidence_sources=evidence_sources,
    )
    summary = "이번 식단에서 보완하거나 줄여볼 포인트를 생활관리 관점으로 정리했어요."
    if not disease_comments:
        summary = (
            "이번 식단은 식생활 균형을 보완하는 쪽으로 참고해 보세요. "
            "채소 반찬이나 잡곡밥처럼 부담 없이 더할 수 있는 선택부터 시작해 보세요."
        )
    return {
        "enabled": True,
        "fallback_used": False,
        "summary": summary,
        "disease_comments": disease_comments,
        "evidence_sources": evidence_sources,
        "safety_notice": SAFETY_NOTICE,
    }


def _resolve_rag_target_codes(*, issue_keys: list[str], rag_disease_codes: list[str]) -> list[str]:
    enabled_codes = enabled_rag_codes()
    relevant_disease_codes = set(rag_disease_codes)
    candidates: list[str] = []
    for issue_key in issue_keys:
        for code in ISSUE_TO_RAG_CODES.get(issue_key, ()):
            if code in GENERAL_RAG_CODES or code in relevant_disease_codes:
                candidates.append(code)
    return _dedupe([code for code in candidates if code in enabled_codes])


def _retrieve_diet_rag_matches(*, issue_keys: list[str], target_codes: list[str]) -> list[KeywordRagMatch]:
    matches_by_source_id: dict[str, KeywordRagMatch] = {}
    for code in target_codes:
        query = DIET_RAG_QUERY_TEMPLATES.get(code, " ".join(issue_keys))
        for match in retrieve_keyword_rag_matches(
            user_message=query,
            disease_code=code,
            issue_keys=issue_keys,
            top_k=2,
            include_safety_disclaimer=False,
        ):
            if match.document.metadata.disease_code not in target_codes:
                continue
            existing = matches_by_source_id.get(match.source_id)
            if existing is None or match.score > existing.score:
                matches_by_source_id[match.source_id] = match
    return sorted(matches_by_source_id.values(), key=lambda match: (-match.score, match.source_id))[
        :MAX_RAG_EVIDENCE_SOURCES
    ]


async def _retrieve_diet_vector_documents(
    *,
    issue_keys: list[str],
    target_codes: list[str],
    vector_retriever: Any,
) -> list[RetrievedDocument]:
    documents_by_key: dict[str, RetrievedDocument] = {}
    for code in target_codes:
        query = DIET_RAG_QUERY_TEMPLATES.get(code, " ".join(issue_keys))
        result = await vector_retriever.retrieve(
            query=query,
            disease_code=code,
            issue_keys=issue_keys,
            top_k=2,
            include_safety_disclaimer=False,
        )
        for document in getattr(result, "documents", []) or []:
            metadata = document.metadata if isinstance(document.metadata, dict) else {}
            disease_code = str(metadata.get("disease_code") or "")
            if disease_code not in target_codes:
                continue
            key = str(metadata.get("chunk_key") or metadata.get("id") or f"{document.title}:{document.url}")
            if key not in documents_by_key:
                documents_by_key[key] = document
            if len(documents_by_key) >= MAX_RAG_EVIDENCE_SOURCES:
                return list(documents_by_key.values())
    return list(documents_by_key.values())[:MAX_RAG_EVIDENCE_SOURCES]


def _evidence_sources(matches: list[KeywordRagMatch]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for match in matches[:MAX_RAG_EVIDENCE_SOURCES]:
        metadata = match.document.metadata
        sources.append(
            {
                "title": _public_source_title(metadata.title),
                "disease_code": metadata.disease_code,
                "review_status": _public_review_status(metadata.review_status),
            }
        )
    return sources


def _evidence_sources_from_documents(
    documents: list[RetrievedDocument],
    *,
    target_codes: list[str],
) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for document in documents[:MAX_RAG_EVIDENCE_SOURCES]:
        metadata = document.metadata if isinstance(document.metadata, dict) else {}
        disease_code = str(metadata.get("disease_code") or "")
        if disease_code not in target_codes:
            continue
        title = _public_source_title(document.title or metadata.get("title") or disease_code)
        key = (title, disease_code)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": title,
                "disease_code": disease_code,
                "review_status": _public_review_status(metadata),
            }
        )
    return sources


def _public_review_status(value: Any) -> str:
    if isinstance(value, dict):
        raw_status = str(value.get("review_status") or value.get("status") or "reference")
    else:
        raw_status = str(value or "reference")
    if raw_status == "candidate_unreviewed":
        return "reference"
    if raw_status == "missing_source":
        return "unavailable"
    if raw_status == "approved":
        return "approved"
    if raw_status == "reviewed":
        return "reviewed"
    return raw_status


def _public_source_title(value: Any) -> str:
    title = str(value or "").strip()
    title = title.replace(" 후보 지식", "").replace("후보 지식", "").strip()
    return title or "참고 문서"


def _rag_disease_comments(
    *,
    rag_disease_codes: list[str],
    target_codes: list[str],
    evidence_sources: list[dict[str, str]],
) -> list[dict[str, str]]:
    evidence_codes = {source["disease_code"] for source in evidence_sources}
    comments: list[dict[str, str]] = []
    for disease_code in rag_disease_codes:
        if disease_code not in RAG_COMMENT_TEMPLATES:
            continue
        has_direct_evidence = disease_code in evidence_codes
        has_caution_fallback = (
            disease_code == "CKD" and "DIET_CAUTION" in evidence_codes and "DIET_CAUTION" in target_codes
        )
        if not has_direct_evidence and not has_caution_fallback:
            continue
        rule = DISEASE_RULES.get(disease_code, {})
        comments.append(
            {
                "disease_code": disease_code,
                "label": str(rule.get("label") or disease_code),
                "comment": RAG_COMMENT_TEMPLATES[disease_code],
                "basis": _basis_from_evidence(disease_code=disease_code, evidence_sources=evidence_sources),
            }
        )
    return comments


def _basis_from_evidence(*, disease_code: str, evidence_sources: list[dict[str, str]]) -> str:
    titles = [
        source["title"]
        for source in evidence_sources
        if source["disease_code"] == disease_code
        or (disease_code == "CKD" and source["disease_code"] == "DIET_CAUTION")
    ]
    if not titles:
        return "생활관리 참고 자료 기반"
    return f"관련 참고: {', '.join(titles[:2])}"


def _normalized_diet_rag_strategy() -> str:
    strategy = str(getattr(config, "DIET_RECOMMENDATION_RAG_STRATEGY", DIET_RAG_STRATEGY_KEYWORD_ONLY) or "").strip()
    if strategy not in SUPPORTED_DIET_RAG_STRATEGIES:
        return DIET_RAG_STRATEGY_KEYWORD_ONLY
    return strategy


def _should_try_vector_fallback(*, strategy: str, keyword_count: int) -> bool:
    return (
        strategy == DIET_RAG_STRATEGY_KEYWORD_FIRST_VECTOR_FALLBACK
        and keyword_count < DIET_RAG_MIN_KEYWORD_RESULTS
        and _diet_vector_gate_enabled()
    )


def _diet_vector_gate_enabled() -> bool:
    return (
        bool(getattr(config, "RAG_EMBEDDING_ENABLED", False))
        and str(getattr(config, "RAG_EMBEDDING_PROVIDER", "") or "").strip().lower() == "openai"
        and has_openai_config(config)
    )


def _strategy_skip_reason(*, strategy: str, keyword_count: int) -> str | None:
    if strategy == DIET_RAG_STRATEGY_VECTOR_DISABLED:
        return "vector_disabled"
    if strategy == DIET_RAG_STRATEGY_KEYWORD_ONLY:
        return "keyword_only"
    if keyword_count >= DIET_RAG_MIN_KEYWORD_RESULTS:
        return "keyword_result_sufficient"
    if not _diet_vector_gate_enabled():
        return "vector_gate_disabled"
    return None


def _build_diet_vector_retriever() -> VectorRagRetriever | None:
    try:
        provider = get_embedding_provider(config)
    except Exception:
        return None
    if provider is None:
        return None
    return VectorRagRetriever(embedding_provider=provider)


def _trace_diet_rag_strategy(
    *,
    strategy: str,
    keyword_returned_count: int,
    vector_documents: list[RetrievedDocument],
    fallback_used: bool,
    fallback_reason: str | None,
    target_codes: list[str],
    issue_keys: list[str],
    latency_ms: float,
) -> None:
    if not config.RAG_ENABLED:
        return
    metadata = {
        "source": "diet_recommendation_rag",
        "rag_strategy": strategy,
        "keyword_returned_count": keyword_returned_count,
        "vector_returned_count": len(vector_documents),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "disease_code": target_codes,
        "issue_keys": issue_keys,
        "selected_chunk_keys": [
            str(document.metadata.get("chunk_key") or document.metadata.get("id"))
            for document in vector_documents
            if isinstance(document.metadata, dict)
        ],
        "scores": [document.score for document in vector_documents],
        "embedding_provider": _first_metadata_value(vector_documents, "embedding_provider"),
        "embedding_model": _first_metadata_value(vector_documents, "embedding_model"),
        "latency_ms": latency_ms,
    }
    record_langfuse_event(
        name="diet.rag_strategy",
        input_payload={
            "rag_strategy": strategy,
            "target_codes": target_codes,
            "issue_keys": issue_keys,
        },
        output_payload={
            "keyword_returned_count": keyword_returned_count,
            "vector_returned_count": len(vector_documents),
            "fallback_used": fallback_used,
        },
        metadata=metadata,
    )


def _first_metadata_value(documents: list[RetrievedDocument], key: str) -> Any | None:
    for document in documents:
        if isinstance(document.metadata, dict) and document.metadata.get(key) is not None:
            return document.metadata.get(key)
    return None


def _normalize_risk_level(value: Any) -> str:
    raw = getattr(value, "value", value)
    return re.sub(r"\s+", "", str(raw or "").upper())


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
