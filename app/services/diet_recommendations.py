import re
from collections.abc import Iterable
from typing import Any

from fastapi import HTTPException
from starlette import status

from app.models.analysis import AnalysisType, RiskLevel
from app.repositories import analysis_repository, health_repository
from app.services import challenges as challenge_service
from app.services import diets as diet_service

SAFETY_NOTICE = (
    "이 내용은 진단이나 처방이 아닌 생활관리 참고 정보입니다. 실제 섭취량이 확정되지 않아 영양 판단은 참고용입니다."
)

MAX_FINDINGS = 5
MAX_RECOMMENDED_CHALLENGES = 3

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
        "message": "현재 식단 후보에서 나트륨이 높은 음식이 포함된 것으로 보여 주의가 필요합니다.",
        "reason": "나트륨 관리와 연결됩니다.",
    },
    "carbohydrate_high": {
        "type": "excess_candidate",
        "nutrient": "carbohydrate_g",
        "label": "탄수화물 주의",
        "message": "현재 식단 후보에서 탄수화물 비중이 높은 음식이 포함된 것으로 보여요.",
        "reason": "탄수화물 선택과 혈당 관리 습관에 연결됩니다.",
    },
    "sugar_high": {
        "type": "excess_candidate",
        "nutrient": "sugar_g",
        "label": "당류 주의",
        "message": "현재 식단 후보에서 당류가 높은 음식이 포함됐을 가능성이 있어요.",
        "reason": "당류를 줄이는 식습관과 연결됩니다.",
    },
    "fat_high": {
        "type": "excess_candidate",
        "nutrient": "fat_g",
        "label": "지방 주의",
        "message": "현재 식단 후보에서 지방이 높은 음식이 포함된 것으로 보여요.",
        "reason": "기름기 조절과 균형식 습관에 연결됩니다.",
    },
    "calorie_high": {
        "type": "excess_candidate",
        "nutrient": "calories_kcal",
        "label": "열량 주의",
        "message": "현재 식단 후보에서 열량이 높은 음식이 포함됐을 가능성이 있어요.",
        "reason": "야식과 과식을 줄이는 생활습관과 연결됩니다.",
    },
    "protein_support": {
        "type": "support_candidate",
        "nutrient": "protein_g",
        "label": "단백질 보완",
        "message": "단백질 반찬을 함께 보완하면 식사 균형에 도움이 될 수 있습니다.",
        "reason": "균형 잡힌 반찬 구성과 연결됩니다.",
    },
    "fiber_support": {
        "type": "support_candidate",
        "nutrient": "fiber_g",
        "label": "식이섬유 보완",
        "message": "채소, 잡곡 등 식이섬유가 있는 음식을 보완하면 좋습니다.",
        "reason": "식이섬유 보완 습관과 연결됩니다.",
    },
    "iron_support": {
        "type": "support_candidate",
        "nutrient": "iron_mg",
        "label": "철분 보완",
        "message": "철분과 비타민C가 있는 반찬을 보완하면 좋습니다.",
        "reason": "철분 보완 식습관과 연결됩니다.",
    },
    "late_night_or_irregular": {
        "type": "habit_candidate",
        "nutrient": None,
        "label": "식사 시간 주의",
        "message": "식사 시간이 늦거나 불규칙한 경우라면 규칙적인 식사 습관을 우선 추천합니다.",
        "reason": "야식과 불규칙 식사 조절에 연결됩니다.",
    },
    "alcohol_liver_support": {
        "type": "habit_candidate",
        "nutrient": None,
        "label": "간 건강 식습관",
        "message": "간 건강 관리 관점에서 음주와 당류, 기름진 음식은 주의해서 살펴보는 것이 좋습니다.",
        "reason": "절주와 간 건강 생활습관에 연결됩니다.",
    },
    "kidney_caution": {
        "type": "medical_caution",
        "nutrient": None,
        "label": "신장 관련 주의",
        "message": "신장 관련 수치가 걱정되는 경우 제한식을 단정하지 말고 의료진과 상담하며 식사일지를 남겨보세요.",
        "reason": "의료진 상담 전 식사 기록 습관과 연결됩니다.",
    },
    "balanced_support": {
        "type": "support_candidate",
        "nutrient": None,
        "label": "균형 유지",
        "message": "현재 식단은 확정 평가보다 식사 기록과 균형 유지 관점에서 참고해 주세요.",
        "reason": "균형식과 식사 기록 습관에 연결됩니다.",
    },
}

DISEASE_RULES = {
    "HTN": {
        "analysis_types": {AnalysisType.HYPERTENSION},
        "label": "혈압 관리",
        "message": "혈압 관리 관점에서 저염 식습관을 우선 추천합니다.",
        "issues": ("sodium_high",),
    },
    "DM": {
        "analysis_types": {AnalysisType.DIABETES},
        "label": "혈당 관리",
        "message": "혈당 관리 관점에서 탄수화물과 당류 선택을 천천히 확인해 보세요.",
        "issues": ("carbohydrate_high", "sugar_high"),
    },
    "DL": {
        "analysis_types": {AnalysisType.DYSLIPIDEMIA},
        "label": "콜레스테롤·중성지방 관리",
        "message": "지질 관리 관점에서 기름진 음식과 식이섬유 보완을 함께 살펴보세요.",
        "issues": ("fat_high", "fiber_support"),
    },
    "OBE": {
        "analysis_types": {AnalysisType.OBESITY, AnalysisType.ABDOMINAL_OBESITY},
        "label": "체중 관리",
        "message": "체중 관리 관점에서 열량이 높은 음식과 야식 습관을 참고해 보세요.",
        "issues": ("calorie_high", "late_night_or_irregular"),
    },
    "ANEM": {
        "analysis_types": {AnalysisType.ANEMIA},
        "label": "빈혈 관리",
        "message": "빈혈 관리 관점에서 철분과 단백질 반찬 보완을 참고해 보세요.",
        "issues": ("iron_support", "protein_support"),
    },
    "FL": {
        "analysis_types": {AnalysisType.FATTY_LIVER, AnalysisType.LIVER_FUNCTION},
        "label": "간 건강 관리",
        "message": "간 건강 관리 관점에서 음주, 당류, 기름진 음식과 야식 습관을 참고해 보세요.",
        "issues": ("alcohol_liver_support", "sugar_high", "fat_high", "calorie_high"),
    },
    "CKD": {
        "analysis_types": {AnalysisType.KIDNEY_FUNCTION, AnalysisType.CHRONIC_KIDNEY_DISEASE},
        "label": "신장 관리",
        "message": "신장 관련 식사는 개인 수치에 따라 달라질 수 있어 제한식을 단정하지 말고 의료진 상담을 권장합니다.",
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
    "fiber_support": (("채소 반찬", "잡곡밥", "해조류 반찬"), ()),
    "iron_support": (("철분이 있는 반찬", "비타민C가 있는 과일"), ("식사 직후 진한 차나 커피")),
    "late_night_or_irregular": (("규칙적인 식사", "가벼운 저녁 구성"), ("늦은 야식",)),
    "alcohol_liver_support": (("물", "담백한 단백질 반찬"), ("음주", "기름진 안주")),
    "kidney_caution": (("식사일지",), ("검증되지 않은 고단백 보충제",)),
    "balanced_support": (("채소 반찬", "단백질 반찬", "잡곡밥"), ()),
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
    return build_diet_health_recommendation_response(
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
        _finding_payload(issue_key, basis_by_issue.get(issue_key))
        for issue_key in issue_keys
        if issue_key in ISSUE_DEFINITIONS
    ]
    disease_context = [_disease_context_payload(code) for code in disease_codes if code in DISEASE_RULES]
    recommended_foods, caution_foods = _food_messages(issue_keys)
    recommended_challenges = _recommended_challenges(issue_keys, active_challenges)
    return {
        "diet_record_id": int(diet_record.id),
        "nutrition_findings": nutrition_findings,
        "disease_context": disease_context,
        "recommended_foods": recommended_foods,
        "caution_foods": caution_foods,
        "recommended_challenges": recommended_challenges,
        "safety_notice": SAFETY_NOTICE,
    }


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
    return "MFDS 기준 영양성분 후보"


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
                basis[issue_key] = str(food.get("basis") or "MFDS 기준 영양성분 후보")
                break
    return basis


def _finding_payload(issue_key: str, basis: str | None) -> dict[str, Any]:
    definition = ISSUE_DEFINITIONS[issue_key]
    return {
        "type": definition["type"],
        "issue_key": issue_key,
        "nutrient": definition["nutrient"],
        "label": definition["label"],
        "message": definition["message"],
        "basis": basis or "MFDS 기준 영양성분 후보",
    }


def _disease_context_payload(code: str) -> dict[str, str]:
    rule = DISEASE_RULES[code]
    return {
        "disease_code": code,
        "label": rule["label"],
        "message": rule["message"],
    }


def _food_messages(issue_keys: list[str]) -> tuple[list[str], list[str]]:
    recommended: list[str] = []
    caution: list[str] = []
    for issue_key in issue_keys:
        foods, cautions = FOOD_RECOMMENDATIONS.get(issue_key, ((), ()))
        recommended.extend(foods)
        caution.extend(cautions)
    return _dedupe(recommended)[:5], _dedupe(caution)[:5]


def _recommended_challenges(issue_keys: list[str], active_challenges: Iterable[Any]) -> list[dict[str, Any]]:
    active_by_title = {str(challenge.title): challenge for challenge in active_challenges}
    recommendations: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for issue_key in issue_keys:
        for title in DIET_CHALLENGE_RULES.get(issue_key, ()):
            challenge = active_by_title.get(title)
            if challenge is None:
                continue
            challenge_id = int(challenge.id)
            if challenge_id in seen_ids:
                continue
            seen_ids.add(challenge_id)
            recommendations.append(
                {
                    "challenge_id": challenge_id,
                    "title": str(challenge.title),
                    "reason": str(ISSUE_DEFINITIONS.get(issue_key, {}).get("reason") or "식단 관리 습관과 연결됩니다."),
                }
            )
            if len(recommendations) >= MAX_RECOMMENDED_CHALLENGES:
                return recommendations
    return recommendations


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
