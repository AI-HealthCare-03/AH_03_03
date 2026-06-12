from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MatchStatus(StrEnum):
    MATCHED = "matched"
    LIKELY_MATCH = "likely_match"
    MULTIPLE_CANDIDATES = "multiple_candidates"
    WEAK_MATCH = "weak_match"
    NO_CANDIDATES = "no_candidates"
    NEEDS_USER_CONFIRMATION = "needs_user_confirmation"
    FALLBACK_USED = "fallback_used"
    API_UNAVAILABLE = "api_unavailable"
    PARSE_FAILED = "parse_failed"


PROCESSED_FOOD_MARKERS = {
    "과자",
    "크래커",
    "케이크",
    "라떼",
    "새우칩",
    "칩",
    "스낵",
    "와플",
    "음료",
    "아이스크림",
    "초콜릿",
    "사탕",
    "쿠키",
    "시리얼",
    "소스",
    "맛",
    "큐브",
    "파우더",
    "분말",
    "시럽",
    "젤라또",
    "스푼",
}
FALLBACK_RULES = {
    "blt샌드위치": ["BLT 샌드위치", "샌드위치"],
    "고등어구이": ["고등어"],
    "비빔냉면": ["냉면"],
    "달걀볶음밥": ["볶음밥"],
    "육회비빔밥": ["비빔밥"],
    "핫초코": ["코코아", "핫초코"],
    "노랑파프리카": ["파프리카"],
    "녹색피망": ["피망"],
}


@dataclass(frozen=True)
class RankedCandidate:
    candidate: dict[str, Any]
    rank_score: float
    rank_reason: str
    match_status: str
    source_query: str


def build_query_variants(query: str) -> list[str]:
    cleaned_query = query.strip()
    variants = [cleaned_query]
    variants.extend(FALLBACK_RULES.get(normalize_food_text(cleaned_query), []))
    if cleaned_query.lower().startswith("blt") and " " not in cleaned_query:
        variants.append(cleaned_query[:3].upper() + " " + cleaned_query[3:])
    return dedupe_preserve_order([variant for variant in variants if variant.strip()])


def rerank_candidates(
    *,
    query: str,
    candidates: list[dict[str, Any]],
) -> tuple[list[RankedCandidate], RankedCandidate | None]:
    ranked = [
        rank_candidate(query=str(candidate.get("source_query") or query), candidate=candidate)
        for candidate in candidates
        if str(candidate.get("food_name") or "").strip()
    ]
    ranked.sort(key=lambda item: item.rank_score, reverse=True)
    if not ranked:
        return [], None

    top = ranked[0]
    top = RankedCandidate(
        candidate=top.candidate,
        rank_score=top.rank_score,
        rank_reason=top.rank_reason,
        match_status=resolve_match_status(top=top, ranked=ranked).value,
        source_query=top.source_query,
    )
    ranked[0] = top
    return ranked, top


def rank_candidate(*, query: str, candidate: dict[str, Any]) -> RankedCandidate:
    food_name = str(candidate.get("food_name") or "")
    normalized_query = normalize_food_text(query)
    normalized_name = normalize_food_text(food_name)
    query_tokens = set(food_tokens(query))
    name_tokens = set(food_tokens(food_name))
    score = 0.0
    reasons: list[str] = []

    if normalized_query and normalized_query == normalized_name:
        score += 100
        reasons.append("exact_normalized_match")
    if normalized_query and normalized_name.startswith(normalized_query):
        score += 45
        reasons.append("name_startswith_query")
    if normalized_query and normalized_query in normalized_name:
        score += 30
        reasons.append("query_contained_in_name")
    if normalized_name and normalized_name in normalized_query:
        score += 20
        reasons.append("name_contained_in_query")

    overlap = query_tokens & name_tokens
    if query_tokens:
        overlap_score = 30 * (len(overlap) / len(query_tokens))
        if overlap_score:
            score += overlap_score
            reasons.append(f"token_overlap:{','.join(sorted(overlap))}")

    penalty = processed_food_penalty(food_name=food_name, query=query)
    if penalty:
        score -= penalty
        reasons.append(f"processed_food_penalty:-{penalty:g}")

    length_penalty = long_name_penalty(food_name=food_name, query=query)
    if length_penalty:
        score -= length_penalty
        reasons.append(f"long_name_penalty:-{length_penalty:g}")

    brand_penalty = brand_like_penalty(food_name=food_name)
    if brand_penalty:
        score -= brand_penalty
        reasons.append(f"brand_like_penalty:-{brand_penalty:g}")

    score = max(score, 0.0)
    return RankedCandidate(
        candidate=candidate,
        rank_score=round(score, 4),
        rank_reason="; ".join(reasons) if reasons else "no_textual_match",
        match_status="unresolved",
        source_query=query,
    )


def resolve_match_status(*, top: RankedCandidate, ranked: list[RankedCandidate]) -> MatchStatus:
    score = top.rank_score
    if not ranked:
        return MatchStatus.NO_CANDIDATES
    if "exact_normalized_match" in top.rank_reason:
        if "processed_food_penalty" in top.rank_reason and score < 160:
            return MatchStatus.LIKELY_MATCH
        return MatchStatus.MATCHED
    if "processed_food_penalty" in top.rank_reason:
        return MatchStatus.WEAK_MATCH if score >= 25 else MatchStatus.NEEDS_USER_CONFIRMATION
    close_candidates = [candidate for candidate in ranked if candidate.rank_score >= max(score - 5, 0)]
    if score >= 45 and len(close_candidates) > 1:
        return MatchStatus.MULTIPLE_CANDIDATES
    if score >= 55:
        return MatchStatus.LIKELY_MATCH
    if score >= 25:
        return MatchStatus.WEAK_MATCH
    return MatchStatus.NEEDS_USER_CONFIRMATION


def needs_fallback(match_status: str | None) -> bool:
    return match_status in {
        MatchStatus.WEAK_MATCH.value,
        MatchStatus.NO_CANDIDATES.value,
        MatchStatus.NEEDS_USER_CONFIRMATION.value,
    }


def needs_user_confirmation(match_status: str | None) -> bool:
    return match_status in {
        MatchStatus.LIKELY_MATCH.value,
        MatchStatus.MULTIPLE_CANDIDATES.value,
        MatchStatus.WEAK_MATCH.value,
        MatchStatus.NO_CANDIDATES.value,
        MatchStatus.NEEDS_USER_CONFIRMATION.value,
        MatchStatus.FALLBACK_USED.value,
        MatchStatus.API_UNAVAILABLE.value,
        MatchStatus.PARSE_FAILED.value,
    }


def normalize_food_text(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"\([^)]*\)", "", normalized)
    normalized = re.sub(r"[\s_\-·,/]+", "", normalized)
    return normalized


def food_tokens(value: str) -> list[str]:
    spaced = re.sub(r"([A-Za-z]+)", r" \1 ", value)
    return [token.lower() for token in re.split(r"[\s_\-·,/()]+", spaced) if token and not token.isspace()]


def processed_food_penalty(*, food_name: str, query: str) -> float:
    normalized_name = normalize_food_text(food_name)
    normalized_query = normalize_food_text(query)
    if normalized_name == normalized_query:
        return 0.0
    if normalized_query in {"샌드위치", "냉면", "미역국"}:
        return 0.0
    markers = [marker for marker in PROCESSED_FOOD_MARKERS if marker in normalized_name]
    if not markers:
        return 0.0
    return min(30.0, 10.0 + 5.0 * len(markers))


def brand_like_penalty(*, food_name: str) -> float:
    penalty = 0.0
    if "[" in food_name or "]" in food_name:
        penalty += 10.0
    if "&" in food_name:
        penalty += 5.0
    if "_" in food_name:
        penalty += 3.0
    if sum(char.isdigit() for char in food_name) >= 2:
        penalty += 5.0
    return min(penalty, 20.0)


def long_name_penalty(*, food_name: str, query: str) -> float:
    normalized_name = normalize_food_text(food_name)
    normalized_query = normalize_food_text(query)
    extra_length = len(normalized_name) - len(normalized_query)
    if extra_length <= 8:
        return 0.0
    return min(20.0, float(extra_length - 8))


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_food_text(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result
