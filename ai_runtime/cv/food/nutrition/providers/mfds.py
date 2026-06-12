from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ai_runtime.cv.food.matcher import FoodDbMatcher, FoodMatchResult
from ai_runtime.cv.food.normalization import cleanup_food_query, normalize_food_name

MFDS_ENDPOINT = "https://apis.data.go.kr/1471000/FoodNtrCpntDbInfo02/getFoodNtrCpntDbInq02"
REQUEST_USER_AGENT = "AH_03_03-diet-runtime/0.1"
REQUEST_RETRY_COUNT = 2
REQUEST_RETRY_BACKOFF_SECONDS = (0.3, 0.6)
TOP_CANDIDATE_METADATA_LIMIT = 5


class MfdsMatchStatus(StrEnum):
    MATCHED = "matched"
    LIKELY_MATCH = "likely_match"
    MULTIPLE_CANDIDATES = "multiple_candidates"
    WEAK_MATCH = "weak_match"
    NO_CANDIDATES = "no_candidates"
    API_UNAVAILABLE = "api_unavailable"
    PARSE_FAILED = "parse_failed"


@dataclass(frozen=True)
class MfdsCandidate:
    food_name: str
    food_code: str | None
    source_query: str
    rank_score: float
    rank_reason: str
    nutrition: dict[str, Any] | None = None


FetchPayload = Callable[[str], dict[str, Any]]


class MfdsFoodDbMatcher:
    """FoodDbMatcher adapter backed by the MFDS public nutrition API.

    This adapter only enriches detected food names with a candidate name/code.
    It intentionally keeps needs_user_confirmation=True so this stage does not
    auto-confirm nutrition candidates before the confirmation API exists.
    """

    applies_lookup_gating = True

    def __init__(
        self,
        *,
        service_key: str,
        encoded_service_key: str | None = None,
        timeout_seconds: float = 5.0,
        max_candidates: int = 5,
        fallback_matcher: FoodDbMatcher | None = None,
        fetch_payload: FetchPayload | None = None,
        retry_count: int = REQUEST_RETRY_COUNT,
        retry_backoff_seconds: tuple[float, ...] = REQUEST_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.service_key = service_key.strip()
        self.encoded_service_key = (encoded_service_key or "").strip() or None
        self.timeout_seconds = timeout_seconds
        self.max_candidates = max(1, max_candidates)
        self.fallback_matcher = fallback_matcher
        self._fetch_payload = fetch_payload
        self.retry_count = max(0, retry_count)
        self.retry_backoff_seconds = retry_backoff_seconds

    def match(self, query: str) -> FoodMatchResult:
        query_name = cleanup_food_query(query)
        if not query_name:
            return FoodMatchResult(original_name=query.strip(), query_name=query_name)

        started = time.perf_counter()
        try:
            candidates = self._lookup_candidates(query_name)
        except MfdsParseError:
            return self._api_failure_result(query=query, query_name=query_name, status=MfdsMatchStatus.PARSE_FAILED)
        except Exception:
            return self._api_failure_result(query=query, query_name=query_name, status=MfdsMatchStatus.API_UNAVAILABLE)

        if not candidates:
            return FoodMatchResult(
                original_name=query.strip(),
                query_name=query_name,
                match_source="mfds_no_candidates",
                needs_user_confirmation=True,
            )

        top_candidates = sorted(candidates, key=lambda candidate: candidate.rank_score, reverse=True)
        top = top_candidates[0]
        status = _resolve_match_status(top, top_candidates)
        latency_ms = round((time.perf_counter() - started) * 1000)
        return FoodMatchResult(
            original_name=query.strip(),
            query_name=query_name,
            matched_food_name=top.food_name,
            matched_food_code=top.food_code,
            match_source=f"mfds_{status.value}",
            match_confidence=round(min(top.rank_score / 100, 1.0), 4),
            needs_user_confirmation=True,
            metadata={
                "provider": "mfds",
                "status": status.value,
                "candidate_count": len(candidates),
                "used_query": top.source_query,
                "rank_score": top.rank_score,
                "rank_reason": top.rank_reason,
                "nutrition": top.nutrition,
                "top_candidates": _serialize_top_candidates(top_candidates[:TOP_CANDIDATE_METADATA_LIMIT]),
                "latency_ms": latency_ms,
            },
        )

    def _api_failure_result(self, *, query: str, query_name: str, status: MfdsMatchStatus) -> FoodMatchResult:
        if self.fallback_matcher is not None:
            fallback = self.fallback_matcher.match(query)
            if fallback.matched_food_name:
                return fallback
        return FoodMatchResult(
            original_name=query.strip(),
            query_name=query_name,
            match_source=f"mfds_{status.value}",
            needs_user_confirmation=True,
        )

    def _lookup_candidates(self, query: str) -> list[MfdsCandidate]:
        candidates: list[MfdsCandidate] = []
        for query_variant in _query_variants(query):
            payload = self._fetch_query_payload(query_variant)
            variant_candidates: list[MfdsCandidate] = []
            for item in _extract_items(payload)[: self.max_candidates]:
                candidate = _candidate_from_item(item=item, source_query=query_variant, original_query=query)
                if candidate is not None:
                    variant_candidates.append(candidate)
            candidates.extend(variant_candidates)
            if variant_candidates and not _should_try_next_query_variant(variant_candidates):
                break
        return candidates

    def _fetch_query_payload(self, query: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt_index in range(self.retry_count + 1):
            try:
                return self._fetch_payload(query) if self._fetch_payload else self._request_payload(query)
            except MfdsParseError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt_index >= self.retry_count:
                    break
                backoff = (
                    self.retry_backoff_seconds[min(attempt_index, len(self.retry_backoff_seconds) - 1)]
                    if self.retry_backoff_seconds
                    else 0
                )
                if backoff > 0:
                    time.sleep(backoff)
        if last_error is not None:
            raise last_error
        msg = "MFDS request failed"
        raise RuntimeError(msg)

    def _request_payload(self, query: str) -> dict[str, Any]:
        attempts = [(self.service_key, False)]
        if self.encoded_service_key and self.encoded_service_key != self.service_key:
            attempts.append((self.encoded_service_key, True))

        errors: list[Exception] = []
        for service_key, service_key_is_encoded in attempts:
            try:
                payload = _request_json(
                    query=query,
                    service_key=service_key,
                    service_key_is_encoded=service_key_is_encoded,
                    timeout_seconds=self.timeout_seconds,
                    max_candidates=self.max_candidates,
                )
                if _looks_like_auth_failure(payload):
                    continue
                return payload
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[-1]
        msg = "MFDS service key authentication failed"
        raise RuntimeError(msg)


class MfdsParseError(ValueError):
    pass


def _request_json(
    *,
    query: str,
    service_key: str,
    service_key_is_encoded: bool,
    timeout_seconds: float,
    max_candidates: int,
) -> dict[str, Any]:
    params = {
        "pageNo": "1",
        "numOfRows": str(max_candidates),
        "type": "json",
        "FOOD_NM_KR": query,
    }
    query_string = urllib.parse.urlencode(params)
    encoded_key = service_key if service_key_is_encoded else urllib.parse.quote(service_key, safe="")
    url = f"{MFDS_ENDPOINT}?{query_string}&serviceKey={encoded_key}"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": REQUEST_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise MfdsParseError(f"Failed to parse MFDS JSON response: {exc}") from exc
    except (TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(f"MFDS request failed: {type(exc).__name__}") from exc
    if not isinstance(parsed, dict):
        msg = "MFDS JSON root is not an object"
        raise MfdsParseError(msg)
    return parsed


def _query_variants(query: str) -> list[str]:
    variants = [query]
    normalized = normalize_food_name(query)
    fallback_rules = {
        "blt샌드위치": ["BLT 샌드위치", "샌드위치"],
        "고등어구이": ["고등어"],
        "비빔냉면": ["냉면"],
        "달걀볶음밥": ["볶음밥"],
        "육회비빔밥": ["비빔밥"],
        "핫초코": ["코코아", "핫초코"],
        "노랑파프리카": ["파프리카"],
        "녹색피망": ["피망"],
        "국물면요리": ["면요리", "국수"],
        "중식면요리": ["중식면", "면요리"],
    }
    variants.extend(fallback_rules.get(normalized, []))
    if query.strip().lower().startswith("blt") and " " not in query.strip():
        variants.append(query.strip()[:3].upper() + " " + query.strip()[3:])
    return _dedupe_preserve_order([variant for variant in variants if variant.strip()])


def _candidate_from_item(*, item: dict[str, Any], source_query: str, original_query: str) -> MfdsCandidate | None:
    food_name = _first_value(item, ["FOOD_NM_KR", "FOOD_NM"])
    if not food_name:
        return None
    food_code = _first_value(item, ["FOOD_CD"])
    score, reason = _rank_candidate(query=source_query or original_query, food_name=food_name)
    return MfdsCandidate(
        food_name=food_name,
        food_code=food_code,
        source_query=source_query,
        rank_score=score,
        rank_reason=reason,
        nutrition=_nutrition_from_item(item),
    )


def _rank_candidate(*, query: str, food_name: str) -> tuple[float, str]:
    normalized_query = normalize_food_name(query)
    normalized_name = normalize_food_name(food_name)
    query_tokens = set(_food_tokens(query))
    name_tokens = set(_food_tokens(food_name))
    score = 0.0
    reasons: list[str] = []

    if normalized_query and normalized_query == normalized_name:
        score += 100.0
        reasons.append("exact_normalized_match")
    if normalized_query and normalized_name.startswith(normalized_query):
        score += 45.0
        reasons.append("name_startswith_query")
    if normalized_query and normalized_query in normalized_name:
        score += 30.0
        reasons.append("query_contained_in_name")
    if normalized_name and normalized_name in normalized_query:
        score += 20.0
        reasons.append("name_contained_in_query")

    overlap = query_tokens & name_tokens
    if query_tokens and overlap:
        score += 30.0 * (len(overlap) / len(query_tokens))
        reasons.append(f"token_overlap:{','.join(sorted(overlap))}")

    processed_penalty = _processed_food_penalty(food_name=food_name, query=query)
    if processed_penalty:
        score -= processed_penalty
        reasons.append(f"processed_food_penalty:-{processed_penalty:g}")

    score = round(max(score, 0.0), 4)
    return score, "; ".join(reasons) if reasons else "no_textual_match"


def _resolve_match_status(top: MfdsCandidate, ranked: list[MfdsCandidate]) -> MfdsMatchStatus:
    if "exact_normalized_match" in top.rank_reason:
        return MfdsMatchStatus.MATCHED
    close_candidates = [candidate for candidate in ranked if candidate.rank_score >= max(top.rank_score - 5, 0)]
    if top.rank_score >= 45 and len(close_candidates) > 1:
        return MfdsMatchStatus.MULTIPLE_CANDIDATES
    if top.rank_score >= 55:
        return MfdsMatchStatus.LIKELY_MATCH
    if top.rank_score >= 25:
        return MfdsMatchStatus.WEAK_MATCH
    return MfdsMatchStatus.WEAK_MATCH


def _should_try_next_query_variant(candidates: list[MfdsCandidate]) -> bool:
    ranked = sorted(candidates, key=lambda candidate: candidate.rank_score, reverse=True)
    if not ranked:
        return True
    status = _resolve_match_status(ranked[0], ranked)
    return status == MfdsMatchStatus.WEAK_MATCH


def _serialize_top_candidates(candidates: list[MfdsCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "food_name": candidate.food_name,
            "food_code": candidate.food_code,
            "rank_score": candidate.rank_score,
            "rank_reason": candidate.rank_reason,
            "nutrition": candidate.nutrition,
        }
        for candidate in candidates
    ]


def _nutrition_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    nutrition = {
        "calories_kcal": _optional_float(_first_value(item, ["AMT_NUM1", "ENERGY_KCAL", "ENERC", "NUTR_CONT1"])),
        "carbohydrate_g": _optional_float(_first_value(item, ["AMT_NUM6", "CARBOHYDRATE_G", "CHOCDF", "NUTR_CONT2"])),
        "protein_g": _optional_float(_first_value(item, ["AMT_NUM3", "PROTEIN_G", "PROCNT", "NUTR_CONT3"])),
        "fat_g": _optional_float(_first_value(item, ["AMT_NUM4", "FAT_G", "FATCE", "NUTR_CONT4"])),
        "sodium_mg": _optional_float(_first_value(item, ["AMT_NUM13", "SODIUM_MG", "NA", "NUTR_CONT6"])),
    }
    if not any(value is not None for value in nutrition.values()):
        return None

    serving_size = _first_value(
        item,
        [
            "SERVING_SIZE",
            "SERVING_SIZE_G",
            "SERVING_UNIT",
            "NUTR_CONT_BASE",
            "MAKER_SERVING_SIZE",
            "1회제공량",
            "총내용량",
        ],
    )
    return {
        **nutrition,
        **_basis_from_serving_size(serving_size),
    }


def _basis_from_serving_size(serving_size: str | None) -> dict[str, Any]:
    if not serving_size:
        return {
            "basis_amount": None,
            "basis_unit": None,
            "basis_label": "기준량 확인 필요",
        }

    normalized = serving_size.strip()
    lower = normalized.lower()
    number_match = re.search(r"(\d+(?:\.\d+)?)\s*(g|그램|ml|mL|㎖)", normalized)
    if number_match:
        amount = _optional_float(number_match.group(1))
        raw_unit = number_match.group(2).lower()
        unit = "ml" if raw_unit in {"ml", "㎖"} else "g"
        label = f"{amount:g}{unit} 기준" if amount is not None else "기준량 확인 필요"
        return {
            "basis_amount": amount,
            "basis_unit": unit,
            "basis_label": label,
            "serving_size": normalized,
        }

    if "1회" in normalized or "serving" in lower:
        return {
            "basis_amount": 1,
            "basis_unit": "serving",
            "basis_label": "1회 제공량 기준",
            "serving_size": normalized,
        }

    return {
        "basis_amount": None,
        "basis_unit": None,
        "basis_label": "기준량 확인 필요",
        "serving_size": normalized,
    }


def _processed_food_penalty(*, food_name: str, query: str) -> float:
    normalized_name = normalize_food_name(food_name)
    normalized_query = normalize_food_name(query)
    if normalized_name == normalized_query:
        return 0.0
    markers = ("과자", "크래커", "케이크", "라떼", "아이스크림", "초콜릿", "사탕", "쿠키", "소스", "맛")
    marker_count = sum(1 for marker in markers if marker in normalized_name)
    return min(30.0, 10.0 + 5.0 * marker_count) if marker_count else 0.0


def _extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        _deep_get(payload, ["body", "items", "item"]),
        _deep_get(payload, ["body", "items"]),
        _deep_get(payload, ["response", "body", "items", "item"]),
        _deep_get(payload, ["response", "body", "items"]),
        payload.get("items"),
        payload.get("data"),
    ]
    for candidate in candidates:
        normalized = _normalize_items(candidate)
        if normalized:
            return normalized
    return []


def _normalize_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _looks_like_auth_failure(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, ensure_ascii=False).lower()
    result_code = str(
        _deep_get(payload, ["response", "header", "resultCode"]) or _deep_get(payload, ["header", "resultCode"]) or ""
    ).strip()
    if result_code and result_code not in {"00", "0", "INFO-000"}:
        return any(marker in text for marker in ("service key", "servicekey", "인증", "invalid", "unauthorized"))
    return False


def _deep_get(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_value(item: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if value not in {None, ""}:
            text = str(value).strip()
            if text:
                return text
    return None


def _optional_float(value: Any) -> float | None:
    try:
        text = str(value or "").replace(",", "").strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _food_tokens(value: str) -> list[str]:
    return [token.lower() for token in value.replace("_", " ").replace("-", " ").split() if token.strip()]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = cleanup_food_query(value).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result
