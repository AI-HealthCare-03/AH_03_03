from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from reranking import MatchStatus, build_query_variants, needs_fallback, needs_user_confirmation, rerank_candidates
from schemas import NutritionCandidate, NutritionLookupResult

from ai_runtime.cv.food.normalization import normalize_food_name

MFDS_ENDPOINT = "https://apis.data.go.kr/1471000/FoodNtrCpntDbInfo02/getFoodNtrCpntDbInq02"
REQUEST_TIMEOUT_SECONDS = 15.0
REQUEST_RETRY_COUNT = 2


class MfdsNutritionProvider:
    provider_name = "mfds"

    def __init__(self, *, max_candidates: int = 10, enable_fallback: bool = True) -> None:
        load_env_file()
        self.max_candidates = max_candidates
        self.enable_fallback = enable_fallback

    def lookup(self, query: str) -> NutritionLookupResult:
        started = time.perf_counter()
        normalized_query = normalize_food_name(query)
        if not normalized_query:
            return NutritionLookupResult(
                query=query,
                normalized_query=normalized_query,
                provider=self.provider_name,
                status="no_query",
                original_query=query,
                used_query=query,
                match_status="no_query",
                latency_seconds=round(time.perf_counter() - started, 4),
                error_message="empty query",
                needs_user_confirmation=True,
            )

        query_variants = build_query_variants(query) if self.enable_fallback else [query]
        first_candidates: list[dict[str, Any]] = []
        all_candidates: list[dict[str, Any]] = []
        errors: list[str] = []

        for index, query_variant in enumerate(query_variants):
            if index > 0 and not should_try_fallback(query=query, candidates=first_candidates):
                break
            try:
                items = self.fetch_items(query_variant)
            except MfdsParseError as exc:
                return self.failure_result(
                    query=query,
                    normalized_query=normalized_query,
                    started=started,
                    status=MatchStatus.PARSE_FAILED.value,
                    error_message=str(exc),
                )
            except Exception as exc:
                errors.append(f"{query_variant}: {type(exc).__name__}")
                continue

            candidates = [build_candidate(item, source_query=query_variant) for item in items]
            if index == 0:
                first_candidates = candidates
            all_candidates.extend(candidates)
            if index == 0 and not should_try_fallback(query=query, candidates=candidates):
                break

        if not all_candidates and errors:
            return self.failure_result(
                query=query,
                normalized_query=normalized_query,
                started=started,
                status=MatchStatus.API_UNAVAILABLE.value,
                error_message="; ".join(errors),
            )

        ranked_candidates, top = rerank_candidates(query=query, candidates=all_candidates)
        if top is None:
            return NutritionLookupResult(
                query=query,
                normalized_query=normalized_query,
                provider=self.provider_name,
                status=MatchStatus.NO_CANDIDATES.value,
                original_query=query,
                used_query=query,
                fallback_queries=query_variants[1:],
                fallback_used=False,
                match_status=MatchStatus.NO_CANDIDATES.value,
                candidate_count=0,
                source="mfds",
                latency_seconds=round(time.perf_counter() - started, 4),
                needs_user_confirmation=True,
            )

        selected = top.candidate
        fallback_used = top.source_query != query
        match_status = top.match_status
        status = MatchStatus.FALLBACK_USED.value if fallback_used else match_status
        return NutritionLookupResult(
            query=query,
            normalized_query=normalized_query,
            provider=self.provider_name,
            status=status,
            original_query=query,
            used_query=top.source_query,
            fallback_queries=query_variants[1:],
            fallback_used=fallback_used,
            match_status=match_status,
            rank_score=top.rank_score,
            rank_reason=top.rank_reason,
            matched_food_name=optional_string(selected.get("food_name")),
            matched_food_code=optional_string(selected.get("food_code")),
            candidate_count=len(all_candidates),
            top_candidates=[
                NutritionCandidate(
                    food_name=str(candidate.candidate.get("food_name") or ""),
                    food_code=optional_string(candidate.candidate.get("food_code")),
                    confidence=candidate.rank_score,
                )
                for candidate in ranked_candidates[:5]
            ],
            energy_kcal=optional_float(selected.get("energy_kcal")),
            carbohydrate_g=optional_float(selected.get("carbohydrate_g")),
            protein_g=optional_float(selected.get("protein_g")),
            fat_g=optional_float(selected.get("fat_g")),
            sodium_mg=optional_float(selected.get("sodium_mg")),
            serving_size=optional_string(selected.get("serving_size")),
            source="mfds",
            latency_seconds=round(time.perf_counter() - started, 4),
            needs_user_confirmation=fallback_used or needs_user_confirmation(match_status),
        )

    def fetch_items(self, query: str) -> list[dict[str, Any]]:
        payload = self.fetch_payload(query)
        items = extract_items(payload)
        return items[: self.max_candidates]

    def fetch_payload(self, query: str) -> dict[str, Any]:
        base_params = {
            "pageNo": "1",
            "numOfRows": str(self.max_candidates),
            "type": "json",
            "FOOD_NM_KR": query,
        }
        attempts = build_key_attempts()
        if not attempts:
            raise RuntimeError("MFDS_SERVICE_KEY is not set")

        errors: list[str] = []
        for key_mode, service_key, service_key_is_encoded in attempts:
            try:
                payload = request_json(
                    MFDS_ENDPOINT,
                    base_params,
                    service_key_param="serviceKey",
                    service_key=service_key,
                    service_key_is_encoded=service_key_is_encoded,
                )
                if looks_like_auth_failure(payload):
                    errors.append(f"{key_mode}: authentication failure")
                    continue
                return payload
            except Exception as exc:
                errors.append(f"{key_mode}: {type(exc).__name__}")
        raise RuntimeError("MFDS request failed with all configured key modes: " + "; ".join(errors))

    def failure_result(
        self,
        *,
        query: str,
        normalized_query: str,
        started: float,
        status: str,
        error_message: str,
    ) -> NutritionLookupResult:
        return NutritionLookupResult(
            query=query,
            normalized_query=normalized_query,
            provider=self.provider_name,
            status=status,
            original_query=query,
            used_query=query,
            match_status=status,
            source="mfds",
            latency_seconds=round(time.perf_counter() - started, 4),
            error_message=error_message,
            needs_user_confirmation=True,
        )


class MfdsParseError(ValueError):
    pass


def load_env_file(path: Path | None = None) -> None:
    env_path = path or Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def build_key_attempts() -> list[tuple[str, str, bool]]:
    decoded_key = os.getenv("MFDS_SERVICE_KEY", "").strip()
    encoded_key = os.getenv("MFDS_SERVICE_KEY_ENCODED", "").strip()
    attempts: list[tuple[str, str, bool]] = []
    if decoded_key:
        attempts.append(("decoded", decoded_key, False))
    if encoded_key and encoded_key != decoded_key:
        attempts.append(("encoded", encoded_key, True))
    return attempts


def request_json(
    url: str,
    params: dict[str, str],
    *,
    service_key_param: str,
    service_key: str,
    service_key_is_encoded: bool,
) -> dict[str, Any]:
    query_string = build_query_string(
        params,
        service_key_param=service_key_param,
        service_key=service_key,
        service_key_is_encoded=service_key_is_encoded,
    )
    request = urllib.request.Request(
        f"{url}?{query_string}",
        headers={"Accept": "application/json", "User-Agent": "AH_03_03-nutrition-lookup-eval/0.1"},
    )
    last_error: Exception | None = None
    for attempt in range(REQUEST_RETRY_COUNT + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise MfdsParseError("API response JSON root is not an object")
            return parsed
        except json.JSONDecodeError as exc:
            raise MfdsParseError(f"Failed to parse MFDS JSON response: {exc}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt >= REQUEST_RETRY_COUNT:
                break
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"API request failed: {type(last_error).__name__}")


def build_query_string(
    params: dict[str, str],
    *,
    service_key_param: str,
    service_key: str,
    service_key_is_encoded: bool,
) -> str:
    query_string = urllib.parse.urlencode(params)
    encoded_key = service_key if service_key_is_encoded else urllib.parse.quote(service_key, safe="")
    return f"{query_string}&{urllib.parse.quote(service_key_param)}={encoded_key}"


def extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        deep_get(payload, ["body", "items"]),
        deep_get(payload, ["body", "items", "item"]),
        deep_get(payload, ["response", "body", "items"]),
        deep_get(payload, ["response", "body", "items", "item"]),
        payload.get("items"),
        payload.get("data"),
    ]
    for candidate in candidates:
        normalized = normalize_items(candidate)
        if normalized:
            return normalized
    return []


def normalize_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def build_candidate(item: dict[str, Any], *, source_query: str) -> dict[str, Any]:
    return {
        "source_query": source_query,
        "food_name": first_value(item, ["FOOD_NM_KR", "FOOD_NM"]),
        "food_code": first_value(item, ["FOOD_CD"]),
        "serving_size": first_value(item, ["SERVING_SIZE"]),
        "energy_kcal": optional_float(first_value(item, ["AMT_NUM1"])),
        "protein_g": optional_float(first_value(item, ["AMT_NUM3"])),
        "fat_g": optional_float(first_value(item, ["AMT_NUM4"])),
        "carbohydrate_g": optional_float(first_value(item, ["AMT_NUM6"])),
        "sodium_mg": optional_float(first_value(item, ["AMT_NUM13"])),
    }


def should_try_fallback(*, query: str, candidates: list[dict[str, Any]]) -> bool:
    if not candidates:
        return True
    _, top = rerank_candidates(query=query, candidates=candidates)
    return needs_fallback(top.match_status if top else MatchStatus.NO_CANDIDATES.value)


def looks_like_auth_failure(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, ensure_ascii=False).lower()
    auth_markers = ["service key", "servicekey", "인증", "등록되지 않은", "invalid", "unauthorized", "auth"]
    result_code = str(
        deep_get(payload, ["response", "header", "resultCode"]) or deep_get(payload, ["header", "resultCode"]) or ""
    ).strip()
    if result_code and result_code not in {"00", "0", "INFO-000"}:
        return any(marker in text for marker in auth_markers)
    return any(marker in text for marker in auth_markers)


def deep_get(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_value(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in {None, ""}:
            return value
    return None


def optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def optional_float(value: object) -> float | None:
    try:
        raw_value = str(value).replace(",", "").strip()
        return float(raw_value) if raw_value else None
    except (TypeError, ValueError):
        return None
