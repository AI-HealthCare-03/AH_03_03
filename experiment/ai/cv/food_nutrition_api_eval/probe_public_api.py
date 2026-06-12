from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from reranking import MatchStatus, build_query_variants, needs_fallback, needs_user_confirmation, rerank_candidates

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES = [
    "가래떡",
    "가리비",
    "샌드위치",
    "blt샌드위치",
    "BLT 샌드위치",
    "파프리카",
    "노랑파프리카",
    "녹색피망",
    "자몽",
    "홍차",
    "핫초코",
    "냉면",
    "비빔냉면",
    "미역국",
    "고등어",
    "고등어구이",
    "달걀볶음밥",
    "육회비빔밥",
    "카페라떼",
    "블루베리",
]
OUTPUT_COLUMNS = [
    "provider",
    "operation_url",
    "key_mode",
    "query",
    "original_query",
    "used_query",
    "query_variants",
    "fallback_queries",
    "fallback_used",
    "status",
    "match_status",
    "needs_user_confirmation",
    "candidate_count",
    "top_candidates",
    "original_top_candidate",
    "reranked_top_candidate",
    "original_top_food_name",
    "reranked_food_name",
    "reranked_food_code",
    "reranked_source_query",
    "rank_score",
    "rank_reason",
    "food_name",
    "food_code",
    "serving_size",
    "energy_kcal",
    "carbohydrate_g",
    "protein_g",
    "fat_g",
    "sodium_mg",
    "nutrition_field_completeness",
    "raw_response_sample",
    "latency_seconds",
    "error_message",
]
MFDS_ENDPOINT = "https://apis.data.go.kr/1471000/FoodNtrCpntDbInfo02/getFoodNtrCpntDbInq02"
RDA_ENDPOINT = "https://koreanfood.rda.go.kr/kfi/openapi/service"
REQUEST_TIMEOUT_SECONDS = 15.0
REQUEST_RETRY_COUNT = 2
MFDS_FIELD_MAPPING = {
    "food_name": "FOOD_NM_KR",
    "food_code": "FOOD_CD",
    "serving_size": "SERVING_SIZE",
    "energy_kcal": "AMT_NUM1",
    "protein_g": "AMT_NUM3",
    "fat_g": "AMT_NUM4",
    "carbohydrate_g": "AMT_NUM6",
    "sodium_mg": "AMT_NUM13",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe public food nutrition API response shapes.")
    parser.add_argument("--provider", choices=["mfds", "rda", "all"], required=True)
    parser.add_argument("--queries", nargs="*", default=None, help="Optional query list.")
    parser.add_argument("--limit", type=int, default=None, help="Limit query count.")
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR / "outputs"))
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=10)
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument("--enable-fallback", dest="enable_fallback", action="store_true")
    fallback_group.add_argument("--disable-fallback", dest="enable_fallback", action="store_false")
    parser.set_defaults(enable_fallback=True)
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()
    providers = ["mfds", "rda"] if args.provider == "all" else [args.provider]
    queries = (args.queries or DEFAULT_QUERIES)[: args.limit]
    if not queries:
        print("No query to probe.")
        return 0

    missing_keys = [provider for provider in providers if not api_key_for_provider(provider)]
    if missing_keys:
        print("Public nutrition API probe skipped because required API key is missing.")
        for provider in missing_keys:
            env_name = "MFDS_SERVICE_KEY" if provider == "mfds" else "RDA_SERVICE_KEY"
            print(f"- {provider}: set {env_name} in environment or .env")
        print("No API key value was printed.")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for provider in providers:
        for query in queries:
            rows.append(
                probe_query(
                    provider=provider,
                    query=query,
                    page_size=args.max_candidates or args.page_size,
                    enable_fallback=args.enable_fallback,
                )
            )

    write_csv(output_dir / "api_probe_results.csv", rows)
    write_json(output_dir / "api_probe_results.json", rows)
    write_report(output_dir / "api_probe_report.md", rows)
    print(f"Wrote {output_dir / 'api_probe_results.csv'}")
    print(f"Wrote {output_dir / 'api_probe_results.json'}")
    print(f"Wrote {output_dir / 'api_probe_report.md'}")
    return 0


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


def api_key_for_provider(provider: str) -> str | None:
    env_name = "MFDS_SERVICE_KEY" if provider == "mfds" else "RDA_SERVICE_KEY"
    value = os.getenv(env_name)
    return value.strip() if value and value.strip() else None


def encoded_api_key_for_provider(provider: str) -> str | None:
    if provider != "mfds":
        return None
    value = os.getenv("MFDS_SERVICE_KEY_ENCODED")
    return value.strip() if value and value.strip() else None


def probe_query(*, provider: str, query: str, page_size: int, enable_fallback: bool) -> dict[str, Any]:
    started = time.perf_counter()
    if provider == "mfds":
        return probe_mfds_query(
            query=query,
            page_size=page_size,
            started=started,
            enable_fallback=enable_fallback,
        )

    try:
        payload, key_mode, operation_url = fetch_provider_payload(
            provider=provider,
            query=query,
            page_size=page_size,
        )
        items = extract_items(payload)
        result = normalize_probe_result(provider=provider, query=query, items=items, payload=payload)
        result["operation_url"] = operation_url
        result["key_mode"] = key_mode
        result["latency_seconds"] = round(time.perf_counter() - started, 4)
        return result
    except Exception as exc:
        return {
            "provider": provider,
            "operation_url": endpoint_for_provider(provider),
            "key_mode": None,
            "query": query,
            "original_query": query,
            "used_query": query,
            "query_variants": [],
            "fallback_queries": [],
            "fallback_used": False,
            "status": MatchStatus.API_UNAVAILABLE.value,
            "match_status": MatchStatus.API_UNAVAILABLE.value,
            "needs_user_confirmation": True,
            "candidate_count": 0,
            "top_candidates": [],
            "original_top_candidate": None,
            "reranked_top_candidate": None,
            "original_top_food_name": None,
            "reranked_food_name": None,
            "reranked_food_code": None,
            "reranked_source_query": None,
            "rank_score": None,
            "rank_reason": None,
            "food_name": None,
            "food_code": None,
            "serving_size": None,
            "energy_kcal": None,
            "carbohydrate_g": None,
            "protein_g": None,
            "fat_g": None,
            "sodium_mg": None,
            "nutrition_field_completeness": 0.0,
            "raw_response_sample": {},
            "latency_seconds": round(time.perf_counter() - started, 4),
            "error_message": str(exc),
        }


def probe_mfds_query(*, query: str, page_size: int, started: float, enable_fallback: bool) -> dict[str, Any]:
    query_variants = build_query_variants(query) if enable_fallback else [query]
    first_payload: dict[str, Any] = {}
    first_items: list[dict[str, Any]] = []
    first_candidates: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    key_mode = None
    operation_url = MFDS_ENDPOINT
    error_messages: list[str] = []

    for index, query_variant in enumerate(query_variants):
        if index > 0 and not should_try_fallback(query=query, candidates=first_candidates):
            break
        try:
            payload, current_key_mode, operation_url = fetch_mfds_payload(query=query_variant, page_size=page_size)
        except Exception as exc:
            error_messages.append(f"{query_variant}: {exc}")
            continue
        key_mode = key_mode or current_key_mode
        items = extract_items(payload)
        candidates = build_candidates(provider="mfds", items=items)
        for candidate in candidates:
            candidate["source_query"] = query_variant
        if index == 0:
            first_payload = payload
            first_items = items
            first_candidates = candidates
        all_candidates.extend(candidates)
        if index == 0 and not should_try_fallback(query=query, candidates=candidates):
            break

    if not first_payload and error_messages:
        return api_failure_row(
            query=query,
            started=started,
            error_message=" | ".join(error_messages),
            parse_failed=any("json" in message.lower() for message in error_messages),
        )

    result = normalize_probe_result(
        provider="mfds",
        query=query,
        items=first_items,
        payload=first_payload,
        candidates=all_candidates,
        original_candidates=first_candidates,
        query_variants=query_variants,
    )
    result["operation_url"] = operation_url
    result["key_mode"] = key_mode
    result["latency_seconds"] = round(time.perf_counter() - started, 4)
    if error_messages:
        result["error_message"] = " | ".join(error_messages)
    return result


def should_try_fallback(*, query: str, candidates: list[dict[str, Any]]) -> bool:
    if not candidates:
        return True
    _, top = rerank_candidates(query=query, candidates=candidates)
    return needs_fallback(top.match_status if top else MatchStatus.NO_CANDIDATES.value)


def api_failure_row(*, query: str, started: float, error_message: str, parse_failed: bool) -> dict[str, Any]:
    status = MatchStatus.PARSE_FAILED.value if parse_failed else MatchStatus.API_UNAVAILABLE.value
    return {
        "provider": "mfds",
        "operation_url": MFDS_ENDPOINT,
        "key_mode": None,
        "query": query,
        "original_query": query,
        "used_query": query,
        "query_variants": [query],
        "fallback_queries": [],
        "fallback_used": False,
        "status": status,
        "match_status": status,
        "needs_user_confirmation": True,
        "candidate_count": 0,
        "top_candidates": [],
        "original_top_candidate": None,
        "reranked_top_candidate": None,
        "original_top_food_name": None,
        "reranked_food_name": None,
        "reranked_food_code": None,
        "reranked_source_query": None,
        "rank_score": None,
        "rank_reason": None,
        "food_name": None,
        "food_code": None,
        "serving_size": None,
        "energy_kcal": None,
        "carbohydrate_g": None,
        "protein_g": None,
        "fat_g": None,
        "sodium_mg": None,
        "nutrition_field_completeness": 0.0,
        "raw_response_sample": {},
        "latency_seconds": round(time.perf_counter() - started, 4),
        "error_message": error_message,
    }


def fetch_provider_payload(*, provider: str, query: str, page_size: int) -> tuple[dict[str, Any], str, str]:
    if provider == "mfds":
        return fetch_mfds_payload(query=query, page_size=page_size)
    payload = request_json(
        RDA_ENDPOINT,
        {
            "apiKey": api_key_for_provider(provider) or "",
            "serviceType": "AA002",
            "nowPage": "1",
            "pageSize": str(page_size),
            "fdNm": query,
        },
    )
    return payload, "plain", RDA_ENDPOINT


def endpoint_for_provider(provider: str) -> str:
    return MFDS_ENDPOINT if provider == "mfds" else RDA_ENDPOINT


def fetch_mfds_payload(*, query: str, page_size: int) -> tuple[dict[str, Any], str, str]:
    base_params = {
        "pageNo": "1",
        "numOfRows": str(page_size),
        "type": "json",
        "FOOD_NM_KR": query,
    }
    decoded_key = api_key_for_provider("mfds")
    encoded_key = encoded_api_key_for_provider("mfds")
    attempts: list[tuple[str, str, bool]] = []
    if decoded_key:
        attempts.append(("decoded", decoded_key, False))
    if encoded_key and encoded_key != decoded_key:
        attempts.append(("encoded", encoded_key, True))

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
                errors.append(f"{key_mode}: authentication failure response")
                continue
            return payload, key_mode, MFDS_ENDPOINT
        except Exception as exc:
            errors.append(f"{key_mode}: {exc}")
    msg = "MFDS request failed with all configured key modes"
    if errors:
        msg = f"{msg}; " + " | ".join(errors)
    raise RuntimeError(msg)


def request_json(
    url: str,
    params: dict[str, str],
    *,
    service_key_param: str | None = None,
    service_key: str | None = None,
    service_key_is_encoded: bool = False,
) -> dict[str, Any]:
    query_string = build_query_string(
        params,
        service_key_param=service_key_param,
        service_key=service_key,
        service_key_is_encoded=service_key_is_encoded,
    )
    request = urllib.request.Request(
        f"{url}?{query_string}",
        headers={"Accept": "application/json", "User-Agent": "AH_03_03-nutrition-api-probe/0.1"},
    )
    last_error: Exception | None = None
    for attempt in range(REQUEST_RETRY_COUNT + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                msg = "API response JSON root is not an object"
                raise ValueError(msg)
            return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= REQUEST_RETRY_COUNT:
                break
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"API request failed: {last_error}")


def build_query_string(
    params: dict[str, str],
    *,
    service_key_param: str | None,
    service_key: str | None,
    service_key_is_encoded: bool,
) -> str:
    query_string = urllib.parse.urlencode(params)
    if not service_key_param or service_key is None:
        return query_string
    encoded_key = service_key if service_key_is_encoded else urllib.parse.quote(service_key, safe="")
    return f"{query_string}&{urllib.parse.quote(service_key_param)}={encoded_key}"


def looks_like_auth_failure(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, ensure_ascii=False).lower()
    auth_markers = [
        "service key",
        "servicekey",
        "인증",
        "등록되지 않은",
        "invalid",
        "unauthorized",
        "auth",
    ]
    result_code = str(
        deep_get(payload, ["response", "header", "resultCode"]) or deep_get(payload, ["header", "resultCode"]) or ""
    ).strip()
    if result_code and result_code not in {"00", "0", "INFO-000"}:
        return any(marker in text for marker in auth_markers)
    return any(marker in text for marker in auth_markers)


def extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        deep_get(payload, ["response", "body", "items", "item"]),
        deep_get(payload, ["response", "body", "items"]),
        deep_get(payload, ["body", "items", "item"]),
        deep_get(payload, ["body", "items"]),
        payload.get("items"),
        payload.get("list"),
        payload.get("row"),
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


def normalize_probe_result(
    *,
    provider: str,
    query: str,
    items: list[dict[str, Any]],
    payload: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
    original_candidates: list[dict[str, Any]] | None = None,
    query_variants: list[str] | None = None,
) -> dict[str, Any]:
    candidates = candidates if candidates is not None else build_candidates(provider=provider, items=items)
    original_candidates = original_candidates if original_candidates is not None else candidates
    ranked_candidates, reranked_top = rerank_candidates(query=query, candidates=candidates)
    original_top_candidate = original_candidates[0] if original_candidates else {}
    top_candidate = reranked_top.candidate if reranked_top else {}
    match_status = reranked_top.match_status if reranked_top else MatchStatus.NO_CANDIDATES.value
    used_query = reranked_top.source_query if reranked_top else query
    query_variants = query_variants or [query]
    fallback_queries = query_variants[1:]
    fallback_used = bool(reranked_top and reranked_top.source_query != query)
    status = MatchStatus.FALLBACK_USED.value if fallback_used else match_status
    return {
        "operation_url": endpoint_for_provider(provider),
        "key_mode": None,
        "provider": provider,
        "query": query,
        "original_query": query,
        "used_query": used_query,
        "query_variants": query_variants,
        "fallback_queries": fallback_queries,
        "fallback_used": fallback_used,
        "status": status,
        "match_status": match_status,
        "needs_user_confirmation": fallback_used or needs_user_confirmation(match_status),
        "candidate_count": len(candidates),
        "top_candidates": serialize_ranked_candidates(ranked_candidates[:5]) if ranked_candidates else candidates[:5],
        "original_top_candidate": original_top_candidate or None,
        "reranked_top_candidate": top_candidate or None,
        "original_top_food_name": original_top_candidate.get("food_name"),
        "reranked_food_name": top_candidate.get("food_name"),
        "reranked_food_code": top_candidate.get("food_code"),
        "reranked_source_query": reranked_top.source_query if reranked_top else None,
        "rank_score": reranked_top.rank_score if reranked_top else None,
        "rank_reason": reranked_top.rank_reason if reranked_top else None,
        "food_name": top_candidate.get("food_name"),
        "food_code": top_candidate.get("food_code"),
        "serving_size": top_candidate.get("serving_size"),
        "energy_kcal": top_candidate.get("energy_kcal"),
        "carbohydrate_g": top_candidate.get("carbohydrate_g"),
        "protein_g": top_candidate.get("protein_g"),
        "fat_g": top_candidate.get("fat_g"),
        "sodium_mg": top_candidate.get("sodium_mg"),
        "nutrition_field_completeness": nutrition_field_completeness(top_candidate),
        "raw_response_sample": sample_response(payload=payload, items=items),
        "error_message": None,
    }


def serialize_ranked_candidates(ranked_candidates: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for ranked_candidate in ranked_candidates:
        candidate = dict(ranked_candidate.candidate)
        candidate["rank_score"] = ranked_candidate.rank_score
        candidate["rank_reason"] = ranked_candidate.rank_reason
        candidate["match_status"] = ranked_candidate.match_status
        candidate["source_query"] = ranked_candidate.source_query
        serialized.append(candidate)
    return serialized


def build_candidates(*, provider: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if provider == "rda":
        return build_rda_candidates(items)
    return [build_row_candidate(item) for item in items]


def nutrition_field_completeness(candidate: dict[str, Any]) -> float:
    fields = ["energy_kcal", "carbohydrate_g", "protein_g", "fat_g", "sodium_mg"]
    if not candidate:
        return 0.0
    present_count = sum(candidate.get(field) is not None for field in fields)
    return round(present_count / len(fields), 4)


def build_rda_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    nutrient_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        food_code = first_value(item, ["fdCode", "foodCode", "FOOD_CD"]) or ""
        food_name = first_value(item, ["fdNm", "foodName", "FOOD_NM"]) or ""
        key = (food_code, food_name)
        if key not in grouped:
            grouped[key] = build_row_candidate(item)
        nutrient_rows[key].append(item)

    candidates: list[dict[str, Any]] = []
    for key, candidate in grouped.items():
        for item in nutrient_rows[key]:
            nutrient_name = str(first_value(item, ["irdntNm", "nutrientName", "NUTRIENT_NM"]) or "")
            value = parse_float(first_value(item, ["contInfo", "content", "value"]))
            apply_rda_nutrient(candidate, nutrient_name=nutrient_name, value=value)
        candidates.append(candidate)
    return candidates


def build_row_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "food_name": first_value(
            item,
            [
                "FOOD_NM_KR",
                "FOOD_NM",
                "foodNm",
                "foodName",
                "DESC_KOR",
                "fdNm",
                "FD_NM",
                "foodNmKr",
            ],
        ),
        "food_code": first_value(
            item,
            ["FOOD_CD", "FOOD_CODE", "foodCd", "foodCode", "fdCode", "FD_CODE", "foodCd"],
        ),
        "serving_size": first_value(
            item,
            [
                "SERVING_SIZE",
                "servingSize",
                "NUT_CON_TP",
                "NUTR_CONT_UNIT",
                "foodSize",
                "FD_WGH",
                "servingSize",
            ],
        ),
        "energy_kcal": parse_float(
            first_value(
                item,
                ["ENERC_KCAL", "ENERC", "energyKcal", "NUTR_CONT1", "AMT_NUM1", "energy", "calorie"],
            )
        ),
        "carbohydrate_g": parse_float(
            first_value(
                item,
                ["CHOCDF_G", "CHOCDF", "carbohydrate", "NUTR_CONT2", "AMT_NUM6", "carbohydrateG"],
            )
        ),
        "protein_g": parse_float(
            first_value(item, ["PROT_G", "PROT", "protein", "NUTR_CONT3", "AMT_NUM3", "proteinG"])
        ),
        "fat_g": parse_float(first_value(item, ["FATCE_G", "FATCE", "fat", "NUTR_CONT4", "AMT_NUM4", "fatG"])),
        "sodium_mg": parse_float(
            first_value(item, ["NATR_MG", "NATR", "sodium", "NUTR_CONT6", "AMT_NUM13", "sodiumMg"])
        ),
    }


def apply_rda_nutrient(candidate: dict[str, Any], *, nutrient_name: str, value: float | None) -> None:
    if value is None:
        return
    normalized = nutrient_name.replace(" ", "").lower()
    if "에너지" in normalized or "열량" in normalized or "energy" in normalized:
        candidate["energy_kcal"] = value
    elif "탄수화물" in normalized or "carbohydrate" in normalized:
        candidate["carbohydrate_g"] = value
    elif "단백질" in normalized or "protein" in normalized:
        candidate["protein_g"] = value
    elif "지방" in normalized or "fat" in normalized:
        candidate["fat_g"] = value
    elif "나트륨" in normalized or "sodium" in normalized:
        candidate["sodium_mg"] = value


def first_value(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in {None, ""}:
            return value
    lower_index = {str(key).lower(): value for key, value in item.items()}
    for key in keys:
        value = lower_index.get(key.lower())
        if value not in {None, ""}:
            return value
    return None


def parse_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def sample_response(*, payload: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    header = deep_get(payload, ["response", "header"]) or payload.get("header")
    return {
        "header": header if isinstance(header, dict) else None,
        "first_item": items[0] if items else None,
    }


def deep_get(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: serialize_cell(row.get(column)) for column in OUTPUT_COLUMNS})


def serialize_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    status_counts = count_by(rows, "match_status")
    confirmation_rows = [row for row in rows if row.get("needs_user_confirmation")]
    fallback_rows = [row for row in rows if row.get("fallback_used")]
    lines = [
        "# Public Nutrition API Probe Report",
        "",
        "This report is experiment-only and is not connected to production runtime.",
        "",
        "## Summary",
        "",
        f"- total_requests: {total}",
        *[f"- {status}: {count}" for status, count in sorted(status_counts.items())],
        f"- needs_user_confirmation: {len(confirmation_rows)}",
        f"- fallback_used: {len(fallback_rows)}",
        "",
        "## Operation URLs",
        "",
    ]
    for provider in sorted({str(row.get("provider") or "") for row in rows}):
        if not provider:
            continue
        lines.append(f"- {provider}: `{endpoint_for_provider(provider)}`")
    key_modes = sorted({str(row.get("key_mode") or "") for row in rows if row.get("key_mode")})
    if key_modes:
        lines.extend(["", f"- successful_key_modes: {', '.join(key_modes)}"])
    lines.extend(["", "API keys are not written to this report.", ""])
    if any(row.get("provider") == "mfds" for row in rows):
        lines.extend(
            [
                "## MFDS Field Mapping",
                "",
                *[f"- {field}: `{source}`" for field, source in MFDS_FIELD_MAPPING.items()],
                "",
            ]
        )
    lines.extend(
        [
            "## Original Top1 vs Reranked Top1",
            "",
            "| provider | key_mode | original_query | used_query | status | match_status | fallback_used | candidate_count | API top | reranked top | score | reason | nutrition_completeness |",
            "|---|---|---|---|---|---|---|---:|---|---|---:|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {provider} | {key_mode} | {original_query} | {used_query} | {status} | {match_status} | "
            "{fallback_used} | {candidate_count} | {original_top_food_name} | {food_name} | {rank_score} | "
            "{rank_reason} | {nutrition_field_completeness} |".format(
                provider=row.get("provider") or "",
                key_mode=row.get("key_mode") or "",
                original_query=row.get("original_query") or row.get("query") or "",
                used_query=row.get("used_query") or "",
                status=row.get("status") or "",
                match_status=row.get("match_status") or "",
                fallback_used=row.get("fallback_used"),
                candidate_count=row.get("candidate_count") or 0,
                original_top_food_name=row.get("original_top_food_name") or "",
                food_name=row.get("food_name") or "",
                rank_score=display_value(row.get("rank_score")),
                rank_reason=row.get("rank_reason") or "",
                nutrition_field_completeness=display_value(row.get("nutrition_field_completeness")),
            )
        )
    append_food_list_section(
        lines,
        title="Automatically Acceptable Candidates",
        rows=[
            row for row in rows if row.get("match_status") == MatchStatus.MATCHED.value and not row.get("fallback_used")
        ],
    )
    append_food_list_section(
        lines,
        title="Likely Match Candidates",
        rows=[row for row in rows if row.get("match_status") == MatchStatus.LIKELY_MATCH.value],
    )
    append_food_list_section(
        lines,
        title="Needs User Confirmation Candidates",
        rows=confirmation_rows,
    )
    append_food_list_section(
        lines,
        title="No Candidates",
        rows=[row for row in rows if row.get("match_status") == MatchStatus.NO_CANDIDATES.value],
    )
    append_food_list_section(lines, title="Fallback Used", rows=fallback_rows)
    risky_rows = [
        row
        for row in rows
        if row.get("match_status")
        in {
            MatchStatus.WEAK_MATCH.value,
            MatchStatus.MULTIPLE_CANDIDATES.value,
            MatchStatus.NEEDS_USER_CONFIRMATION.value,
        }
    ]
    append_food_list_section(lines, title="Risky With MFDS Only", rows=risky_rows)
    lines.extend(
        [
            "",
            "## Before Production Integration",
            "",
            "- Treat `matched` as the only auto-accept candidate in this experiment.",
            "- Route `likely_match`, `multiple_candidates`, and `weak_match` to user confirmation or candidate selection.",
            "- Add query normalization for generic ingredients, beverages, colors, and romanized names.",
            "- Compare MFDS against RDA/FoodNara-style APIs before using a single provider in production.",
            "- The processed-food penalty keywords are experimental heuristics and need validation on a larger sample.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_value(value: Any) -> str:
    return "" if value is None else str(value)


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def append_food_list_section(lines: list[str], *, title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend(["", f"## {title}", ""])
    if not rows:
        lines.append("- none")
        return
    for row in rows:
        original_query = row.get("original_query") or row.get("query") or ""
        used_query = row.get("used_query") or ""
        food_name = row.get("food_name") or ""
        match_status = row.get("match_status") or ""
        score = display_value(row.get("rank_score"))
        lines.append(f"- {original_query} -> {food_name} ({match_status}, score={score}, used_query={used_query})")


if __name__ == "__main__":
    sys.exit(main())
