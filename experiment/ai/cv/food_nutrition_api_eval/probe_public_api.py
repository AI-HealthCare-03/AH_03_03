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

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES = [
    "가래떡",
    "가리비",
    "샌드위치",
    "blt샌드위치",
    "파프리카",
    "자몽",
    "홍차",
    "핫초코",
    "냉면",
    "미역국",
]
OUTPUT_COLUMNS = [
    "provider",
    "operation_url",
    "key_mode",
    "query",
    "status",
    "candidate_count",
    "top_candidates",
    "food_name",
    "food_code",
    "serving_size",
    "energy_kcal",
    "carbohydrate_g",
    "protein_g",
    "fat_g",
    "sodium_mg",
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
            rows.append(probe_query(provider=provider, query=query, page_size=args.page_size))

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


def probe_query(*, provider: str, query: str, page_size: int) -> dict[str, Any]:
    started = time.perf_counter()
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
            "status": "api_error",
            "candidate_count": 0,
            "top_candidates": [],
            "food_name": None,
            "food_code": None,
            "serving_size": None,
            "energy_kcal": None,
            "carbohydrate_g": None,
            "protein_g": None,
            "fat_g": None,
            "sodium_mg": None,
            "raw_response_sample": {},
            "latency_seconds": round(time.perf_counter() - started, 4),
            "error_message": str(exc),
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
) -> dict[str, Any]:
    candidates = build_candidates(provider=provider, items=items)
    top_candidate = candidates[0] if candidates else {}
    status = "success" if candidates else "no_candidates"
    return {
        "operation_url": endpoint_for_provider(provider),
        "key_mode": None,
        "provider": provider,
        "query": query,
        "status": status,
        "candidate_count": len(candidates),
        "top_candidates": candidates[:5],
        "food_name": top_candidate.get("food_name"),
        "food_code": top_candidate.get("food_code"),
        "serving_size": top_candidate.get("serving_size"),
        "energy_kcal": top_candidate.get("energy_kcal"),
        "carbohydrate_g": top_candidate.get("carbohydrate_g"),
        "protein_g": top_candidate.get("protein_g"),
        "fat_g": top_candidate.get("fat_g"),
        "sodium_mg": top_candidate.get("sodium_mg"),
        "raw_response_sample": sample_response(payload=payload, items=items),
        "error_message": None,
    }


def build_candidates(*, provider: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if provider == "rda":
        return build_rda_candidates(items)
    return [build_row_candidate(item) for item in items]


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
    success = sum(row.get("status") == "success" for row in rows)
    no_candidates = sum(row.get("status") == "no_candidates" for row in rows)
    api_error = sum(row.get("status") == "api_error" for row in rows)
    lines = [
        "# Public Nutrition API Probe Report",
        "",
        "This report is experiment-only and is not connected to production runtime.",
        "",
        "## Summary",
        "",
        f"- total_requests: {total}",
        f"- success: {success}",
        f"- no_candidates: {no_candidates}",
        f"- api_error: {api_error}",
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
            "## Results",
            "",
            "| provider | key_mode | query | status | candidate_count | top food | energy_kcal | carbs_g | protein_g | fat_g | sodium_mg |",
            "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {provider} | {key_mode} | {query} | {status} | {candidate_count} | {food_name} | {energy_kcal} | "
            "{carbohydrate_g} | {protein_g} | {fat_g} | {sodium_mg} |".format(
                provider=row.get("provider") or "",
                key_mode=row.get("key_mode") or "",
                query=row.get("query") or "",
                status=row.get("status") or "",
                candidate_count=row.get("candidate_count") or 0,
                food_name=row.get("food_name") or "",
                energy_kcal=display_value(row.get("energy_kcal")),
                carbohydrate_g=display_value(row.get("carbohydrate_g")),
                protein_g=display_value(row.get("protein_g")),
                fat_g=display_value(row.get("fat_g")),
                sodium_mg=display_value(row.get("sodium_mg")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def display_value(value: Any) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":
    sys.exit(main())
