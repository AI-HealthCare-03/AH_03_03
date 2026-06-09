from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from providers.base import NutritionProvider
from providers.local_cache import LocalJsonNutritionCache
from providers.stub_provider import StubNutritionProvider
from schemas import NutritionLookupResult

from ai_runtime.cv.food.normalization import normalize_food_name

OUTPUT_COLUMNS = [
    "row_id",
    "image_path",
    "expected_foods",
    "query",
    "normalized_query",
    "provider",
    "status",
    "matched_food_name",
    "matched_food_code",
    "candidate_count",
    "top_candidates",
    "energy_kcal",
    "carbohydrate_g",
    "protein_g",
    "fat_g",
    "sodium_mg",
    "serving_size",
    "source",
    "latency_seconds",
    "error_message",
    "cache_hit",
    "needs_user_confirmation",
    "top1_food_match",
    "top3_candidate_hit",
    "nutrition_field_completeness",
    "score_available",
]

NUTRITION_FIELDS = ["energy_kcal", "carbohydrate_g", "protein_g", "fat_g", "sodium_mg"]
SUCCESS_STATUSES = {"matched", "multiple_candidates"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate food nutrition API lookup using GPT Vision predictions.")
    parser.add_argument("--predictions", required=True, help="GPT Vision predictions.csv path.")
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR / "outputs"), help="Output directory.")
    parser.add_argument("--provider", choices=["stub"], default="stub", help="Nutrition provider to use.")
    parser.add_argument("--limit", type=int, default=None, help="Limit prediction rows for smoke runs.")
    parser.add_argument("--cache-path", default=None, help="Optional local JSON cache path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = build_provider(args.provider)
    cache_path = Path(args.cache_path) if args.cache_path else output_dir / "nutrition_lookup_cache.json"
    cache = LocalJsonNutritionCache(cache_path)

    prediction_rows = load_prediction_rows(Path(args.predictions), limit=args.limit)
    output_rows: list[dict[str, object]] = []
    for row in prediction_rows:
        query_candidates = build_query_candidates(row)
        if not query_candidates:
            query_candidates = [""]
        for query in query_candidates:
            result = lookup_with_cache(provider, cache, query)
            output_rows.append(build_output_row(row, result))
    cache.flush()

    metrics = compute_metrics(output_rows)
    write_predictions_csv(output_dir / "nutrition_predictions.csv", output_rows)
    write_metrics_json(output_dir / "nutrition_metrics.json", metrics)
    write_report(output_dir / "nutrition_report.md", metrics)
    print(f"Wrote {output_dir / 'nutrition_predictions.csv'}")
    print(f"Wrote {output_dir / 'nutrition_metrics.json'}")
    print(f"Wrote {output_dir / 'nutrition_report.md'}")


def build_provider(provider_name: str) -> NutritionProvider:
    if provider_name == "stub":
        return StubNutritionProvider()
    msg = f"Unsupported provider: {provider_name}"
    raise ValueError(msg)


def load_prediction_rows(path: Path, *, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key: str(value or "") for key, value in row.items()})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_query_candidates(row: dict[str, str]) -> list[str]:
    candidates: list[str] = []
    for column in ("raw_food_names", "canonical_food_names", "allowed_food_names"):
        candidates.extend(split_food_names(row.get(column, "")))
    return dedupe(candidates)


def lookup_with_cache(
    provider: NutritionProvider,
    cache: LocalJsonNutritionCache,
    query: str,
) -> NutritionLookupResult:
    if query:
        cached = cache.get(query)
        if cached is not None:
            return cached
    result = provider.lookup(query)
    if query:
        cache.set(query, result)
    return result


def build_output_row(row: dict[str, str], result: NutritionLookupResult) -> dict[str, object]:
    expected_foods = split_food_names(row.get("expected_foods", ""))
    return {
        "row_id": row.get("row_id", ""),
        "image_path": row.get("image_path", ""),
        "expected_foods": "|".join(expected_foods),
        "query": result.query,
        "normalized_query": result.normalized_query,
        "provider": result.provider,
        "status": result.status,
        "matched_food_name": result.matched_food_name or "",
        "matched_food_code": result.matched_food_code or "",
        "candidate_count": result.candidate_count,
        "top_candidates": "|".join(candidate.food_name for candidate in result.top_candidates),
        "energy_kcal": result.energy_kcal if result.energy_kcal is not None else "",
        "carbohydrate_g": result.carbohydrate_g if result.carbohydrate_g is not None else "",
        "protein_g": result.protein_g if result.protein_g is not None else "",
        "fat_g": result.fat_g if result.fat_g is not None else "",
        "sodium_mg": result.sodium_mg if result.sodium_mg is not None else "",
        "serving_size": result.serving_size or "",
        "source": result.source or "",
        "latency_seconds": result.latency_seconds,
        "error_message": result.error_message or "",
        "cache_hit": result.cache_hit,
        "needs_user_confirmation": result.needs_user_confirmation,
        "top1_food_match": top1_food_match(result, expected_foods),
        "top3_candidate_hit": top3_candidate_hit(result, expected_foods),
        "nutrition_field_completeness": nutrition_field_completeness(result),
        "score_available": score_available(result),
    }


def compute_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    row_count = len(rows)
    success_rows = [row for row in rows if str(row["status"]) in SUCCESS_STATUSES]
    latencies = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") != ""]
    return {
        "row_count": row_count,
        "nutrition_lookup_success_rate": rate(len(success_rows), row_count),
        "top1_food_match_rate": rate(sum(bool(row["top1_food_match"]) for row in rows), row_count),
        "top3_candidate_hit_rate": rate(sum(bool(row["top3_candidate_hit"]) for row in rows), row_count),
        "multiple_candidate_rate": rate(sum(str(row["status"]) == "multiple_candidates" for row in rows), row_count),
        "needs_user_confirmation_rate": rate(sum(bool(row["needs_user_confirmation"]) for row in rows), row_count),
        "api_failure_rate": rate(sum(str(row["status"]) == "api_error" for row in rows), row_count),
        "cache_hit_rate": rate(sum(bool(row["cache_hit"]) for row in rows), row_count),
        "avg_lookup_latency": round(mean(latencies), 4) if latencies else 0.0,
        "nutrition_field_completeness": round(mean(float(row["nutrition_field_completeness"]) for row in rows), 4)
        if rows
        else 0.0,
        "score_available_rate": rate(sum(bool(row["score_available"]) for row in rows), row_count),
        "status_counts": status_counts(rows),
    }


def write_predictions_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_json(path: Path, metrics: dict[str, object]) -> None:
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, metrics: dict[str, object]) -> None:
    lines = [
        "# Food Nutrition API Lookup Eval Report",
        "",
        "## Summary",
        "",
        f"- row_count: {metrics['row_count']}",
        f"- nutrition_lookup_success_rate: {metrics['nutrition_lookup_success_rate']}",
        f"- top1_food_match_rate: {metrics['top1_food_match_rate']}",
        f"- top3_candidate_hit_rate: {metrics['top3_candidate_hit_rate']}",
        f"- multiple_candidate_rate: {metrics['multiple_candidate_rate']}",
        f"- needs_user_confirmation_rate: {metrics['needs_user_confirmation_rate']}",
        f"- api_failure_rate: {metrics['api_failure_rate']}",
        f"- cache_hit_rate: {metrics['cache_hit_rate']}",
        f"- avg_lookup_latency: {metrics['avg_lookup_latency']}",
        f"- nutrition_field_completeness: {metrics['nutrition_field_completeness']}",
        f"- score_available_rate: {metrics['score_available_rate']}",
        "",
        "## Status Counts",
        "",
    ]
    status_payload = metrics.get("status_counts", {})
    if isinstance(status_payload, dict):
        lines.extend(f"- {status}: {count}" for status, count in status_payload.items())
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This experiment is not connected to production runtime.",
            "- The stub provider is a fake response provider for E2E flow validation only.",
            "- MFDS/RDA API clients should be implemented and validated in this experiment before service integration.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def top1_food_match(result: NutritionLookupResult, expected_foods: list[str]) -> bool:
    if not result.matched_food_name:
        return False
    expected = normalized_set(expected_foods)
    return normalize_food_name(result.matched_food_name) in expected


def top3_candidate_hit(result: NutritionLookupResult, expected_foods: list[str]) -> bool:
    expected = normalized_set(expected_foods)
    candidates = {normalize_food_name(candidate.food_name) for candidate in result.top_candidates}
    return bool(expected & candidates)


def nutrition_field_completeness(result: NutritionLookupResult) -> float:
    values = result.to_dict()
    present = sum(values.get(field) is not None for field in NUTRITION_FIELDS)
    return round(present / len(NUTRITION_FIELDS), 4)


def score_available(result: NutritionLookupResult) -> bool:
    return result.status in SUCCESS_STATUSES and nutrition_field_completeness(result) >= 1.0


def status_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def normalized_set(values: list[str]) -> set[str]:
    return {normalized for value in values if (normalized := normalize_food_name(value))}


def split_food_names(value: str) -> list[str]:
    return [item.strip() for item in value.replace("|", ",").split(",") if item.strip()]


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


if __name__ == "__main__":
    main()
