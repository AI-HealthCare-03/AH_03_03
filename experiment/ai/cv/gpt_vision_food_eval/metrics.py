from __future__ import annotations

from collections import Counter
from statistics import mean

from schemas import PredictionRow

from ai_runtime.cv.food.matcher import match_food_name
from ai_runtime.cv.food.normalization import normalize_food_name


def split_food_names(value: str) -> list[str]:
    return [item.strip() for item in value.replace("|", ",").split(",") if item.strip()]


def canonicalize_food_names(food_names: list[str]) -> list[str]:
    canonical: list[str] = []
    for food_name in food_names:
        match = match_food_name(food_name)
        canonical_name = match.matched_food_name or match.query_name
        if canonical_name:
            canonical.append(canonical_name)
    return canonical


def normalized_food_set(food_names: list[str]) -> set[str]:
    return {normalized for name in food_names if (normalized := normalize_food_name(name))}


def compute_metrics(predictions: list[PredictionRow]) -> dict[str, float | int | dict[str, int]]:
    total_rows = len(predictions)
    data_missing_count = sum(_is_data_missing(row) for row in predictions)
    evaluable_predictions = [row for row in predictions if not _is_data_missing(row)]
    evaluable_image_count = len(evaluable_predictions)
    if evaluable_image_count == 0:
        return {
            "total_rows": total_rows,
            "evaluable_image_count": 0,
            "data_missing_count": data_missing_count,
            "api_success_rate": 0.0,
            "json_parse_success_rate": 0.0,
            "exact_match_rate_raw": 0.0,
            "exact_match_rate_canonical": 0.0,
            "any_food_hit_rate": 0.0,
            "unmatched_food_rate": 0.0,
            "empty_result_rate": 0.0,
            "avg_latency_seconds": 0.0,
            "p95_latency_seconds": 0.0,
            "class_distribution": class_distribution(predictions),
        }

    latencies = [row.latency_seconds for row in evaluable_predictions if row.latency_seconds is not None]
    predicted_food_count = sum(len(row.raw_food_names) for row in evaluable_predictions)
    unmatched_food_count = sum(len(row.unmatched_food_names) for row in evaluable_predictions)

    return {
        "total_rows": total_rows,
        "evaluable_image_count": evaluable_image_count,
        "data_missing_count": data_missing_count,
        "api_success_rate": _rate(sum(row.api_success for row in evaluable_predictions), evaluable_image_count),
        "json_parse_success_rate": _rate(
            sum(row.json_parse_success for row in evaluable_predictions),
            evaluable_image_count,
        ),
        "exact_match_rate_raw": _rate(
            sum(_is_exact_raw_match(row) for row in evaluable_predictions),
            evaluable_image_count,
        ),
        "exact_match_rate_canonical": _rate(
            sum(_is_exact_canonical_match(row) for row in evaluable_predictions),
            evaluable_image_count,
        ),
        "any_food_hit_rate": _rate(sum(_has_any_food_hit(row) for row in evaluable_predictions), evaluable_image_count),
        "unmatched_food_rate": _rate(unmatched_food_count, predicted_food_count),
        "empty_result_rate": _rate(sum(row.empty_result for row in evaluable_predictions), evaluable_image_count),
        "avg_latency_seconds": round(mean(latencies), 4) if latencies else 0.0,
        "p95_latency_seconds": round(_p95(latencies), 4) if latencies else 0.0,
        "class_distribution": class_distribution(predictions),
    }


def class_distribution(predictions: list[PredictionRow]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in predictions:
        counter.update(row.expected_foods)
    return dict(counter.most_common())


def _is_exact_raw_match(row: PredictionRow) -> bool:
    return normalized_food_set(row.raw_food_names) == normalized_food_set(row.expected_foods)


def _is_exact_canonical_match(row: PredictionRow) -> bool:
    return normalized_food_set(row.canonical_food_names) == normalized_food_set(
        canonicalize_food_names(row.expected_foods)
    )


def _has_any_food_hit(row: PredictionRow) -> bool:
    expected = normalized_food_set(row.expected_foods) | normalized_food_set(
        canonicalize_food_names(row.expected_foods)
    )
    predicted = normalized_food_set(row.raw_food_names) | normalized_food_set(row.canonical_food_names)
    return bool(expected & predicted)


def _is_data_missing(row: PredictionRow) -> bool:
    return row.error_type == "data_missing"


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(0.95 * (len(sorted_values) - 1)))
    return sorted_values[index]
