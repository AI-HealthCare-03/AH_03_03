from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any

from schemas import PredictionRow

from ai_runtime.cv.food.matcher import match_food_name
from ai_runtime.cv.food.normalization import normalize_food_name

CONFIDENCE_BINS = [
    ("0.0-0.3", 0.0, 0.3),
    ("0.3-0.5", 0.3, 0.5),
    ("0.5-0.7", 0.5, 0.7),
    ("0.7-0.9", 0.7, 0.9),
    ("0.9-1.0", 0.9, 1.000001),
]


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


def compute_metrics(predictions: list[PredictionRow]) -> dict[str, Any]:
    total_rows = len(predictions)
    data_missing_count = sum(is_data_missing(row) for row in predictions)
    evaluable_predictions = [row for row in predictions if not is_data_missing(row)]
    evaluable_image_count = len(evaluable_predictions)
    api_failed_count = sum(is_api_failed(row) for row in evaluable_predictions)
    json_parse_failed_count = sum(is_json_parse_failed(row) for row in evaluable_predictions)
    empty_result_count = sum(
        row.empty_result and row.api_success and row.json_parse_success for row in evaluable_predictions
    )
    invalid_label_count = sum(row.invalid_label_count for row in evaluable_predictions)
    unknown_count = sum(food == "unknown" for row in evaluable_predictions for food in row.allowed_food_names)
    predicted_food_count = sum(len(row.raw_food_names) for row in evaluable_predictions)
    latencies = [row.latency_seconds for row in evaluable_predictions if row.latency_seconds is not None]
    confidences = [row.confidence for row in evaluable_predictions if row.confidence is not None]

    micro = aggregate_precision_recall_f1(evaluable_predictions)
    class_metrics = class_level_metrics(evaluable_predictions)
    macro = macro_scores(class_metrics)
    confidence_metrics = confidence_summary(evaluable_predictions)

    return {
        "total_rows": total_rows,
        "evaluable_image_count": evaluable_image_count,
        "data_missing_count": data_missing_count,
        "api_failed_count": api_failed_count,
        "json_parse_failed_count": json_parse_failed_count,
        "empty_result_count": empty_result_count,
        "api_success_rate": _rate(sum(row.api_success for row in evaluable_predictions), evaluable_image_count),
        "json_parse_success_rate": _rate(
            sum(row.json_parse_success for row in evaluable_predictions),
            evaluable_image_count,
        ),
        "raw_exact_match_rate": _rate(
            sum(is_exact_raw_match(row) for row in evaluable_predictions), evaluable_image_count
        ),
        "canonical_exact_match_rate": _rate(
            sum(is_exact_canonical_match(row) for row in evaluable_predictions),
            evaluable_image_count,
        ),
        "constrained_exact_match_rate": _rate(
            sum(is_exact_allowed_match(row) for row in evaluable_predictions),
            evaluable_image_count,
        ),
        "any_hit_rate": _rate(sum(has_any_food_hit(row) for row in evaluable_predictions), evaluable_image_count),
        "canonical_any_hit_rate": _rate(
            sum(has_any_canonical_hit(row) for row in evaluable_predictions),
            evaluable_image_count,
        ),
        "constrained_any_hit_rate": _rate(
            sum(has_any_allowed_hit(row) for row in evaluable_predictions),
            evaluable_image_count,
        ),
        "precision": micro["precision"],
        "recall": micro["recall"],
        "f1_score": micro["f1_score"],
        "macro_precision": macro["macro_precision"],
        "macro_recall": macro["macro_recall"],
        "macro_f1_score": macro["macro_f1_score"],
        "invalid_label_count": invalid_label_count,
        "invalid_label_rate": _rate(invalid_label_count, predicted_food_count),
        "constrained_by_allowed_foods": any(row.constrained_by_allowed_foods for row in predictions),
        "unknown_count": unknown_count,
        "unknown_rate": _rate(unknown_count, predicted_food_count),
        "unmatched_food_rate": _rate(
            sum(len(row.unmatched_food_names) for row in evaluable_predictions),
            predicted_food_count,
        ),
        "empty_result_rate": _rate(empty_result_count, evaluable_image_count),
        "avg_confidence": round(mean(confidences), 4) if confidences else 0.0,
        "confidence_correct_avg": confidence_metrics["confidence_correct_avg"],
        "confidence_wrong_avg": confidence_metrics["confidence_wrong_avg"],
        "confidence_bins": confidence_metrics["confidence_bins"],
        "avg_latency_seconds": round(mean(latencies), 4) if latencies else 0.0,
        "p50_latency_seconds": round(median(latencies), 4) if latencies else 0.0,
        "p95_latency_seconds": round(_p95(latencies), 4) if latencies else 0.0,
        "max_latency_seconds": round(max(latencies), 4) if latencies else 0.0,
        "class_distribution": class_distribution(predictions),
        "class_level_metrics": class_metrics,
    }


def class_distribution(predictions: list[PredictionRow]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in predictions:
        counter.update(row.expected_foods)
    return dict(counter.most_common())


def row_precision_recall_f1(row: PredictionRow) -> dict[str, float]:
    expected = normalized_food_set(row.expected_foods)
    predicted = selected_prediction_set(row)
    if not expected and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
    true_positive = len(expected & predicted)
    precision = _rate(true_positive, len(predicted))
    recall = _rate(true_positive, len(expected))
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": _f1(precision, recall),
    }


def aggregate_precision_recall_f1(rows: list[PredictionRow]) -> dict[str, float]:
    expected_total = 0
    predicted_total = 0
    true_positive_total = 0
    for row in rows:
        expected = normalized_food_set(row.expected_foods)
        predicted = selected_prediction_set(row)
        expected_total += len(expected)
        predicted_total += len(predicted)
        true_positive_total += len(expected & predicted)
    precision = _rate(true_positive_total, predicted_total)
    recall = _rate(true_positive_total, expected_total)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": _f1(precision, recall),
    }


def class_level_metrics(rows: list[PredictionRow]) -> list[dict[str, float | int | str]]:
    expected_classes = sorted({food for row in rows for food in row.expected_foods})
    metrics: list[dict[str, float | int | str]] = []
    for food_name in expected_classes:
        normalized_food = normalize_food_name(food_name)
        sample_rows = [row for row in rows if normalized_food in normalized_food_set(row.expected_foods)]
        true_positive = sum(normalized_food in selected_prediction_set(row) for row in rows if row in sample_rows)
        false_negative = len(sample_rows) - true_positive
        false_positive = sum(
            normalized_food not in normalized_food_set(row.expected_foods)
            and normalized_food in selected_prediction_set(row)
            for row in rows
        )
        precision = _rate(true_positive, true_positive + false_positive)
        recall = _rate(true_positive, true_positive + false_negative)
        metrics.append(
            {
                "expected_food": food_name,
                "sample_count": len(sample_rows),
                "exact_match_rate": _rate(sum(is_exact_canonical_match(row) for row in sample_rows), len(sample_rows)),
                "any_hit_rate": _rate(sum(has_any_canonical_hit(row) for row in sample_rows), len(sample_rows)),
                "precision": precision,
                "recall": recall,
                "f1_score": _f1(precision, recall),
            }
        )
    return metrics


def macro_scores(class_metrics: list[dict[str, float | int | str]]) -> dict[str, float]:
    if not class_metrics:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1_score": 0.0,
        }
    return {
        "macro_precision": round(mean(float(row["precision"]) for row in class_metrics), 4),
        "macro_recall": round(mean(float(row["recall"]) for row in class_metrics), 4),
        "macro_f1_score": round(mean(float(row["f1_score"]) for row in class_metrics), 4),
    }


def confidence_summary(rows: list[PredictionRow]) -> dict[str, Any]:
    correct_confidences = [
        row.confidence for row in rows if row.confidence is not None and is_exact_canonical_match(row)
    ]
    wrong_confidences = [
        row.confidence for row in rows if row.confidence is not None and not is_exact_canonical_match(row)
    ]
    bins: dict[str, dict[str, float | int]] = {}
    for label, lower, upper in CONFIDENCE_BINS:
        bin_rows = [row for row in rows if row.confidence is not None and lower <= row.confidence < upper]
        bins[label] = {
            "sample_count": len(bin_rows),
            "accuracy": _rate(sum(is_exact_canonical_match(row) for row in bin_rows), len(bin_rows)),
        }
    return {
        "confidence_correct_avg": round(mean(correct_confidences), 4) if correct_confidences else 0.0,
        "confidence_wrong_avg": round(mean(wrong_confidences), 4) if wrong_confidences else 0.0,
        "confidence_bins": bins,
    }


def selected_prediction_set(row: PredictionRow) -> set[str]:
    if row.constrained_by_allowed_foods and row.allowed_food_names:
        return normalized_food_set(food for food in row.allowed_food_names if food != "unknown")
    if row.canonical_food_names:
        return normalized_food_set(row.canonical_food_names)
    return normalized_food_set(row.raw_food_names)


def is_exact_raw_match(row: PredictionRow) -> bool:
    return normalized_food_set(row.raw_food_names) == normalized_food_set(row.expected_foods)


def is_exact_canonical_match(row: PredictionRow) -> bool:
    return normalized_food_set(row.canonical_food_names) == normalized_food_set(
        canonicalize_food_names(row.expected_foods)
    )


def is_exact_allowed_match(row: PredictionRow) -> bool:
    return normalized_food_set(row.allowed_food_names) == normalized_food_set(row.expected_foods)


def has_any_food_hit(row: PredictionRow) -> bool:
    expected = normalized_food_set(row.expected_foods) | normalized_food_set(
        canonicalize_food_names(row.expected_foods)
    )
    predicted = normalized_food_set(row.raw_food_names) | normalized_food_set(row.canonical_food_names)
    return bool(expected & predicted)


def has_any_canonical_hit(row: PredictionRow) -> bool:
    expected = normalized_food_set(canonicalize_food_names(row.expected_foods))
    predicted = normalized_food_set(row.canonical_food_names)
    return bool(expected & predicted)


def has_any_allowed_hit(row: PredictionRow) -> bool:
    expected = normalized_food_set(row.expected_foods)
    predicted = normalized_food_set(row.allowed_food_names)
    return bool(expected & predicted)


def is_data_missing(row: PredictionRow) -> bool:
    return row.error_type == "data_missing"


def is_api_failed(row: PredictionRow) -> bool:
    return row.error_type not in {None, "data_missing", "GptJsonParseError"} and not row.api_success


def is_json_parse_failed(row: PredictionRow) -> bool:
    return row.api_success and not row.json_parse_success


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(0.95 * (len(sorted_values) - 1)))
    return sorted_values[index]
