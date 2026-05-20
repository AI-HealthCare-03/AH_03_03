import csv
import json
import sys
from pathlib import Path
from typing import Any

CLOVA_OCR_ROOT = Path(__file__).resolve().parents[1]
if str(CLOVA_OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(CLOVA_OCR_ROOT))

from parsers.health_exam_result_parser import VALUE_FIELDS, parse_health_exam_result  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CLOVA_OCR_DIR = PROJECT_ROOT / "ai_worker" / "vision" / "clova_ocr"
OUTPUT_ROOT_DIR = CLOVA_OCR_DIR / "outputs"
GROUND_TRUTH_DIR = CLOVA_OCR_DIR / "ground_truth"
SUMMARY_JSON_PATH = OUTPUT_ROOT_DIR / "value_accuracy_summary.json"
SUMMARY_CSV_PATH = OUTPUT_ROOT_DIR / "value_accuracy_summary.csv"

CSV_COLUMNS = [
    "pdf_stem",
    "source",
    "field_count",
    "predicted_count",
    "matched_count",
    "missing_prediction_count",
    "wrong_value_count",
    "field_coverage",
    "value_accuracy",
    "precision_like_accuracy",
]


def main() -> None:
    """Evaluate parsed OCR values against human-written ground truth JSON.

    This evaluation is valid only when `ground_truth/{pdf_stem}.json` is filled
    manually. Ground truth fields set to null are excluded from scoring.

    Run after OCR outputs and ground truth JSON files exist:
        uv run python ai_worker/vision/clova_ocr/experiments/evaluate_value_accuracy.py
    """

    rows = []
    details = []
    for output_dir in sorted(path for path in OUTPUT_ROOT_DIR.iterdir() if path.is_dir()):
        ground_truth_path = GROUND_TRUTH_DIR / f"{output_dir.name}.json"
        if not ground_truth_path.is_file():
            print(f"[SKIP] ground truth not found: {ground_truth_path}")
            continue

        ground_truth = _load_ground_truth(ground_truth_path)
        detail = {
            "pdf_stem": output_dir.name,
            "ground_truth_path": str(ground_truth_path),
            "note": (
                "Null ground truth fields are excluded. Numeric values use exact matching "
                "with 0.01 tolerance for floats; lists use set comparison."
            ),
            "sources": {},
        }

        for source, text_path in {
            "pdf_direct": output_dir / "pdf_direct_ocr.txt",
            "image_all_pages": output_dir / "image_all_pages_ocr.txt",
        }.items():
            prediction = parse_health_exam_result(_read_text(text_path))
            metrics = _compare_prediction(ground_truth, prediction)
            detail["sources"][source] = {
                "text_path": str(text_path),
                "prediction": prediction,
                **metrics,
            }
            rows.append({"pdf_stem": output_dir.name, "source": source, **metrics["summary"]})

        detail_path = output_dir / "value_accuracy_detail.json"
        _save_json(detail, detail_path)
        details.append(detail)
        _print_pdf_summary(output_dir.name, detail)

    summary = {
        "note": (
            "This value-level evaluation requires human-written ground truth JSON. "
            "Ground truth files may contain personal health information and must not be committed."
        ),
        "aggregate": _aggregate(rows),
        "items": rows,
    }
    _save_json(summary, SUMMARY_JSON_PATH)
    _save_csv(rows, SUMMARY_CSV_PATH)
    _print_batch_summary(summary["aggregate"])


def _load_ground_truth(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {field: raw.get(field) for field in VALUE_FIELDS}


def _compare_prediction(ground_truth: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    evaluated_fields = [field for field in VALUE_FIELDS if ground_truth.get(field) is not None]
    mismatches = []
    matched_count = 0
    predicted_count = 0
    missing_prediction_count = 0

    for field in evaluated_fields:
        expected = ground_truth.get(field)
        predicted = prediction.get(field)
        if predicted is None:
            missing_prediction_count += 1
            mismatches.append({"field": field, "expected": expected, "predicted": predicted, "reason": "missing"})
            continue

        predicted_count += 1
        if _values_match(expected, predicted):
            matched_count += 1
        else:
            mismatches.append({"field": field, "expected": expected, "predicted": predicted, "reason": "wrong_value"})

    field_count = len(evaluated_fields)
    wrong_value_count = len([item for item in mismatches if item["reason"] == "wrong_value"])
    summary = {
        "field_count": field_count,
        "predicted_count": predicted_count,
        "matched_count": matched_count,
        "missing_prediction_count": missing_prediction_count,
        "wrong_value_count": wrong_value_count,
        "field_coverage": predicted_count / field_count if field_count else 0.0,
        "value_accuracy": matched_count / field_count if field_count else 0.0,
        "precision_like_accuracy": matched_count / predicted_count if predicted_count else 0.0,
    }
    return {
        "summary": summary,
        "evaluated_fields": evaluated_fields,
        "mismatches": mismatches,
    }


def _values_match(expected: Any, predicted: Any) -> bool:
    if isinstance(expected, list):
        return set(map(str, expected)) == set(map(str, predicted if isinstance(predicted, list) else [predicted]))

    if isinstance(expected, int | float):
        if not isinstance(predicted, int | float):
            return False
        tolerance = 0.01 if isinstance(expected, float) or isinstance(predicted, float) else 0.0
        return abs(float(expected) - float(predicted)) <= tolerance

    return str(expected).strip() == str(predicted).strip()


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_rows": len(rows),
        "pdf_direct": _aggregate_source(rows, "pdf_direct"),
        "image_all_pages": _aggregate_source(rows, "image_all_pages"),
    }


def _aggregate_source(rows: list[dict[str, Any]], source: str) -> dict[str, float | int | None]:
    source_rows = [row for row in rows if row["source"] == source]
    return {
        "count": len(source_rows),
        "avg_field_coverage": _avg(source_rows, "field_coverage"),
        "avg_value_accuracy": _avg(source_rows, "value_accuracy"),
        "avg_precision_like_accuracy": _avg(source_rows, "precision_like_accuracy"),
        "total_matched_count": sum(row["matched_count"] for row in source_rows),
        "total_wrong_value_count": sum(row["wrong_value_count"] for row in source_rows),
        "total_missing_prediction_count": sum(row["missing_prediction_count"] for row in source_rows),
    }


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), int | float)]
    if not values:
        return None
    return sum(values) / len(values)


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _print_pdf_summary(pdf_stem: str, detail: dict[str, Any]) -> None:
    print(f"===== Value Accuracy: {pdf_stem} =====")
    for source, payload in detail["sources"].items():
        summary = payload["summary"]
        print(
            f"{source}: coverage={summary['field_coverage']:.4f}, "
            f"value_accuracy={summary['value_accuracy']:.4f}, "
            f"matched={summary['matched_count']}/{summary['field_count']}"
        )


def _print_batch_summary(aggregate: dict[str, Any]) -> None:
    print("\n===== VALUE ACCURACY SUMMARY =====")
    print(f"total_rows: {aggregate['total_rows']}")
    print(f"summary_json: {SUMMARY_JSON_PATH}")
    print(f"summary_csv: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
