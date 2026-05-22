import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CLOVA_OCR_DIR = PROJECT_ROOT / "ai_worker" / "vision" / "clova_ocr"
OUTPUT_ROOT_DIR = CLOVA_OCR_DIR / "outputs"
SUMMARY_JSON_PATH = OUTPUT_ROOT_DIR / "required_field_eval_summary.json"
SUMMARY_CSV_PATH = OUTPUT_ROOT_DIR / "required_field_eval_summary.csv"

# This evaluates whether required field labels are present in OCR text.
# It does not validate extracted values against ground truth. Real accuracy
# requires human-labeled values/text for comparison.
REQUIRED_FIELDS = {
    "성명": ["성명", "이름"],
    "검진일": ["검진일", "검진일자", "검사일", "검사일자"],
    "키": ["키", "신장"],
    "몸무게": ["몸무게", "체중"],
    "체질량지수": ["체질량지수", "BMI"],
    "허리둘레": ["허리둘레", "허리 둘레"],
    "혈압": ["혈압", "수축기", "이완기"],
    "혈색소": ["혈색소", "헤모글로빈"],
    "공복혈당": ["공복혈당", "공복 혈당", "식전혈당"],
    "총콜레스테롤": ["총콜레스테롤", "총 콜레스테롤", "Total Cholesterol"],
    "중성지방": ["중성지방", "트리글리세라이드", "Triglyceride"],
    "HDL": ["HDL", "HDL콜레스테롤", "HDL 콜레스테롤"],
    "LDL": ["LDL", "LDL콜레스테롤", "LDL 콜레스테롤"],
    "혈청크레아티닌": ["혈청크레아티닌", "크레아티닌", "Creatinine"],
    "eGFR": ["eGFR", "사구체여과율", "추정사구체여과율"],
    "AST": ["AST", "SGOT"],
    "ALT": ["ALT", "SGPT"],
    "감마지티피": ["감마지티피", "감마 지티피", "감마GTP", "GGT", "γ-GTP"],
    "요단백": ["요단백", "요 단백", "단백뇨"],
    "흉부촬영": ["흉부촬영", "흉부 촬영", "흉부X선", "흉부 X선"],
    "과거병력": ["과거병력", "과거 병력", "병력"],
    "약물치료": ["약물치료", "약물 치료", "복약", "투약"],
    "흡연": ["흡연", "담배"],
    "음주": ["음주", "술"],
    "신체활동": ["신체활동", "신체 활동", "운동"],
    "근력운동": ["근력운동", "근력 운동"],
    "종합판정": ["종합판정", "종합 판정", "판정"],
    "의심질환": ["의심질환", "의심 질환"],
}

CSV_COLUMNS = [
    "pdf_stem",
    "pdf_direct_required_field_rate",
    "image_required_field_rate",
    "pdf_direct_found_count",
    "image_found_count",
    "required_field_count",
    "pdf_direct_missing_fields",
    "image_missing_fields",
]


def main() -> None:
    rows = []
    for output_dir in sorted(path for path in OUTPUT_ROOT_DIR.iterdir() if path.is_dir()):
        row = _evaluate_pdf_output(output_dir)
        rows.append(row)
        _print_pdf_summary(row)

    summary = {
        "note": (
            "This metric checks required field label presence in OCR text. "
            "It is not value-level accuracy; ground-truth labels are required for true accuracy."
        ),
        "required_fields": list(REQUIRED_FIELDS.keys()),
        "aggregate": _aggregate(rows),
        "items": rows,
    }

    _save_json(summary, SUMMARY_JSON_PATH)
    _save_csv(rows, SUMMARY_CSV_PATH)
    _print_batch_summary(summary["aggregate"])


def _evaluate_pdf_output(output_dir: Path) -> dict[str, Any]:
    pdf_text = _read_text(output_dir / "pdf_direct_ocr.txt")
    image_text = _read_text(output_dir / "image_all_pages_ocr.txt")

    pdf_eval = _evaluate_required_fields(pdf_text)
    image_eval = _evaluate_required_fields(image_text)

    return {
        "pdf_stem": output_dir.name,
        "pdf_direct_required_field_rate": pdf_eval["rate"],
        "image_required_field_rate": image_eval["rate"],
        "pdf_direct_found_count": pdf_eval["found_count"],
        "image_found_count": image_eval["found_count"],
        "required_field_count": len(REQUIRED_FIELDS),
        "pdf_direct_found_fields": pdf_eval["found_fields"],
        "image_found_fields": image_eval["found_fields"],
        "pdf_direct_missing_fields": pdf_eval["missing_fields"],
        "image_missing_fields": image_eval["missing_fields"],
    }


def _evaluate_required_fields(text: str) -> dict[str, Any]:
    normalized_text = _normalize(text)
    found_fields = []
    missing_fields = []

    for field_name, aliases in REQUIRED_FIELDS.items():
        if any(_normalize(alias) in normalized_text for alias in aliases):
            found_fields.append(field_name)
        else:
            missing_fields.append(field_name)

    found_count = len(found_fields)
    return {
        "found_count": found_count,
        "rate": found_count / len(REQUIRED_FIELDS) if REQUIRED_FIELDS else 0.0,
        "found_fields": found_fields,
        "missing_fields": missing_fields,
    }


def _normalize(text: str) -> str:
    return "".join(str(text).lower().split())


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_pdf_count": len(rows),
        "avg_pdf_direct_required_field_rate": _avg(rows, "pdf_direct_required_field_rate"),
        "avg_image_required_field_rate": _avg(rows, "image_required_field_rate"),
        "image_better_count": sum(
            row["image_required_field_rate"] > row["pdf_direct_required_field_rate"] for row in rows
        ),
        "pdf_direct_better_count": sum(
            row["pdf_direct_required_field_rate"] > row["image_required_field_rate"] for row in rows
        ),
        "tie_count": sum(row["pdf_direct_required_field_rate"] == row["image_required_field_rate"] for row in rows),
    }


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), int | float)]
    if not values:
        return None
    return sum(values) / len(values)


def _save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in CSV_COLUMNS if key in row},
                    "pdf_direct_missing_fields": "|".join(row["pdf_direct_missing_fields"]),
                    "image_missing_fields": "|".join(row["image_missing_fields"]),
                }
            )


def _print_pdf_summary(row: dict[str, Any]) -> None:
    print(f"===== Required Field Eval: {row['pdf_stem']} =====")
    print(f"pdf_direct_required_field_rate: {row['pdf_direct_required_field_rate']:.4f}")
    print(f"image_required_field_rate: {row['image_required_field_rate']:.4f}")
    print(f"pdf_direct_missing_fields: {row['pdf_direct_missing_fields']}")
    print(f"image_missing_fields: {row['image_missing_fields']}")


def _print_batch_summary(aggregate: dict[str, Any]) -> None:
    print("\n===== REQUIRED FIELD BATCH SUMMARY =====")
    print(f"total_pdf_count: {aggregate['total_pdf_count']}")
    print(f"avg_pdf_direct_required_field_rate: {aggregate['avg_pdf_direct_required_field_rate']}")
    print(f"avg_image_required_field_rate: {aggregate['avg_image_required_field_rate']}")
    print(f"summary_json: {SUMMARY_JSON_PATH}")
    print(f"summary_csv: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
