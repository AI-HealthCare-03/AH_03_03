import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CLOVA_OCR_DIR = PROJECT_ROOT / "ai_worker" / "vision" / "clova_ocr"
OUTPUT_ROOT_DIR = CLOVA_OCR_DIR / "outputs"
GROUND_TRUTH_DIR = CLOVA_OCR_DIR / "ground_truth"

GROUND_TRUTH_FIELDS = [
    "exam_date",
    "height_cm",
    "weight_kg",
    "bmi",
    "waist_cm",
    "systolic_bp",
    "diastolic_bp",
    "hemoglobin",
    "fasting_glucose",
    "total_cholesterol",
    "triglyceride",
    "hdl",
    "ldl",
    "creatinine",
    "egfr",
    "ast",
    "alt",
    "gamma_gtp",
    "urine_protein",
    "chest_xray",
    "suspected_diseases",
    "lifestyle_smoking",
    "lifestyle_drinking",
    "lifestyle_physical_activity",
    "lifestyle_strength_training",
]


def main() -> None:
    """Create human-fillable ground truth JSON templates.

    Ground truth may contain personal health information and must stay local.
    The generated files are ignored by git via `ground_truth/*.json`.

    Run after OCR text files exist:
        uv run python ai_worker/vision/clova_ocr/experiments/export_ground_truth_template.py
    """

    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    output_dirs = sorted(path for path in OUTPUT_ROOT_DIR.iterdir() if path.is_dir())

    created_count = 0
    skipped_count = 0
    for output_dir in output_dirs:
        source_text_path = _select_source_text_path(output_dir)
        if source_text_path is None:
            print(f"[SKIP] OCR text not found: {output_dir.name}")
            skipped_count += 1
            continue

        template_path = GROUND_TRUTH_DIR / f"{output_dir.name}.json"
        if template_path.exists():
            print(f"[SKIP] already exists: {template_path}")
            skipped_count += 1
            continue

        template = {
            "_note": (
                "Fill this file manually from the original health exam report. "
                "Null fields are excluded from value-level accuracy evaluation. "
                "Do not commit this file because it can contain personal health information."
            ),
            "_source_ocr_text": str(source_text_path),
            **{field: None for field in GROUND_TRUTH_FIELDS},
        }
        template_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[CREATE] {template_path}")
        created_count += 1

    print("===== GROUND TRUTH TEMPLATE EXPORT =====")
    print(f"created_count: {created_count}")
    print(f"skipped_count: {skipped_count}")
    print(f"ground_truth_dir: {GROUND_TRUTH_DIR}")


def _select_source_text_path(output_dir: Path) -> Path | None:
    image_text = output_dir / "image_all_pages_ocr.txt"
    if image_text.is_file():
        return image_text

    pdf_text = output_dir / "pdf_direct_ocr.txt"
    if pdf_text.is_file():
        return pdf_text

    return None


if __name__ == "__main__":
    main()
