import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CLOVA_OCR_DIR = PROJECT_ROOT / "ai_worker" / "vision" / "clova_ocr"
OUTPUT_ROOT_DIR = CLOVA_OCR_DIR / "outputs"


def main() -> None:
    """Print image/page counts from CLOVA OCR raw JSON outputs.

    Run:
        uv run python ai_worker/vision/clova_ocr/experiments/debug_raw_json_pages.py
    """

    raw_paths = sorted(OUTPUT_ROOT_DIR.glob("*/pdf_direct_raw.json"))
    if not raw_paths:
        print(f"No pdf_direct_raw.json files found under: {OUTPUT_ROOT_DIR}")
        return

    for raw_path in raw_paths:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        images = data.get("images") or []
        field_counts = [len(image.get("fields") or []) for image in images]
        print(f"===== {raw_path.parent.name} =====")
        print(f"raw_json: {raw_path}")
        print(f"images_count: {len(images)}")
        print(f"field_counts_by_image: {field_counts}")


if __name__ == "__main__":
    main()
