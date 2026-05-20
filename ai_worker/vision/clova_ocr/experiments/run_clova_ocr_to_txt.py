import sys
import time
from pathlib import Path

CLOVA_OCR_ROOT = Path(__file__).resolve().parents[1]
if str(CLOVA_OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(CLOVA_OCR_ROOT))

from clova_client import ClovaOCRClient  # noqa: E402
from extractor import (  # noqa: E402
    calculate_ocr_metrics,
    extract_fields,
    extract_plain_text,
    get_low_confidence_fields,
    save_json,
    save_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CLOVA_OCR_DIR = PROJECT_ROOT / "ai_worker" / "vision" / "clova_ocr"
DEFAULT_IMAGE_PATH = CLOVA_OCR_DIR / "data" / "images" / "1.jpg"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "ai_worker" / "vision" / "clova_ocr" / "outputs" / "health_exam_ocr.txt"
DEFAULT_RAW_JSON_PATH = PROJECT_ROOT / "ai_worker" / "vision" / "clova_ocr" / "outputs" / "health_exam_ocr_raw.json"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "ai_worker" / "vision" / "clova_ocr" / "outputs" / "health_exam_ocr_metrics.json"


def main() -> None:
    """Run CLOVA OCR PoC and save extracted text.

    Required environment variables:
        export CLOVA_OCR_API_URL="..."
        export CLOVA_OCR_SECRET_KEY="..."

    Run:
        uv run python ai_worker/vision/clova_ocr/experiments/run_clova_ocr_to_txt.py
    """

    client = ClovaOCRClient()
    started_at = time.perf_counter()
    ocr_result = client.request_ocr(str(DEFAULT_IMAGE_PATH))
    elapsed_seconds = time.perf_counter() - started_at

    fields = extract_fields(ocr_result)
    text = extract_plain_text(ocr_result)
    metrics = calculate_ocr_metrics(fields, elapsed_seconds, text)
    low_confidence_fields = get_low_confidence_fields(fields, threshold=0.8)

    save_text(text, str(DEFAULT_OUTPUT_PATH))
    save_json(ocr_result, str(DEFAULT_RAW_JSON_PATH))
    save_json(
        {
            **metrics,
            "low_confidence_fields_lt_0_8": low_confidence_fields,
            "note": (
                "confidence is CLOVA OCR field recognition confidence, not ground-truth accuracy. "
                "Human-labeled text or labels are required to calculate real accuracy."
            ),
        },
        str(DEFAULT_METRICS_PATH),
    )

    print("===== CLOVA OCR RESULT =====")
    print(f"이미지 경로: {_relative_to_project(DEFAULT_IMAGE_PATH)}")
    print(f"텍스트 저장: {_relative_to_project(DEFAULT_OUTPUT_PATH)}")
    print(f"Raw JSON 저장: {_relative_to_project(DEFAULT_RAW_JSON_PATH)}")
    print(f"Metrics 저장: {_relative_to_project(DEFAULT_METRICS_PATH)}")

    print("\n===== SPEED =====")
    print(f"OCR API 호출 시간: {metrics['elapsed_seconds']:.3f} sec")
    print(f"추출 field 수: {metrics['field_count']}")
    print(f"처리량: {metrics['fields_per_second']:.2f} fields/sec")

    print("\n===== CONFIDENCE =====")
    print(f"평균 confidence: {metrics['avg_confidence']:.4f}")
    print(f"최소 confidence: {metrics['min_confidence']:.4f}")
    print(f"최대 confidence: {metrics['max_confidence']:.4f}")
    print(f"confidence < 0.8 개수: {metrics['low_confidence_count_under_0_8']}")
    print(f"confidence < 0.7 개수: {metrics['low_confidence_count_under_0_7']}")

    print("\n===== LOW CONFIDENCE FIELDS (<0.8) =====")
    if low_confidence_fields:
        for field in low_confidence_fields:
            print(f'- text="{field["text"]}", confidence={field["confidence"]:.4f}')
    else:
        print("- 없음")

    print("\n===== EXTRACTED TEXT =====")
    print(text)


def _relative_to_project(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
