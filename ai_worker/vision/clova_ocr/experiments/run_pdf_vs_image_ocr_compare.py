import sys
import time
from pathlib import Path
from typing import Any

import requests

CLOVA_OCR_ROOT = Path(__file__).resolve().parents[1]
if str(CLOVA_OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(CLOVA_OCR_ROOT))

from clova_client import ClovaOCRClient  # noqa: E402
from extractor import (  # noqa: E402
    calculate_ocr_metrics,
    extract_fields,
    extract_plain_text,
    save_json,
    save_text,
)
from pdf_converter import convert_pdf_all_pages_to_images  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CLOVA_OCR_DIR = PROJECT_ROOT / "ai_worker" / "vision" / "clova_ocr"

DEFAULT_PDF_PATH = CLOVA_OCR_DIR / "data" / "pdfs" / "1-20260519113151821.pdf"
IMAGE_ROOT_DIR = CLOVA_OCR_DIR / "data" / "images"
OUTPUT_ROOT_DIR = CLOVA_OCR_DIR / "outputs"


def main() -> None:
    """Compare direct PDF OCR with OCR on every page converted to JPG.

    Required environment variables:
        export CLOVA_OCR_API_URL="..."
        export CLOVA_OCR_SECRET_KEY="..."

    PDF conversion requires PyMuPDF:
        uv pip install pymupdf

    Run:
        uv run python ai_worker/vision/clova_ocr/experiments/run_pdf_vs_image_ocr_compare.py
    """

    client = ClovaOCRClient()
    pdf_stem = DEFAULT_PDF_PATH.stem
    image_output_dir = IMAGE_ROOT_DIR / pdf_stem
    output_dir = OUTPUT_ROOT_DIR / pdf_stem
    errors = []

    pdf_result = _run_ocr_experiment(
        client=client,
        input_path=DEFAULT_PDF_PATH,
        text_output_path=output_dir / "pdf_direct_ocr.txt",
        raw_output_path=output_dir / "pdf_direct_raw.json",
        metrics_output_path=output_dir / "pdf_direct_metrics.json",
    )
    if not pdf_result["ok"]:
        errors.append({"stage": "pdf_direct_ocr", **_error_payload(pdf_result)})

    try:
        image_paths = [
            Path(path) for path in convert_pdf_all_pages_to_images(str(DEFAULT_PDF_PATH), str(image_output_dir))
        ]
    except Exception as exc:
        message = f"PDF 전체 페이지 이미지 변환 실패: {exc}"
        print(f"[ERROR] {message}")
        errors.append({"stage": "pdf_to_images", "error": message})
        image_paths = []

    page_results = []
    for page_number, image_path in enumerate(image_paths, 1):
        page_label = f"page_{page_number:03d}"
        page_result = _run_ocr_experiment(
            client=client,
            input_path=image_path,
            text_output_path=output_dir / f"image_{page_label}_ocr.txt",
            raw_output_path=output_dir / f"image_{page_label}_raw.json",
            metrics_output_path=output_dir / f"image_{page_label}_metrics.json",
        )
        page_result["page_number"] = page_number
        page_result["page_image_path"] = str(image_path)
        page_results.append(page_result)

        if not page_result["ok"]:
            errors.append({"stage": f"image_{page_label}_ocr", **_error_payload(page_result)})

    image_all_pages = _build_all_pages_result(page_results)
    save_text(image_all_pages["text"], str(output_dir / "image_all_pages_ocr.txt"))
    save_json(image_all_pages["metrics"], str(output_dir / "image_all_pages_metrics.json"))

    comparison = _build_comparison(
        pdf_path=DEFAULT_PDF_PATH,
        page_count=len(image_paths),
        pdf_result=pdf_result,
        image_all_pages=image_all_pages,
        errors=errors,
    )
    save_json(comparison, str(output_dir / "compare.json"))
    _print_summary(pdf_result, image_all_pages, comparison)


def _run_ocr_experiment(
    client: ClovaOCRClient,
    input_path: Path,
    text_output_path: Path,
    raw_output_path: Path,
    metrics_output_path: Path,
) -> dict[str, Any]:
    try:
        started_at = time.perf_counter()
        ocr_result = client.request_ocr(str(input_path))
        elapsed_seconds = time.perf_counter() - started_at
    except requests.HTTPError as exc:
        response = exc.response
        status_code = response.status_code if response is not None else "unknown"
        message = response.text if response is not None else str(exc)
        print(f"[ERROR] CLOVA OCR API 실패: status={status_code}, response={message}")
        return {"ok": False, "error": str(exc), "status_code": status_code, "response": message, "metrics": None}
    except Exception as exc:
        print(f"[ERROR] OCR 실험 실패: {input_path} | {exc}")
        return {"ok": False, "error": str(exc), "metrics": None}

    fields = extract_fields(ocr_result)
    text = extract_plain_text(ocr_result)
    metrics = calculate_ocr_metrics(fields, elapsed_seconds, text)

    save_text(text, str(text_output_path))
    save_json(ocr_result, str(raw_output_path))
    save_json(
        {
            **metrics,
            "note": (
                "confidence is CLOVA OCR field recognition confidence, not ground-truth accuracy. "
                "Human-labeled text or labels are required to calculate real accuracy."
            ),
        },
        str(metrics_output_path),
    )

    return {
        "ok": True,
        "input_path": str(input_path),
        "text": text,
        "fields": fields,
        "metrics": metrics,
    }


def _build_all_pages_result(page_results: list[dict[str, Any]]) -> dict[str, Any]:
    successful_pages = [result for result in page_results if result.get("ok")]
    all_fields = [field for result in successful_pages for field in result.get("fields", [])]
    combined_text = "\n\n".join(result.get("text", "") for result in successful_pages if result.get("text"))
    total_elapsed_seconds = sum(
        (result.get("metrics") or {}).get("elapsed_seconds", 0.0) for result in successful_pages
    )
    metrics = calculate_ocr_metrics(all_fields, total_elapsed_seconds, combined_text)
    metrics["weighted_avg_confidence"] = metrics["avg_confidence"]
    metrics["total_elapsed_seconds"] = total_elapsed_seconds
    metrics["successful_page_count"] = len(successful_pages)
    metrics["failed_page_count"] = len(page_results) - len(successful_pages)
    metrics["page_metrics"] = [
        {
            "page_number": result.get("page_number"),
            "ok": result.get("ok"),
            "metrics": result.get("metrics"),
            "error": result.get("error"),
        }
        for result in page_results
    ]

    return {"ok": bool(successful_pages), "text": combined_text, "fields": all_fields, "metrics": metrics}


def _build_comparison(
    pdf_path: Path,
    page_count: int,
    pdf_result: dict[str, Any],
    image_all_pages: dict[str, Any],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    pdf_metrics = pdf_result.get("metrics") or {}
    image_metrics = image_all_pages.get("metrics") or {}

    return {
        "pdf_file": str(pdf_path),
        "page_count": page_count,
        "pdf_direct": {
            "ok": pdf_result.get("ok", False),
            "field_count": pdf_metrics.get("field_count"),
            "avg_confidence": pdf_metrics.get("avg_confidence"),
            "min_confidence": pdf_metrics.get("min_confidence"),
            "elapsed_seconds": pdf_metrics.get("elapsed_seconds"),
            "extracted_text_length": pdf_metrics.get("extracted_text_length"),
        },
        "image_all_pages": {
            "ok": image_all_pages.get("ok", False),
            "field_count": image_metrics.get("field_count"),
            "weighted_avg_confidence": image_metrics.get("weighted_avg_confidence"),
            "min_confidence": image_metrics.get("min_confidence"),
            "total_elapsed_seconds": image_metrics.get("total_elapsed_seconds"),
            "extracted_text_length": image_metrics.get("extracted_text_length"),
        },
        "diff": {
            "field_count": _diff(pdf_metrics, image_metrics, "field_count"),
            "confidence": _diff_with_keys(pdf_metrics, "avg_confidence", image_metrics, "weighted_avg_confidence"),
            "elapsed_seconds": _diff_with_keys(pdf_metrics, "elapsed_seconds", image_metrics, "total_elapsed_seconds"),
            "extracted_text_length": _diff(pdf_metrics, image_metrics, "extracted_text_length"),
        },
        "winner": {
            "field_count": _higher_winner(pdf_metrics, "field_count", image_metrics, "field_count"),
            "confidence": _higher_winner(pdf_metrics, "avg_confidence", image_metrics, "weighted_avg_confidence"),
            "speed": _lower_winner(pdf_metrics, "elapsed_seconds", image_metrics, "total_elapsed_seconds"),
            "text_length": _higher_winner(pdf_metrics, "extracted_text_length", image_metrics, "extracted_text_length"),
        },
        "errors": errors,
        "note": (
            "confidence is CLOVA OCR field recognition confidence, not ground-truth accuracy. "
            "True accuracy requires a ground-truth text or label file."
        ),
    }


def _diff(pdf_metrics: dict[str, Any], image_metrics: dict[str, Any], key: str) -> float | int | None:
    return _diff_with_keys(pdf_metrics, key, image_metrics, key)


def _diff_with_keys(
    left_metrics: dict[str, Any], left_key: str, right_metrics: dict[str, Any], right_key: str
) -> float | int | None:
    left_value = left_metrics.get(left_key)
    right_value = right_metrics.get(right_key)
    if left_value is None or right_value is None:
        return None
    return right_value - left_value


def _higher_winner(pdf_metrics: dict[str, Any], pdf_key: str, image_metrics: dict[str, Any], image_key: str) -> str:
    return _winner(pdf_metrics, pdf_key, image_metrics, image_key, higher_is_better=True)


def _lower_winner(pdf_metrics: dict[str, Any], pdf_key: str, image_metrics: dict[str, Any], image_key: str) -> str:
    return _winner(pdf_metrics, pdf_key, image_metrics, image_key, higher_is_better=False)


def _winner(
    pdf_metrics: dict[str, Any],
    pdf_key: str,
    image_metrics: dict[str, Any],
    image_key: str,
    *,
    higher_is_better: bool,
) -> str:
    pdf_value = pdf_metrics.get(pdf_key)
    image_value = image_metrics.get(image_key)
    if pdf_value is None and image_value is None:
        return "unavailable"
    if pdf_value is None:
        return "image_all_pages"
    if image_value is None:
        return "pdf_direct"
    if pdf_value == image_value:
        return "tie"

    pdf_wins = pdf_value > image_value if higher_is_better else pdf_value < image_value
    return "pdf_direct" if pdf_wins else "image_all_pages"


def _error_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "error": result.get("error"),
        "status_code": result.get("status_code"),
        "response": result.get("response"),
    }


def _print_summary(pdf_result: dict[str, Any], image_all_pages: dict[str, Any], comparison: dict[str, Any]) -> None:
    pdf_metrics = pdf_result.get("metrics") or {}
    image_metrics = image_all_pages.get("metrics") or {}

    print("===== PDF DIRECT OCR =====")
    _print_metric_line("field_count", pdf_metrics)
    _print_metric_line("avg_confidence", pdf_metrics)
    _print_metric_line("elapsed_seconds", pdf_metrics)
    _print_metric_line("text_length", pdf_metrics, key="extracted_text_length")

    print("\n===== IMAGE ALL PAGES OCR =====")
    _print_metric_line("field_count", image_metrics)
    _print_metric_line("weighted_avg_confidence", image_metrics)
    _print_metric_line("total_elapsed_seconds", image_metrics)
    _print_metric_line("text_length", image_metrics, key="extracted_text_length")

    print("\n===== COMPARISON =====")
    print(f"field_count winner: {comparison['winner']['field_count']}")
    print(f"confidence winner: {comparison['winner']['confidence']}")
    print(f"speed winner: {comparison['winner']['speed']}")
    print(f"text_length winner: {comparison['winner']['text_length']}")
    print(f"errors: {len(comparison['errors'])}")


def _print_metric_line(label: str, metrics: dict[str, Any], key: str | None = None) -> None:
    value = metrics.get(key or label, "N/A")
    print(f"{label}: {value}")


if __name__ == "__main__":
    main()
