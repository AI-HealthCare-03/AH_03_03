import csv
import sys
from pathlib import Path
from typing import Any

CLOVA_OCR_ROOT = Path(__file__).resolve().parents[1]
if str(CLOVA_OCR_ROOT) not in sys.path:
    sys.path.insert(0, str(CLOVA_OCR_ROOT))

from clova_client import ClovaOCRClient  # noqa: E402
from extractor import save_json  # noqa: E402
from pdf_converter import convert_pdf_all_pages_to_images  # noqa: E402
from run_pdf_vs_image_ocr_compare import (  # noqa: E402
    _build_all_pages_result,
    _build_comparison,
    _error_payload,
    _run_ocr_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CLOVA_OCR_DIR = PROJECT_ROOT / "ai_worker" / "vision" / "clova_ocr"
PDF_DIR = CLOVA_OCR_DIR / "data" / "pdfs"
IMAGE_ROOT_DIR = CLOVA_OCR_DIR / "data" / "images"
OUTPUT_ROOT_DIR = CLOVA_OCR_DIR / "outputs"
SUMMARY_JSON_PATH = OUTPUT_ROOT_DIR / "batch_compare_summary.json"
SUMMARY_CSV_PATH = OUTPUT_ROOT_DIR / "batch_compare_summary.csv"

SUMMARY_COLUMNS = [
    "pdf_stem",
    "pdf_file",
    "page_count",
    "pdf_direct_ok",
    "image_all_pages_ok",
    "pdf_direct_field_count",
    "image_all_pages_field_count",
    "field_count_diff",
    "pdf_direct_avg_confidence",
    "image_all_pages_weighted_avg_confidence",
    "confidence_diff",
    "pdf_direct_elapsed_seconds",
    "image_all_pages_total_elapsed_seconds",
    "elapsed_seconds_diff",
    "pdf_direct_text_length",
    "image_all_pages_text_length",
    "text_length_diff",
    "field_count_winner",
    "confidence_winner",
    "speed_winner",
    "text_length_winner",
    "error_count",
]


def main() -> None:
    """Run PDF-vs-image CLOVA OCR comparison for every PDF in data/pdfs.

    Required environment variables may be provided via project-root `.env`:
        CLOVA_OCR_API_URL="..."
        CLOVA_OCR_SECRET_KEY="..."

    PDF conversion requires PyMuPDF:
        uv pip install pymupdf

    Run:
        uv run python ai_worker/vision/clova_ocr/experiments/run_batch_pdf_vs_image_ocr_compare.py
    """

    client = ClovaOCRClient()
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    rows = []

    for pdf_path in pdf_paths:
        print(f"===== Processing {pdf_path.name} =====")
        try:
            comparison = _process_pdf(client, pdf_path)
            row = _summary_row(pdf_path, comparison)
        except Exception as exc:
            print(f"[ERROR] PDF 전체 처리 실패: {pdf_path} | {exc}")
            comparison = _failed_comparison(pdf_path, exc)
            _save_pdf_compare(pdf_path, comparison)
            row = _summary_row(pdf_path, comparison)

        rows.append(row)
        _print_pdf_summary(row)

    aggregate = _aggregate_rows(rows)
    save_json({"aggregate": aggregate, "items": rows}, str(SUMMARY_JSON_PATH))
    _save_summary_csv(rows, SUMMARY_CSV_PATH)
    _print_batch_summary(aggregate)


def _process_pdf(client: ClovaOCRClient, pdf_path: Path) -> dict[str, Any]:
    pdf_stem = pdf_path.stem
    image_output_dir = IMAGE_ROOT_DIR / pdf_stem
    output_dir = OUTPUT_ROOT_DIR / pdf_stem
    errors = []

    pdf_result = _run_ocr_experiment(
        client=client,
        input_path=pdf_path,
        text_output_path=output_dir / "pdf_direct_ocr.txt",
        raw_output_path=output_dir / "pdf_direct_raw.json",
        metrics_output_path=output_dir / "pdf_direct_metrics.json",
    )
    if not pdf_result["ok"]:
        errors.append({"stage": "pdf_direct_ocr", **_error_payload(pdf_result)})

    try:
        image_paths = [Path(path) for path in convert_pdf_all_pages_to_images(str(pdf_path), str(image_output_dir))]
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
    (output_dir / "image_all_pages_ocr.txt").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "image_all_pages_ocr.txt").write_text(image_all_pages["text"], encoding="utf-8")
    save_json(image_all_pages["metrics"], str(output_dir / "image_all_pages_metrics.json"))

    comparison = _build_comparison(
        pdf_path=pdf_path,
        page_count=len(image_paths),
        pdf_result=pdf_result,
        image_all_pages=image_all_pages,
        errors=errors,
    )
    _save_pdf_compare(pdf_path, comparison)
    return comparison


def _save_pdf_compare(pdf_path: Path, comparison: dict[str, Any]) -> None:
    save_json(comparison, str(OUTPUT_ROOT_DIR / pdf_path.stem / "compare.json"))


def _summary_row(pdf_path: Path, comparison: dict[str, Any]) -> dict[str, Any]:
    pdf_direct = comparison.get("pdf_direct") or {}
    image_all_pages = comparison.get("image_all_pages") or {}
    diff = comparison.get("diff") or {}
    winner = comparison.get("winner") or {}
    errors = comparison.get("errors") or []

    return {
        "pdf_stem": pdf_path.stem,
        "pdf_file": str(pdf_path),
        "page_count": comparison.get("page_count"),
        "pdf_direct_ok": pdf_direct.get("ok", False),
        "image_all_pages_ok": image_all_pages.get("ok", False),
        "pdf_direct_field_count": pdf_direct.get("field_count"),
        "image_all_pages_field_count": image_all_pages.get("field_count"),
        "field_count_diff": diff.get("field_count"),
        "pdf_direct_avg_confidence": pdf_direct.get("avg_confidence"),
        "image_all_pages_weighted_avg_confidence": image_all_pages.get("weighted_avg_confidence"),
        "confidence_diff": diff.get("confidence"),
        "pdf_direct_elapsed_seconds": pdf_direct.get("elapsed_seconds"),
        "image_all_pages_total_elapsed_seconds": image_all_pages.get("total_elapsed_seconds"),
        "elapsed_seconds_diff": diff.get("elapsed_seconds"),
        "pdf_direct_text_length": pdf_direct.get("extracted_text_length"),
        "image_all_pages_text_length": image_all_pages.get("extracted_text_length"),
        "text_length_diff": diff.get("extracted_text_length"),
        "field_count_winner": winner.get("field_count"),
        "confidence_winner": winner.get("confidence"),
        "speed_winner": winner.get("speed"),
        "text_length_winner": winner.get("text_length"),
        "error_count": len(errors),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_pdf_count": len(rows),
        "success_pdf_direct_count": sum(bool(row["pdf_direct_ok"]) for row in rows),
        "success_image_all_pages_count": sum(bool(row["image_all_pages_ok"]) for row in rows),
        "avg_pdf_direct_field_count": _avg(rows, "pdf_direct_field_count"),
        "avg_image_all_pages_field_count": _avg(rows, "image_all_pages_field_count"),
        "avg_pdf_direct_confidence": _avg(rows, "pdf_direct_avg_confidence"),
        "avg_image_all_pages_confidence": _avg(rows, "image_all_pages_weighted_avg_confidence"),
        "avg_pdf_direct_elapsed_seconds": _avg(rows, "pdf_direct_elapsed_seconds"),
        "avg_image_all_pages_elapsed_seconds": _avg(rows, "image_all_pages_total_elapsed_seconds"),
        "avg_pdf_direct_text_length": _avg(rows, "pdf_direct_text_length"),
        "avg_image_all_pages_text_length": _avg(rows, "image_all_pages_text_length"),
        "image_wins_field_count_count": _count_winner(rows, "field_count_winner", "image_all_pages"),
        "pdf_wins_confidence_count": _count_winner(rows, "confidence_winner", "pdf_direct"),
        "pdf_wins_speed_count": _count_winner(rows, "speed_winner", "pdf_direct"),
        "image_wins_text_length_count": _count_winner(rows, "text_length_winner", "image_all_pages"),
    }


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), int | float)]
    if not values:
        return None
    return sum(values) / len(values)


def _count_winner(rows: list[dict[str, Any]], key: str, winner: str) -> int:
    return sum(row.get(key) == winner for row in rows)


def _save_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _failed_comparison(pdf_path: Path, exc: Exception) -> dict[str, Any]:
    return {
        "pdf_file": str(pdf_path),
        "page_count": 0,
        "pdf_direct": {"ok": False},
        "image_all_pages": {"ok": False},
        "diff": {},
        "winner": {
            "field_count": "unavailable",
            "confidence": "unavailable",
            "speed": "unavailable",
            "text_length": "unavailable",
        },
        "errors": [{"stage": "pdf_processing", "error": str(exc)}],
    }


def _print_pdf_summary(row: dict[str, Any]) -> None:
    print(f"PDF direct field_count: {row['pdf_direct_field_count']}")
    print(f"Image all pages field_count: {row['image_all_pages_field_count']}")
    print(f"field_count winner: {row['field_count_winner']}")
    print(f"speed winner: {row['speed_winner']}")
    print(f"errors: {row['error_count']}")


def _print_batch_summary(aggregate: dict[str, Any]) -> None:
    print("\n===== BATCH SUMMARY =====")
    print(f"total_pdf_count: {aggregate['total_pdf_count']}")
    print(f"image_wins_field_count_count: {aggregate['image_wins_field_count_count']}")
    print(f"pdf_wins_speed_count: {aggregate['pdf_wins_speed_count']}")
    print(f"summary_json: {SUMMARY_JSON_PATH}")
    print(f"summary_csv: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
