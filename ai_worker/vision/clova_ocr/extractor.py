import json
from pathlib import Path
from typing import Any


def extract_plain_text(ocr_result: dict) -> str:
    lines = []
    current_page_index = None
    for field in extract_fields(ocr_result):
        page_index = field["page_index"]
        if current_page_index is not None and page_index != current_page_index:
            lines.append("")
        lines.append(field["text"])
        current_page_index = page_index

    return "\n".join(lines)


def extract_fields(ocr_result: dict) -> list[dict[str, Any]]:
    images = ocr_result.get("images") or []
    if not images:
        return []

    extracted = []
    for page_index, image in enumerate(images):
        fields = image.get("fields") or []
        for field in fields:
            text = str(field.get("inferText") or "").strip()
            if not text:
                continue

            extracted.append(
                {
                    "text": text,
                    "confidence": _parse_confidence(field.get("inferConfidence")),
                    "bounding_box": field.get("boundingPoly"),
                    "page_index": page_index,
                    "page_number": page_index + 1,
                }
            )

    return extracted


def calculate_ocr_metrics(fields: list[dict], elapsed_seconds: float, text: str) -> dict[str, Any]:
    """Calculate OCR speed and confidence metrics.

    CLOVA confidence is the OCR service's recognition confidence for each field.
    It is not ground-truth accuracy; true accuracy requires human-labeled text or
    a separate label file for comparison.
    """

    field_count = len(fields)
    confidences = [_parse_confidence(field.get("confidence")) for field in fields]
    elapsed = max(elapsed_seconds, 0.0)

    return {
        "field_count": field_count,
        "avg_confidence": sum(confidences) / field_count if field_count else 0.0,
        "min_confidence": min(confidences) if confidences else 0.0,
        "max_confidence": max(confidences) if confidences else 0.0,
        "low_confidence_count_under_0_8": sum(confidence < 0.8 for confidence in confidences),
        "low_confidence_count_under_0_7": sum(confidence < 0.7 for confidence in confidences),
        "elapsed_seconds": elapsed,
        "fields_per_second": field_count / elapsed if field_count and elapsed > 0 else 0.0,
        "extracted_text_length": len(text),
        "extracted_line_count": len([line for line in text.splitlines() if line.strip()]),
        "page_count": len({field.get("page_index") for field in fields if field.get("page_index") is not None}),
    }


def get_low_confidence_fields(fields: list[dict], threshold: float = 0.8) -> list[dict]:
    return [field for field in fields if _parse_confidence(field.get("confidence")) < threshold]


def save_text(text: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(data: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
