from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "image_filename",
    "expected_foods",
    "label_source",
    "annotation_count",
    "cat_1",
    "cat_2",
    "cat_3",
]


@dataclass
class LabelAggregate:
    image_filename: str
    expected_foods: list[str] = field(default_factory=list)
    label_sources: list[str] = field(default_factory=list)
    annotation_count: int = 0
    cat_1: str = ""
    cat_2: str = ""
    cat_3: str = ""

    def add(
        self,
        *,
        foods: list[str],
        label_source: str,
        annotation_count: int,
        cat_1: str,
        cat_2: str,
        cat_3: str,
    ) -> None:
        for food in foods:
            if food and food not in self.expected_foods:
                self.expected_foods.append(food)
        if label_source and label_source not in self.label_sources:
            self.label_sources.append(label_source)
        self.annotation_count += annotation_count
        self.cat_1 = self.cat_1 or cat_1
        self.cat_2 = self.cat_2 or cat_2
        self.cat_3 = self.cat_3 or cat_3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPT Vision eval labels from AI-Hub food JSON zips.")
    parser.add_argument("--zip-path", required=True, help="Outer AI-Hub archive zip path.")
    parser.add_argument("--output", required=True, help="Output labels CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit output image rows for smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregates, summary = build_labels_from_aihub_zip(zip_path, limit=args.limit)
    write_labels_csv(output_path, aggregates)
    write_summary(output_dir / "aihub_label_summary.json", summary)
    print(f"Wrote {output_path}")
    print(f"Wrote {output_dir / 'aihub_label_summary.json'}")


def build_labels_from_aihub_zip(
    zip_path: Path,
    *,
    limit: int | None = None,
) -> tuple[list[LabelAggregate], dict[str, Any]]:
    aggregates: dict[str, LabelAggregate] = {}
    summary: dict[str, Any] = {
        "zip_path": str(zip_path),
        "nested_zip_count": 0,
        "json_count": 0,
        "skipped_json_count": 0,
        "missing_image_filename_count": 0,
        "output_row_count": 0,
    }

    for nested_zip_name, nested_zip_bytes in iter_nested_label_zip_bytes(zip_path):
        summary["nested_zip_count"] += 1
        with zipfile.ZipFile(io.BytesIO(nested_zip_bytes)) as nested_zip:
            for json_info in nested_zip.infolist():
                if json_info.is_dir() or not json_info.filename.lower().endswith(".json"):
                    continue
                summary["json_count"] += 1
                try:
                    payload = json.loads(nested_zip.read(json_info).decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError, OSError, zipfile.BadZipFile):
                    summary["skipped_json_count"] += 1
                    continue

                extracted = extract_label_payload(
                    payload,
                    nested_zip_name=nested_zip_name,
                    json_path=json_info.filename,
                )
                if extracted is None:
                    summary["missing_image_filename_count"] += 1
                    continue

                aggregate = aggregates.setdefault(
                    extracted.image_filename,
                    LabelAggregate(image_filename=extracted.image_filename),
                )
                aggregate.add(
                    foods=extracted.expected_foods,
                    label_source=extracted.label_sources[0],
                    annotation_count=extracted.annotation_count,
                    cat_1=extracted.cat_1,
                    cat_2=extracted.cat_2,
                    cat_3=extracted.cat_3,
                )
                if limit is not None and len(aggregates) >= limit:
                    summary["output_row_count"] = len(aggregates)
                    return list(aggregates.values()), summary

    summary["output_row_count"] = len(aggregates)
    return list(aggregates.values()), summary


def iter_nested_label_zip_bytes(zip_path: Path) -> list[tuple[str, bytes]]:
    nested_zips: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(zip_path) as outer_zip:
        for info in outer_zip.infolist():
            if info.is_dir():
                continue
            filename = Path(info.filename).name
            if filename.endswith("_Val_json.zip"):
                nested_zips.append((info.filename, outer_zip.read(info)))

    if nested_zips:
        return nested_zips

    if zip_path.name.endswith("_Val_json.zip"):
        return [(zip_path.name, zip_path.read_bytes())]
    return []


def extract_label_payload(
    payload: Any,
    *,
    nested_zip_name: str,
    json_path: str,
) -> LabelAggregate | None:
    image_filename = _first_string_value(payload, "Code Name")
    if not image_filename:
        return None

    path_categories = _path_categories(json_path)
    cat_1 = _first_string_value(payload, "cat_1") or path_categories["cat_1"]
    cat_2 = _first_string_value(payload, "cat_2") or path_categories["cat_2"]
    cat_3 = _first_string_value(payload, "cat_3") or path_categories["cat_3"]
    names = _string_values(payload, "Name")
    foods = _dedupe([name for name in names if name] or [cat_3, cat_2, Path(json_path).parent.name])
    return LabelAggregate(
        image_filename=image_filename,
        expected_foods=foods,
        label_sources=[f"{nested_zip_name}:{json_path}"],
        annotation_count=max(1, len(foods)),
        cat_1=cat_1,
        cat_2=cat_2,
        cat_3=cat_3,
    )


def write_labels_csv(output_path: Path, aggregates: list[LabelAggregate]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for aggregate in aggregates:
            writer.writerow(
                {
                    "image_filename": aggregate.image_filename,
                    "expected_foods": "|".join(aggregate.expected_foods),
                    "label_source": "|".join(aggregate.label_sources),
                    "annotation_count": aggregate.annotation_count,
                    "cat_1": aggregate.cat_1,
                    "cat_2": aggregate.cat_2,
                    "cat_3": aggregate.cat_3,
                }
            )


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _first_string_value(payload: Any, key: str) -> str:
    values = _string_values(payload, key)
    return values[0] if values else ""


def _string_values(payload: Any, key: str) -> list[str]:
    normalized_key = _normalize_key(key)
    values: list[str] = []
    for found_key, value in _walk_key_values(payload):
        if _normalize_key(found_key) != normalized_key:
            continue
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return _dedupe(values)


def _walk_key_values(payload: Any):
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key), value
            yield from _walk_key_values(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _walk_key_values(item)


def _path_categories(json_path: str) -> dict[str, str]:
    parts = [part for part in Path(json_path).parts[:-1] if part not in {".", ""}]
    tail = parts[-3:]
    padded = [""] * (3 - len(tail)) + tail
    return {
        "cat_1": padded[0],
        "cat_2": padded[1],
        "cat_3": padded[2],
    }


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


if __name__ == "__main__":
    main()
