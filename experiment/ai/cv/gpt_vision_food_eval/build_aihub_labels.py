from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CSV_COLUMNS = [
    "image_path",
    "image_filename",
    "expected_foods",
    "image_exists",
    "label_source",
    "annotation_count",
    "cat_1",
    "cat_2",
    "cat_3",
]

MISSING_IMAGE_COLUMNS = [
    "image_filename",
    "expected_foods",
    "json_path",
    "searched_image_root",
    "reason",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class LabelAggregate:
    image_filename: str
    expected_foods: list[str] = field(default_factory=list)
    label_sources: list[str] = field(default_factory=list)
    annotation_count: int = 0
    cat_1: str = ""
    cat_2: str = ""
    cat_3: str = ""
    image_path: str = ""
    image_exists: bool = False

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


@dataclass(frozen=True)
class JsonRecord:
    payload: Any
    source_name: str
    json_path: str


@dataclass(frozen=True)
class ImageIndex:
    image_root: Path | None
    by_name: dict[str, Path]
    by_normalized_name: dict[str, Path]
    by_normalized_stem: dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPT Vision eval labels from AI-Hub food JSON labels.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--zip-path", help="Outer AI-Hub archive zip path or direct *_Val_json.zip path.")
    input_group.add_argument("--json-root", help="Root directory containing extracted AI-Hub JSON files.")
    parser.add_argument("--image-root", help="Root directory containing extracted food images.")
    parser.add_argument("--output", required=True, help="Output labels CSV path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit output image rows for smoke runs.")
    parser.add_argument(
        "--per-class-limit", type=int, default=None, help="Maximum output rows per expected food class."
    )
    parser.add_argument(
        "--include-missing", action="store_true", help="Include image_exists=false rows in balanced samples."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced per-class sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_report_path = output_dir / "aihub_missing_images.csv"
    allowed_foods_path = output_dir / "allowed_foods.json"

    image_root = Path(args.image_root) if args.image_root else None
    build_limit = None if args.per_class_limit is not None else args.limit
    if args.json_root:
        aggregates, summary = build_labels_from_json_root(
            Path(args.json_root),
            image_root=image_root,
            output_path=output_path,
            limit=build_limit,
        )
    else:
        aggregates, summary = build_labels_from_aihub_zip(
            Path(args.zip_path),
            image_root=image_root,
            output_path=output_path,
            limit=build_limit,
        )

    sampled_aggregates = select_output_aggregates(
        aggregates,
        per_class_limit=args.per_class_limit,
        include_missing=args.include_missing,
        seed=args.seed,
        limit=args.limit,
    )
    summary.update(
        build_sampling_summary(
            sampled_aggregates,
            per_class_limit=args.per_class_limit,
            include_missing=args.include_missing,
            seed=args.seed,
        )
    )
    write_labels_csv(output_path, sampled_aggregates)
    write_allowed_foods_json(allowed_foods_path, aggregates)
    write_missing_images_csv(missing_report_path, aggregates, image_root=image_root)
    summary["allowed_foods_path"] = str(allowed_foods_path)
    summary["missing_report_path"] = str(missing_report_path)
    write_summary(output_dir / "aihub_label_summary.json", summary)
    print(f"Wrote {output_path}")
    print(f"Wrote {allowed_foods_path}")
    print(f"Wrote {output_dir / 'aihub_label_summary.json'}")
    print(f"Wrote {missing_report_path}")


def build_labels_from_json_root(
    json_root: Path,
    *,
    image_root: Path | None,
    output_path: Path,
    limit: int | None = None,
) -> tuple[list[LabelAggregate], dict[str, Any]]:
    summary = _base_summary(output_path)
    summary["json_root"] = str(json_root)
    aggregates = _build_labels_from_records(
        iter_json_root_records(json_root, summary=summary),
        image_root=image_root,
        output_path=output_path,
        summary=summary,
        limit=limit,
    )
    return aggregates, summary


def build_labels_from_aihub_zip(
    zip_path: Path,
    *,
    image_root: Path | None,
    output_path: Path,
    limit: int | None = None,
) -> tuple[list[LabelAggregate], dict[str, Any]]:
    summary = _base_summary(output_path)
    summary["zip_path"] = str(zip_path)
    summary["nested_zip_count"] = 0
    aggregates = _build_labels_from_records(
        iter_zip_records(zip_path, summary=summary),
        image_root=image_root,
        output_path=output_path,
        summary=summary,
        limit=limit,
    )
    return aggregates, summary


def _build_labels_from_records(
    records: Iterable[JsonRecord],
    *,
    image_root: Path | None,
    output_path: Path,
    summary: dict[str, Any],
    limit: int | None,
) -> list[LabelAggregate]:
    aggregates: dict[str, LabelAggregate] = {}
    image_index = build_image_index(image_root)

    for record in records:
        summary["parsed_json_count"] += 1
        extracted = extract_label_payload(
            record.payload,
            source_name=record.source_name,
            json_path=record.json_path,
        )
        if extracted is None:
            summary["skipped_json_count"] += 1
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
            break

    apply_image_matches(aggregates.values(), image_index=image_index, output_path=output_path)
    summary["unique_image_count"] = len(aggregates)
    summary["matched_image_count"] = sum(1 for aggregate in aggregates.values() if aggregate.image_exists)
    summary["missing_image_count"] = summary["unique_image_count"] - summary["matched_image_count"]
    summary["image_match_strategy"] = "recursive exact basename match under --image-root"
    summary["searched_image_root"] = str(image_root) if image_root else ""
    return list(aggregates.values())


def select_output_aggregates(
    aggregates: list[LabelAggregate],
    *,
    per_class_limit: int | None,
    include_missing: bool,
    seed: int,
    limit: int | None,
) -> list[LabelAggregate]:
    if per_class_limit is None:
        return aggregates[:limit] if limit is not None else aggregates

    rng = random.Random(seed)
    by_class: dict[str, list[LabelAggregate]] = defaultdict(list)
    for aggregate in aggregates:
        if not include_missing and not aggregate.image_exists:
            continue
        by_class[primary_expected_food(aggregate)].append(aggregate)

    sampled: list[LabelAggregate] = []
    for class_name in sorted(by_class):
        candidates = by_class[class_name]
        rng.shuffle(candidates)
        sampled.extend(candidates[:per_class_limit])

    sampled.sort(key=lambda aggregate: (primary_expected_food(aggregate), aggregate.image_filename))
    return sampled[:limit] if limit is not None else sampled


def build_sampling_summary(
    aggregates: list[LabelAggregate],
    *,
    per_class_limit: int | None,
    include_missing: bool,
    seed: int,
) -> dict[str, Any]:
    return {
        "per_class_limit": per_class_limit,
        "include_missing": include_missing,
        "seed": seed,
        "sampling_strategy": "per_class_random_sample" if per_class_limit is not None else "none",
        "sampled_class_distribution": dict(
            sorted(Counter(primary_expected_food(aggregate) for aggregate in aggregates).items())
        ),
    }


def primary_expected_food(aggregate: LabelAggregate) -> str:
    return aggregate.expected_foods[0] if aggregate.expected_foods else "unknown"


def iter_json_root_records(json_root: Path, *, summary: dict[str, Any]) -> Iterable[JsonRecord]:
    for json_path in sorted(json_root.rglob("*.json")):
        if not json_path.is_file():
            continue
        summary["total_json_count"] += 1
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            summary["skipped_json_count"] += 1
            continue
        yield JsonRecord(
            payload=payload,
            source_name=str(json_root),
            json_path=str(json_path.relative_to(json_root)),
        )


def iter_zip_records(zip_path: Path, *, summary: dict[str, Any]) -> Iterable[JsonRecord]:
    for nested_zip_name, nested_zip_bytes in iter_nested_label_zip_bytes(zip_path):
        summary["nested_zip_count"] += 1
        try:
            with zipfile.ZipFile(io.BytesIO(nested_zip_bytes)) as nested_zip:
                for json_info in nested_zip.infolist():
                    if json_info.is_dir() or not json_info.filename.lower().endswith(".json"):
                        continue
                    summary["total_json_count"] += 1
                    try:
                        payload = json.loads(nested_zip.read(json_info).decode("utf-8-sig"))
                    except (UnicodeDecodeError, json.JSONDecodeError, OSError, zipfile.BadZipFile):
                        summary["skipped_json_count"] += 1
                        continue
                    yield JsonRecord(
                        payload=payload,
                        source_name=nested_zip_name,
                        json_path=json_info.filename,
                    )
        except zipfile.BadZipFile:
            summary["skipped_json_count"] += 1


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
    source_name: str,
    json_path: str,
) -> LabelAggregate | None:
    image_filename = Path(_first_string_value(payload, "Code Name")).name
    if not image_filename:
        return None

    path_categories = _path_categories(json_path)
    cat_1 = _first_string_value(payload, "cat_1") or path_categories["cat_1"]
    cat_2 = _first_string_value(payload, "cat_2") or path_categories["cat_2"]
    cat_3 = _first_string_value(payload, "cat_3") or path_categories["cat_3"]
    folder_food = _food_name_from_path(json_path)
    names = _string_values(payload, "Name")
    foods = _dedupe([folder_food] if folder_food else [name for name in names if name] or [cat_3, cat_2])
    return LabelAggregate(
        image_filename=image_filename,
        expected_foods=foods,
        label_sources=[f"{source_name}:{json_path}"],
        annotation_count=max(1, len(foods)),
        cat_1=cat_1,
        cat_2=cat_2,
        cat_3=cat_3,
    )


def build_image_index(image_root: Path | None) -> ImageIndex:
    if image_root is None:
        return ImageIndex(
            image_root=None,
            by_name={},
            by_normalized_name={},
            by_normalized_stem={},
        )
    by_name: dict[str, Path] = {}
    by_normalized_name: dict[str, Path] = {}
    by_normalized_stem: dict[str, Path] = {}
    for image_path in image_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        by_name.setdefault(image_path.name, image_path)
        by_normalized_name.setdefault(_normalize_filename(image_path.name), image_path)
        by_normalized_stem.setdefault(_normalize_filename(image_path.stem), image_path)
    return ImageIndex(
        image_root=image_root,
        by_name=by_name,
        by_normalized_name=by_normalized_name,
        by_normalized_stem=by_normalized_stem,
    )


def apply_image_matches(
    aggregates: Iterable[LabelAggregate],
    *,
    image_index: ImageIndex,
    output_path: Path,
) -> None:
    for aggregate in aggregates:
        matched_path = image_index.by_name.get(aggregate.image_filename)
        if matched_path is None:
            aggregate.image_path = aggregate.image_filename
            aggregate.image_exists = False
            continue
        aggregate.image_path = os.path.relpath(matched_path, output_path.parent)
        aggregate.image_exists = True


def write_labels_csv(output_path: Path, aggregates: list[LabelAggregate]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for aggregate in aggregates:
            writer.writerow(
                {
                    "image_path": aggregate.image_path or aggregate.image_filename,
                    "image_filename": aggregate.image_filename,
                    "expected_foods": "|".join(aggregate.expected_foods),
                    "image_exists": str(aggregate.image_exists).lower(),
                    "label_source": "|".join(aggregate.label_sources),
                    "annotation_count": aggregate.annotation_count,
                    "cat_1": aggregate.cat_1,
                    "cat_2": aggregate.cat_2,
                    "cat_3": aggregate.cat_3,
                }
            )


def write_allowed_foods_json(path: Path, aggregates: list[LabelAggregate]) -> None:
    allowed_foods = sorted({food for aggregate in aggregates for food in aggregate.expected_foods if food})
    path.write_text(json.dumps(allowed_foods, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_missing_images_csv(
    path: Path,
    aggregates: list[LabelAggregate],
    *,
    image_root: Path | None,
) -> None:
    image_index = build_image_index(image_root)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MISSING_IMAGE_COLUMNS)
        writer.writeheader()
        for aggregate in aggregates:
            if aggregate.image_exists:
                continue
            writer.writerow(
                {
                    "image_filename": aggregate.image_filename,
                    "expected_foods": "|".join(aggregate.expected_foods),
                    "json_path": "|".join(_json_path_from_label_source(source) for source in aggregate.label_sources),
                    "searched_image_root": str(image_root) if image_root else "",
                    "reason": diagnose_missing_image_reason(aggregate.image_filename, image_index),
                }
            )


def diagnose_missing_image_reason(image_filename: str, image_index: ImageIndex) -> str:
    if image_index.image_root is None:
        return "image_root_not_provided"
    normalized_name = _normalize_filename(image_filename)
    if normalized_name in image_index.by_normalized_name:
        return "case_or_unicode_filename_difference"
    if _normalize_filename(Path(image_filename).stem) in image_index.by_normalized_stem:
        return "extension_difference"
    return "not_found_under_recursive_image_root"


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _base_summary(output_path: Path) -> dict[str, Any]:
    return {
        "total_json_count": 0,
        "parsed_json_count": 0,
        "skipped_json_count": 0,
        "unique_image_count": 0,
        "matched_image_count": 0,
        "missing_image_count": 0,
        "output_csv_path": str(output_path),
    }


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
            values.append(_clean_label_text(value))
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
    parts = [_clean_label_text(part) for part in Path(json_path).parts[:-1] if part not in {".", ""}]
    tail = parts[-3:]
    padded = [""] * (3 - len(tail)) + tail
    return {
        "cat_1": padded[0],
        "cat_2": padded[1],
        "cat_3": padded[2],
    }


def _food_name_from_path(json_path: str) -> str:
    for part in reversed(Path(json_path).parts[:-1]):
        cleaned = _clean_folder_label(part)
        if not cleaned:
            continue
        if "_val_json" in cleaned.lower() or cleaned.startswith("[라벨]"):
            continue
        if _contains_korean(cleaned):
            return cleaned
    return ""


def _clean_folder_label(value: str) -> str:
    cleaned = _clean_label_text(value)
    cleaned = re.sub(r"\s+json$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\.json$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _clean_label_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _contains_korean(value: str) -> bool:
    return any("가" <= char <= "힣" for char in unicodedata.normalize("NFC", value))


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _normalize_filename(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold().strip()


def _json_path_from_label_source(value: str) -> str:
    _, separator, json_path = value.partition(":")
    return json_path if separator else value


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = _clean_label_text(value)
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


if __name__ == "__main__":
    main()
