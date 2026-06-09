from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics import compute_metrics, split_food_names
from prompts import FOOD_EVAL_PROMPT
from schemas import LabelRow, PredictionRow

from ai_runtime.cv.food.matcher import match_food_name

DEFAULT_MODEL = "gpt-4o-mini"


class GptJsonParseError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT Vision food detection against image labels.")
    parser.add_argument("--labels", required=True, help="CSV path with image_path and expected_foods columns.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <labels_dir>/outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows for smoke runs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI vision model name.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing API key.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir) if args.output_dir else labels_path.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(labels_path, limit=args.limit)
    predictions = [
        await evaluate_image(
            label=label,
            labels_dir=labels_path.parent,
            model=args.model,
            api_key=os.getenv(args.api_key_env),
        )
        for label in labels
    ]
    metrics = compute_metrics(predictions)
    write_predictions_csv(output_dir / "predictions.csv", predictions)
    write_metrics_json(output_dir / "metrics.json", metrics)
    write_report(output_dir / "report.md", metrics, predictions)
    print(f"Wrote {output_dir / 'predictions.csv'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'report.md'}")


def load_labels(labels_path: Path, *, limit: int | None = None) -> list[LabelRow]:
    rows: list[LabelRow] = []
    with labels_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_id, row in enumerate(reader, start=1):
            image_path = str(row.get("image_path") or "").strip()
            expected_foods = split_food_names(str(row.get("expected_foods") or ""))
            image_exists = _parse_bool(row.get("image_exists"), default=True)
            if not image_path:
                continue
            rows.append(
                LabelRow(
                    row_id=row_id,
                    image_path=image_path,
                    expected_foods=expected_foods,
                    image_exists=image_exists,
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


async def evaluate_image(
    *,
    label: LabelRow,
    labels_dir: Path,
    model: str,
    api_key: str | None,
) -> PredictionRow:
    started = time.perf_counter()
    if not label.image_exists:
        return _failure_row(
            label,
            started=started,
            error_type="data_missing",
            error_message="image_exists=false in labels CSV",
        )

    image_path = _resolve_image_path(label.image_path, labels_dir)
    if not image_path.exists():
        return _failure_row(
            label,
            started=started,
            error_type="data_missing",
            error_message=str(image_path),
        )
    if not api_key:
        return _failure_row(
            label,
            started=started,
            error_type="api_key_missing",
            error_message="OPENAI_API_KEY is not set",
        )

    try:
        raw_response = await call_gpt_vision(
            image_bytes=image_path.read_bytes(),
            media_type=_media_type_for_image(image_path),
            model=model,
            api_key=api_key,
        )
    except GptJsonParseError as exc:
        return PredictionRow(
            row_id=label.row_id,
            image_path=label.image_path,
            expected_foods=label.expected_foods,
            api_success=True,
            json_parse_success=False,
            empty_result=True,
            latency_seconds=round(time.perf_counter() - started, 4),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    except Exception as exc:
        return _failure_row(
            label,
            started=started,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    raw_food_names = extract_food_names(raw_response)
    canonical_food_names, unmatched_food_names = match_canonical_food_names(raw_food_names)
    return PredictionRow(
        row_id=label.row_id,
        image_path=label.image_path,
        expected_foods=label.expected_foods,
        raw_food_names=raw_food_names,
        canonical_food_names=canonical_food_names,
        unmatched_food_names=unmatched_food_names,
        api_success=True,
        json_parse_success=True,
        empty_result=not raw_food_names,
        latency_seconds=round(time.perf_counter() - started, 4),
        raw_response=raw_response,
    )


async def call_gpt_vision(
    *,
    image_bytes: bytes,
    media_type: str,
    model: str,
    api_key: str,
) -> dict[str, Any]:
    from openai import AsyncOpenAI

    data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    client = AsyncOpenAI(api_key=api_key, timeout=20.0, max_retries=2)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FOOD_EVAL_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            }
        ],
    )
    raw_text = response.choices[0].message.content or ""
    return parse_gpt_json(raw_text)


def parse_gpt_json(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        msg = f"Failed to parse GPT JSON: {exc}"
        raise GptJsonParseError(msg) from exc
    if not isinstance(parsed, dict):
        msg = "GPT response JSON is not an object"
        raise GptJsonParseError(msg)
    return parsed


def extract_food_names(raw_response: dict[str, Any]) -> list[str]:
    foods = raw_response.get("foods")
    if not isinstance(foods, list):
        return []
    return [
        name
        for food in foods
        if isinstance(food, dict) and (name := str(food.get("name") or food.get("food_name") or "").strip())
    ]


def match_canonical_food_names(raw_food_names: list[str]) -> tuple[list[str], list[str]]:
    canonical_food_names: list[str] = []
    unmatched_food_names: list[str] = []
    for raw_name in raw_food_names:
        match = match_food_name(raw_name)
        canonical_name = match.matched_food_name or match.query_name
        if canonical_name:
            canonical_food_names.append(canonical_name)
        if match.matched_food_name is None:
            unmatched_food_names.append(raw_name)
    return canonical_food_names, unmatched_food_names


def write_predictions_csv(path: Path, predictions: list[PredictionRow]) -> None:
    fieldnames = [
        "row_id",
        "image_path",
        "expected_foods",
        "raw_food_names",
        "canonical_food_names",
        "unmatched_food_names",
        "api_success",
        "json_parse_success",
        "empty_result",
        "latency_seconds",
        "error_type",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow(
                {
                    "row_id": row.row_id,
                    "image_path": row.image_path,
                    "expected_foods": "|".join(row.expected_foods),
                    "raw_food_names": "|".join(row.raw_food_names),
                    "canonical_food_names": "|".join(row.canonical_food_names),
                    "unmatched_food_names": "|".join(row.unmatched_food_names),
                    "api_success": row.api_success,
                    "json_parse_success": row.json_parse_success,
                    "empty_result": row.empty_result,
                    "latency_seconds": row.latency_seconds if row.latency_seconds is not None else "",
                    "error_type": row.error_type or "",
                    "error_message": row.error_message or "",
                }
            )


def write_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, metrics: dict[str, Any], predictions: list[PredictionRow]) -> None:
    lines = [
        "# GPT Vision Food Eval Report",
        "",
        "## Data Missing Policy",
        "",
        "`image_exists=false` rows and local `image_path` misses are counted as `data_missing`.",
        "They are excluded from model-quality denominators such as API success, JSON parse success, and match rates.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        if key == "class_distribution":
            continue
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Class Distribution",
            "",
        ]
    )
    class_distribution = metrics.get("class_distribution")
    if isinstance(class_distribution, dict) and class_distribution:
        lines.extend(f"- `{food_name}`: {count}" for food_name, count in class_distribution.items())
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Failure Cases",
            "",
        ]
    )
    failures = [row for row in predictions if row.error_type or row.empty_result]
    if not failures:
        lines.append("- None")
    else:
        for row in failures:
            lines.append(f"- row={row.row_id} image={row.image_path} error={row.error_type or 'empty_result'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _failure_row(
    label: LabelRow,
    *,
    started: float,
    error_type: str,
    error_message: str,
) -> PredictionRow:
    return PredictionRow(
        row_id=label.row_id,
        image_path=label.image_path,
        expected_foods=label.expected_foods,
        api_success=False,
        json_parse_success=False,
        empty_result=True,
        latency_seconds=round(time.perf_counter() - started, 4),
        error_type=error_type,
        error_message=error_message,
    )


def _resolve_image_path(image_path: str, labels_dir: Path) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else labels_dir / path


def _parse_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return default


def _media_type_for_image(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


if __name__ == "__main__":
    asyncio.run(main())
