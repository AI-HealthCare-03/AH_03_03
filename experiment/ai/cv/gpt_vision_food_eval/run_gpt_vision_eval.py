from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import difflib
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics import (
    compute_metrics,
    has_any_food_hit,
    is_exact_canonical_match,
    is_exact_raw_match,
    row_precision_recall_f1,
    split_food_names,
)
from prompts import build_food_eval_prompt
from schemas import LabelRow, PredictionRow

from ai_runtime.cv.food.matcher import match_food_name
from ai_runtime.cv.food.normalization import normalize_food_name

DEFAULT_MODEL = "gpt-4o-mini"
UNKNOWN_ALLOWED_FOOD = "unknown"
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROCESSED_RESUME_STATUSES = {"success", "failed", "empty_result", "data_missing"}


class GptJsonParseError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT Vision food detection against image labels.")
    parser.add_argument("--labels", required=True, help="CSV path with image_path and expected_foods columns.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to experiment/ai/cv/gpt_vision_food_eval/outputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows for smoke runs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI vision model name.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing API key.")
    parser.add_argument("--allowed-foods", default=None, help="Optional JSON list of allowed food labels.")
    parser.add_argument("--eval-run-id", default=None, help="Optional Langfuse/grouping ID for this eval run.")
    parser.add_argument("--per-class-limit", type=int, default=None, help="Maximum rows to evaluate per expected food.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for per-class label sampling.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N evaluated rows.")
    parser.add_argument("--save-every", type=int, default=50, help="Save intermediate outputs every N evaluated rows.")
    parser.add_argument(
        "--resume", action="store_true", help="Skip image_path rows already present in predictions.csv."
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir) if args.output_dir else EXPERIMENT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(labels_path, limit=None if args.per_class_limit is not None else args.limit)
    labels = select_eval_labels(
        labels,
        per_class_limit=args.per_class_limit,
        seed=args.seed,
        limit=args.limit,
    )
    allowed_foods = load_allowed_foods(Path(args.allowed_foods)) if args.allowed_foods else None
    eval_run_id = args.eval_run_id or build_eval_run_id()
    prompt = build_food_eval_prompt(allowed_foods)
    tracer = EvalLangfuseTracer.from_env(eval_run_id=eval_run_id, model=args.model, prompt=prompt)
    predictions_path = output_dir / "predictions.csv"
    existing_predictions = load_existing_predictions(predictions_path) if args.resume else []
    processed_image_paths = {row.image_path for row in existing_predictions}
    labels_to_eval = [label for label in labels if label.image_path not in processed_image_paths]
    predictions = list(existing_predictions)
    started = time.perf_counter()
    for processed_count, label in enumerate(labels_to_eval, start=1):
        prediction = await evaluate_image(
            label=label,
            labels_dir=labels_path.parent,
            model=args.model,
            api_key=os.getenv(args.api_key_env),
            allowed_foods=allowed_foods,
        )
        predictions.append(prediction)
        tracer.record_prediction(prediction)
        if args.progress_every > 0 and processed_count % args.progress_every == 0:
            print_progress(
                processed_count=processed_count,
                total_count=len(labels_to_eval),
                predictions=predictions,
                started=started,
            )
        if args.save_every > 0 and processed_count % args.save_every == 0:
            save_outputs(output_dir, predictions)

    tracer.flush()
    save_outputs(output_dir, predictions)
    print(f"Wrote {output_dir / 'predictions.csv'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'report.md'}")


def build_eval_run_id() -> str:
    return f"food-eval-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


class EvalLangfuseTracer:
    def __init__(
        self,
        *,
        client: Any | None,
        eval_run_id: str,
        model: str,
        prompt: str,
    ) -> None:
        self.client = client
        self.eval_run_id = eval_run_id
        self.model = model
        self.prompt = prompt

    @classmethod
    def from_env(cls, *, eval_run_id: str, model: str, prompt: str) -> EvalLangfuseTracer:
        if not _env_flag_enabled(os.getenv("LANGFUSE_ENABLED")):
            return cls(client=None, eval_run_id=eval_run_id, model=model, prompt=prompt)

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
        if not public_key or not secret_key or not host:
            return cls(client=None, eval_run_id=eval_run_id, model=model, prompt=prompt)

        try:
            from langfuse import Langfuse

            client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        except Exception:
            client = None
        return cls(client=client, eval_run_id=eval_run_id, model=model, prompt=prompt)

    def record_prediction(self, row: PredictionRow) -> bool:
        if self.client is None:
            return False

        row_scores = row_precision_recall_f1(row)
        metadata = {
            "eval_run_id": self.eval_run_id,
            "image_path": row.image_path,
            "image_filename": Path(row.image_path).name,
            "expected_foods": row.expected_foods,
            "raw_food_names": row.raw_food_names,
            "allowed_food_names": row.allowed_food_names,
            "canonical_food_names": row.canonical_food_names,
            "unmatched_food_names": row.unmatched_food_names,
            "invalid_label_count": row.invalid_label_count,
            "constrained_by_allowed_foods": row.constrained_by_allowed_foods,
            "status": row_status(row),
            "failure_reason": row.error_message,
            "confidence": row.confidence,
            "latency_seconds": row.latency_seconds,
            "exact_match_raw": is_exact_raw_match(row),
            "exact_match_canonical": is_exact_canonical_match(row),
            "any_hit": has_any_food_hit(row),
            "precision": row_scores["precision"],
            "recall": row_scores["recall"],
            "f1_score": row_scores["f1_score"],
        }
        input_payload = {
            "prompt": self.prompt,
            "image_path": row.image_path,
            "image_filename": Path(row.image_path).name,
            "expected_foods": row.expected_foods,
        }
        output_payload = {
            "raw_food_names": row.raw_food_names,
            "allowed_food_names": row.allowed_food_names,
            "canonical_food_names": row.canonical_food_names,
            "status": row_status(row),
            "failure_reason": row.error_message,
        }
        try:
            observation_context = self.client.start_as_current_observation(
                name="gpt_vision_food_eval",
                as_type="generation",
                input=input_payload,
                metadata=metadata,
                model=self.model,
            )
            observation = observation_context.__enter__()
        except Exception:
            return False

        try:
            try:
                observation.update(output=output_payload)
            except Exception:
                pass
            return True
        finally:
            try:
                observation_context.__exit__(None, None, None)
            except Exception:
                pass

    def flush(self) -> None:
        if self.client is None:
            return
        for method_name in ("flush", "shutdown"):
            method = getattr(self.client, method_name, None)
            if not callable(method):
                continue
            try:
                method()
            except Exception:
                pass


def _env_flag_enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "y", "on"}


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


def select_eval_labels(
    labels: list[LabelRow],
    *,
    per_class_limit: int | None,
    seed: int,
    limit: int | None,
) -> list[LabelRow]:
    if per_class_limit is None:
        return labels

    rng = random.Random(seed)
    by_class: dict[str, list[LabelRow]] = defaultdict(list)
    for label in labels:
        if not label.image_exists:
            continue
        by_class[primary_expected_food(label)].append(label)

    selected: list[LabelRow] = []
    for class_name in sorted(by_class):
        candidates = by_class[class_name]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class_limit])

    selected.sort(key=lambda label: (primary_expected_food(label), label.row_id))
    return selected[:limit] if limit is not None else selected


def primary_expected_food(label: LabelRow) -> str:
    return label.expected_foods[0] if label.expected_foods else "unknown"


def load_allowed_foods(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        msg = f"allowed_foods must be a JSON list: {path}"
        raise ValueError(msg)
    return [food_name for item in payload if (food_name := str(item).strip())]


def load_existing_predictions(path: Path) -> list[PredictionRow]:
    if not path.exists():
        return []

    predictions: list[PredictionRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("status") or "").strip()
            image_path = str(row.get("image_path") or "").strip()
            if not image_path or status not in PROCESSED_RESUME_STATUSES:
                continue
            predictions.append(
                PredictionRow(
                    row_id=_parse_int(row.get("row_id")),
                    image_path=image_path,
                    expected_foods=split_food_names(str(row.get("expected_foods") or "")),
                    raw_food_names=split_food_names(str(row.get("raw_food_names") or "")),
                    allowed_food_names=split_food_names(str(row.get("allowed_food_names") or "")),
                    canonical_food_names=split_food_names(str(row.get("canonical_food_names") or "")),
                    unmatched_food_names=split_food_names(str(row.get("unmatched_food_names") or "")),
                    invalid_label_count=_parse_int(row.get("invalid_label_count")),
                    constrained_by_allowed_foods=_parse_bool(row.get("constrained_by_allowed_foods"), default=False),
                    confidence=_parse_optional_float(row.get("confidence")),
                    api_success=_parse_bool(row.get("api_success"), default=False),
                    json_parse_success=_parse_bool(row.get("json_parse_success"), default=False),
                    empty_result=_parse_bool(row.get("empty_result"), default=status == "empty_result"),
                    latency_seconds=_parse_optional_float(row.get("latency_seconds")),
                    error_type=str(row.get("error_type") or "") or None,
                    error_message=str(row.get("error_message") or row.get("failure_reason") or "") or None,
                )
            )
    return predictions


async def evaluate_image(
    *,
    label: LabelRow,
    labels_dir: Path,
    model: str,
    api_key: str | None,
    allowed_foods: list[str] | None,
) -> PredictionRow:
    started = time.perf_counter()
    if not label.image_exists:
        return _failure_row(
            label,
            started=started,
            error_type="data_missing",
            error_message="image_exists=false in labels CSV",
            constrained_by_allowed_foods=bool(allowed_foods),
        )

    image_path = _resolve_image_path(label.image_path, labels_dir)
    if not image_path.exists():
        return _failure_row(
            label,
            started=started,
            error_type="data_missing",
            error_message=str(image_path),
            constrained_by_allowed_foods=bool(allowed_foods),
        )
    if not api_key:
        return _failure_row(
            label,
            started=started,
            error_type="api_key_missing",
            error_message="OPENAI_API_KEY is not set",
            constrained_by_allowed_foods=bool(allowed_foods),
        )

    try:
        raw_response = await call_gpt_vision(
            image_bytes=image_path.read_bytes(),
            media_type=_media_type_for_image(image_path),
            model=model,
            api_key=api_key,
            allowed_foods=allowed_foods,
        )
    except GptJsonParseError as exc:
        return PredictionRow(
            row_id=label.row_id,
            image_path=label.image_path,
            expected_foods=label.expected_foods,
            api_success=True,
            json_parse_success=False,
            empty_result=False,
            latency_seconds=round(time.perf_counter() - started, 4),
            error_type=type(exc).__name__,
            error_message=str(exc),
            constrained_by_allowed_foods=bool(allowed_foods),
        )
    except Exception as exc:
        return _failure_row(
            label,
            started=started,
            error_type=type(exc).__name__,
            error_message=str(exc),
            constrained_by_allowed_foods=bool(allowed_foods),
        )

    raw_food_names = extract_food_names(raw_response)
    confidence = extract_avg_confidence(raw_response)
    allowed_food_names, invalid_label_count = constrain_food_names(raw_food_names, allowed_foods)
    canonical_food_names, unmatched_food_names = match_canonical_food_names(raw_food_names)
    empty_result = not raw_food_names
    return PredictionRow(
        row_id=label.row_id,
        image_path=label.image_path,
        expected_foods=label.expected_foods,
        raw_food_names=raw_food_names,
        allowed_food_names=allowed_food_names,
        canonical_food_names=canonical_food_names,
        unmatched_food_names=unmatched_food_names,
        invalid_label_count=invalid_label_count,
        constrained_by_allowed_foods=bool(allowed_foods),
        confidence=confidence,
        api_success=True,
        json_parse_success=True,
        empty_result=empty_result,
        latency_seconds=round(time.perf_counter() - started, 4),
        error_message="model_returned_empty_foods" if empty_result else None,
        raw_response=raw_response,
    )


async def call_gpt_vision(
    *,
    image_bytes: bytes,
    media_type: str,
    model: str,
    api_key: str,
    allowed_foods: list[str] | None,
) -> dict[str, Any]:
    from openai import AsyncOpenAI

    data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    prompt = build_food_eval_prompt(allowed_foods)
    client = AsyncOpenAI(api_key=api_key, timeout=20.0, max_retries=2)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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


def extract_avg_confidence(raw_response: dict[str, Any]) -> float | None:
    foods = raw_response.get("foods")
    if not isinstance(foods, list):
        return None
    confidences: list[float] = []
    for food in foods:
        if not isinstance(food, dict):
            continue
        try:
            confidences.append(float(food.get("confidence")))
        except (TypeError, ValueError):
            continue
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 4)


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


def constrain_food_names(
    raw_food_names: list[str],
    allowed_foods: list[str] | None,
) -> tuple[list[str], int]:
    if not allowed_foods:
        return [], 0

    allowed_index = {_normalize_for_allowed(food_name): food_name for food_name in allowed_foods}
    allowed_food_names: list[str] = []
    invalid_label_count = 0
    for raw_name in raw_food_names:
        normalized_raw = _normalize_for_allowed(raw_name)
        if normalized_raw in allowed_index:
            allowed_food_names.append(allowed_index[normalized_raw])
            continue

        invalid_label_count += 1
        allowed_food_names.append(_correct_to_allowed_food(raw_name, allowed_index) or UNKNOWN_ALLOWED_FOOD)
    return allowed_food_names, invalid_label_count


def _correct_to_allowed_food(raw_name: str, allowed_index: dict[str, str]) -> str | None:
    match = match_food_name(raw_name)
    for candidate in (match.matched_food_name, match.query_name):
        normalized_candidate = _normalize_for_allowed(candidate or "")
        if normalized_candidate in allowed_index:
            return allowed_index[normalized_candidate]

    close_matches = difflib.get_close_matches(_normalize_for_allowed(raw_name), allowed_index.keys(), n=1, cutoff=0.86)
    if close_matches:
        return allowed_index[close_matches[0]]
    return None


def write_predictions_csv(path: Path, predictions: list[PredictionRow]) -> None:
    fieldnames = [
        "row_id",
        "image_path",
        "image_filename",
        "expected_foods",
        "raw_food_names",
        "allowed_food_names",
        "canonical_food_names",
        "unmatched_food_names",
        "invalid_label_count",
        "confidence",
        "constrained_by_allowed_foods",
        "status",
        "failure_reason",
        "exact_match_raw",
        "exact_match_canonical",
        "any_hit",
        "precision",
        "recall",
        "f1_score",
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
            row_scores = row_precision_recall_f1(row)
            writer.writerow(
                {
                    "row_id": row.row_id,
                    "image_path": row.image_path,
                    "image_filename": Path(row.image_path).name,
                    "expected_foods": "|".join(row.expected_foods),
                    "raw_food_names": "|".join(row.raw_food_names),
                    "allowed_food_names": "|".join(row.allowed_food_names),
                    "canonical_food_names": "|".join(row.canonical_food_names),
                    "unmatched_food_names": "|".join(row.unmatched_food_names),
                    "invalid_label_count": row.invalid_label_count,
                    "confidence": row.confidence if row.confidence is not None else "",
                    "constrained_by_allowed_foods": row.constrained_by_allowed_foods,
                    "status": row_status(row),
                    "failure_reason": row.error_message or "",
                    "exact_match_raw": is_exact_raw_match(row),
                    "exact_match_canonical": is_exact_canonical_match(row),
                    "any_hit": has_any_food_hit(row),
                    "precision": row_scores["precision"],
                    "recall": row_scores["recall"],
                    "f1_score": row_scores["f1_score"],
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


def save_outputs(output_dir: Path, predictions: list[PredictionRow]) -> None:
    metrics = compute_metrics(predictions)
    write_predictions_csv(output_dir / "predictions.csv", predictions)
    write_metrics_json(output_dir / "metrics.json", metrics)
    write_report(output_dir / "report.md", metrics, predictions)


def print_progress(
    *,
    processed_count: int,
    total_count: int,
    predictions: list[PredictionRow],
    started: float,
) -> None:
    elapsed_seconds = max(time.perf_counter() - started, 0.0)
    remaining_count = max(total_count - processed_count, 0)
    avg_seconds_per_row = elapsed_seconds / processed_count if processed_count else 0.0
    eta_seconds = avg_seconds_per_row * remaining_count
    latencies = [row.latency_seconds for row in predictions if row.latency_seconds is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    status_counts = {
        "success": sum(row_status(row) == "success" for row in predictions),
        "failed": sum(row_status(row) in {"failed", "data_missing"} for row in predictions),
        "empty": sum(row_status(row) == "empty_result" for row in predictions),
    }
    print(
        "[progress] "
        f"{processed_count}/{total_count} "
        f"success={status_counts['success']} "
        f"failed={status_counts['failed']} "
        f"empty={status_counts['empty']} "
        f"avg_latency={avg_latency:.2f}s "
        f"eta={format_duration(eta_seconds)}"
    )


def write_report(path: Path, metrics: dict[str, Any], predictions: list[PredictionRow]) -> None:
    class_distribution = metrics.get("class_distribution")
    class_level_metrics = metrics.get("class_level_metrics")
    confidence_bins = metrics.get("confidence_bins")
    lines = [
        "# GPT Vision Food Eval Report",
        "",
        "## 1. 평가 요약",
        "",
        f"- total_rows: {metrics['total_rows']}",
        f"- evaluable_image_count: {metrics['evaluable_image_count']}",
        f"- raw_exact_match_rate: {metrics['raw_exact_match_rate']}",
        f"- canonical_exact_match_rate: {metrics['canonical_exact_match_rate']}",
        f"- precision / recall / f1: {metrics['precision']} / {metrics['recall']} / {metrics['f1_score']}",
        "",
        "## 2. 데이터 누락/평가 가능 이미지 수",
        "",
        f"- data_missing_count: {metrics['data_missing_count']}",
        f"- api_failed_count: {metrics['api_failed_count']}",
        f"- json_parse_failed_count: {metrics['json_parse_failed_count']}",
        f"- empty_result_count: {metrics['empty_result_count']}",
        "",
        "image_exists=false 또는 image_path 없음은 data_missing으로 분리했고 모델 실패로 계산하지 않았음.",
        "",
        "## 3. Raw 기준 결과",
        "",
        f"- raw_exact_match_rate: {metrics['raw_exact_match_rate']}",
        f"- any_hit_rate: {metrics['any_hit_rate']}",
        "",
        "## 4. Canonical matcher 기준 결과",
        "",
        f"- canonical_exact_match_rate: {metrics['canonical_exact_match_rate']}",
        f"- canonical_any_hit_rate: {metrics['canonical_any_hit_rate']}",
        "",
        "raw 기준과 canonical 기준을 분리해 음식명 흔들림과 서비스 매칭 후 성능을 따로 평가함.",
        "",
        "## 5. Allowed foods constrained 기준 결과",
        "",
        f"- constrained_by_allowed_foods: {metrics['constrained_by_allowed_foods']}",
        f"- constrained_exact_match_rate: {metrics['constrained_exact_match_rate']}",
        f"- constrained_any_hit_rate: {metrics['constrained_any_hit_rate']}",
        f"- invalid_label_count: {metrics['invalid_label_count']}",
        f"- invalid_label_rate: {metrics['invalid_label_rate']}",
        f"- unknown_count: {metrics['unknown_count']}",
        f"- unknown_rate: {metrics['unknown_rate']}",
        "",
        "allowed_foods 제한 평가가 켜진 경우 invalid_label_rate로 GPT가 허용 목록 밖 음식을 생성하는 비율을 측정함.",
        "",
        "## 6. Precision / Recall / F1",
        "",
        f"- precision: {metrics['precision']}",
        f"- recall: {metrics['recall']}",
        f"- f1_score: {metrics['f1_score']}",
        f"- macro_precision: {metrics['macro_precision']}",
        f"- macro_recall: {metrics['macro_recall']}",
        f"- macro_f1_score: {metrics['macro_f1_score']}",
        "",
        "## 7. Confidence 구간별 정확도",
        "",
        f"- avg_confidence: {metrics['avg_confidence']}",
        f"- confidence_correct_avg: {metrics['confidence_correct_avg']}",
        f"- confidence_wrong_avg: {metrics['confidence_wrong_avg']}",
        "",
    ]
    if isinstance(confidence_bins, dict):
        for bin_name, bin_metrics in confidence_bins.items():
            lines.append(
                f"- {bin_name}: sample_count={bin_metrics['sample_count']}, accuracy={bin_metrics['accuracy']}"
            )
    lines.extend(
        [
            "",
            "## 8. 클래스별 결과",
            "",
        ]
    )
    if isinstance(class_distribution, dict) and class_distribution:
        lines.append("### Class Distribution")
        lines.extend(f"- `{food_name}`: {count}" for food_name, count in class_distribution.items())
    if isinstance(class_level_metrics, list) and class_level_metrics:
        lines.extend(["", "### Class Level Metrics"])
        for row in class_level_metrics:
            lines.append(
                "- "
                f"{row['expected_food']}: sample_count={row['sample_count']}, "
                f"exact={row['exact_match_rate']}, any_hit={row['any_hit_rate']}, "
                f"precision={row['precision']}, recall={row['recall']}, f1={row['f1_score']}"
            )
    lines.extend(
        [
            "",
            "## 9. 실패 케이스 요약",
            "",
        ]
    )
    failures = [row for row in predictions if row.error_type or row.empty_result]
    if not failures:
        lines.append("- None")
    else:
        for row in failures[:100]:
            lines.append(
                f"- row={row.row_id} image={row.image_path} status={row_status(row)} reason={row.error_message or ''}"
            )
        if len(failures) > 100:
            lines.append(f"- ... truncated {len(failures) - 100} additional failure rows")
    lines.extend(
        [
            "",
            "## 10. 해석",
            "",
            "- image_exists=false 또는 image_path 없음은 data_missing으로 분리했고 모델 실패로 계산하지 않았음.",
            "- raw 기준과 canonical 기준을 분리해 음식명 흔들림과 서비스 매칭 후 성능을 따로 평가함.",
            "- allowed_foods 제한 평가가 켜진 경우 invalid_label_rate로 GPT가 허용 목록 밖 음식을 생성하는 비율을 측정함.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row_status(row: PredictionRow) -> str:
    if row.error_type == "data_missing":
        return "data_missing"
    if row.error_type:
        return "failed"
    if row.empty_result:
        return "empty_result"
    return "success"


def _failure_row(
    label: LabelRow,
    *,
    started: float,
    error_type: str,
    error_message: str,
    constrained_by_allowed_foods: bool,
) -> PredictionRow:
    return PredictionRow(
        row_id=label.row_id,
        image_path=label.image_path,
        expected_foods=label.expected_foods,
        api_success=False,
        json_parse_success=False,
        empty_result=False,
        latency_seconds=round(time.perf_counter() - started, 4),
        error_type=error_type,
        error_message=error_message,
        constrained_by_allowed_foods=constrained_by_allowed_foods,
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


def _parse_int(value: object, *, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _parse_optional_float(value: object) -> float | None:
    try:
        raw_value = str(value).strip()
        return float(raw_value) if raw_value else None
    except (TypeError, ValueError):
        return None


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, sec = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minute}m {sec}s"
    if minute:
        return f"{minute}m {sec}s"
    return f"{sec}s"


def _normalize_for_allowed(value: str) -> str:
    return normalize_food_name(value).casefold()


def _media_type_for_image(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


if __name__ == "__main__":
    asyncio.run(main())
