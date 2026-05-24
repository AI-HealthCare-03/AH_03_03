from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS = {
    "DM CatBoost": REPO_ROOT / "ai_worker" / "ml" / "artifacts" / "dm" / "catboost",
    "HTN CatBoost": REPO_ROOT / "ai_worker" / "ml" / "artifacts" / "htn" / "catboost",
    "DL CatBoost": REPO_ROOT / "ai_worker" / "ml" / "artifacts" / "dl" / "catboost",
}
ENV_KEYS = [
    "OPENAI_API_KEY",
    "CLOVA_OCR_SECRET_KEY",
    "CLOVA_OCR_API_URL",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]
PROMPT_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"prompt",
        r"system_prompt",
        r"user_prompt",
        r"messages",
        r"role\s*=\s*[\"']system[\"']",
        r"[\"']role[\"']\s*:\s*[\"']system[\"']",
    )
]


@dataclass
class AuditRow:
    area: str
    item: str
    status: str
    detail: str
    category: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ai_worker runtime capability wiring without external API calls.")
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip CatBoost model warmup and only inspect artifact files.",
    )
    args = parser.parse_args()

    rows: list[AuditRow] = []
    rows.extend(_audit_local_model_artifacts(skip_warmup=args.skip_warmup))
    rows.extend(_audit_rule_based_and_scorers())
    rows.extend(_audit_external_provider_code())
    rows.extend(_audit_llm_prompt_locations())
    rows.extend(_audit_ocr_parsers())

    _print_table(rows)
    _print_summary(rows)


def _audit_local_model_artifacts(skip_warmup: bool) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for label, artifact_dir in ARTIFACTS.items():
        cbm_count = len(list(artifact_dir.glob("model_fold*.cbm"))) if artifact_dir.exists() else 0
        metadata_ok = all(
            (artifact_dir / filename).exists()
            for filename in ("feature_columns.json", "threshold.json", "metrics.json", "experiment_config.json")
        )
        status = "OK" if artifact_dir.exists() and cbm_count > 0 and metadata_ok else "FAIL"
        category = "READY_LOCAL_MODEL" if status == "OK" else "NOT_IMPLEMENTED"
        rows.append(
            AuditRow(
                area="local_model",
                item=label,
                status=status,
                detail=f"path={_rel(artifact_dir)}, cbm_count={cbm_count}, metadata_ok={metadata_ok}",
                category=category,
            )
        )

    if skip_warmup:
        rows.append(
            AuditRow(
                area="local_model",
                item="CatBoost predictor warmup",
                status="SKIP",
                detail="skipped by --skip-warmup",
                category="READY_LOCAL_MODEL",
            )
        )
        return rows

    try:
        from ai_worker.ml.inference.disease_risk_service import warmup_chronic_disease_models

        warmup_results = warmup_chronic_disease_models()
    except ImportError as exc:
        rows.append(
            AuditRow(
                area="local_model",
                item="CatBoost predictor warmup",
                status="FAIL",
                detail=f"ImportError:{exc.__class__.__name__}",
                category="NEEDS_DEPENDENCY",
            )
        )
        return rows
    except Exception as exc:
        rows.append(
            AuditRow(
                area="local_model",
                item="CatBoost predictor warmup",
                status="FAIL",
                detail=exc.__class__.__name__,
                category="NEEDS_DEPENDENCY",
            )
        )
        return rows

    for disease_key, payload in warmup_results.items():
        status = "OK" if payload.get("status") == "ok" else "FAIL"
        rows.append(
            AuditRow(
                area="local_model",
                item=f"{disease_key} warmup",
                status=status,
                detail=f"status={payload.get('status')}, models={payload.get('model_count')}, features={payload.get('feature_count')}",
                category="READY_LOCAL_MODEL" if status == "OK" else "NEEDS_DEPENDENCY",
            )
        )
    return rows


def _audit_rule_based_and_scorers() -> list[AuditRow]:
    rows: list[AuditRow] = [
        _import_row(
            area="rule_based",
            item="X2 health stage classifier",
            module="ai_worker.ml.X2.health_stage_classifier",
            ready_category="READY_RULE_BASED",
        ),
        _import_row(
            area="rule_based",
            item="DiseaseFoodScorer",
            module="ai_worker.cv.food.nutrition.scoring.disease_food_scorer",
            ready_category="READY_RULE_BASED",
        ),
    ]

    try:
        from ai_worker.cv.food.nutrition.scoring.disease_food_scorer import DiseaseFoodScorer

        count = len(DiseaseFoodScorer().load_runtime_scores())
        rows.append(
            AuditRow(
                area="rule_based",
                item="DiseaseFoodScorer runtime CSV",
                status="OK",
                detail=f"record_count={count}",
                category="READY_RULE_BASED",
            )
        )
    except Exception as exc:
        rows.append(
            AuditRow(
                area="rule_based",
                item="DiseaseFoodScorer runtime CSV",
                status="FAIL",
                detail=exc.__class__.__name__,
                category="NEEDS_DEPENDENCY",
            )
        )

    rows.extend(
        [
            AuditRow(
                area="rule_based",
                item="OBESITY",
                status="INFO",
                detail="official AnalysisResult target; current path is rule_based because no obesity ML artifact exists",
                category="READY_RULE_BASED",
            ),
            AuditRow(
                area="rule_based",
                item="ANEM",
                status="INFO",
                detail="not an official AnalysisResult target; reference classification for X2/diet scoring",
                category="READY_RULE_BASED",
            ),
        ]
    )
    return rows


def _audit_external_provider_code() -> list[AuditRow]:
    rows = [
        _import_row(
            area="provider",
            item="GPT Vision provider",
            module="ai_worker.cv.providers.gpt_vision",
            ready_category="READY_PROVIDER_CODE_ONLY",
        ),
        _import_row(
            area="provider",
            item="Clova OCR client",
            module="ai_worker.ocr.providers.clova_ocr.clova_client",
            ready_category="READY_PROVIDER_CODE_ONLY",
        ),
        _import_row(
            area="provider",
            item="OpenAI LLM client",
            module="ai_worker.llm.llm_client",
            ready_category="READY_PROVIDER_CODE_ONLY",
        ),
    ]
    for key in ENV_KEYS:
        is_set = bool(os.getenv(key))
        rows.append(
            AuditRow(
                area="provider_env",
                item=key,
                status="SET" if is_set else "MISSING",
                detail="configured" if is_set else "not configured",
                category="READY_PROVIDER_CODE_ONLY" if is_set else "NEEDS_ENV",
            )
        )
    return rows


def _audit_llm_prompt_locations() -> list[AuditRow]:
    rows: list[AuditRow] = []
    llm_dir = REPO_ROOT / "ai_worker" / "llm"
    prompt_locations: list[str] = []
    for path in sorted(llm_dir.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts or path.suffix not in {".py", ".md"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            matched = sorted({pattern.pattern for pattern in PROMPT_PATTERNS if pattern.search(line)})
            if matched:
                prompt_locations.append(f"{_rel(path)}:{line_number} ({', '.join(matched)})")

    if not prompt_locations:
        return [
            AuditRow(
                area="llm_prompt",
                item="Prompt locations",
                status="INFO",
                detail="inline prompt / rule-based fallback 중심",
                category="READY_RULE_BASED",
            )
        ]

    for location in prompt_locations:
        rows.append(
            AuditRow(
                area="llm_prompt",
                item="Prompt reference",
                status="INFO",
                detail=location,
                category="READY_PROVIDER_CODE_ONLY",
            )
        )
    return rows


def _audit_ocr_parsers() -> list[AuditRow]:
    rows = [
        _import_row(
            area="ocr_parser",
            item="Medication OCR parser",
            module="ai_worker.ocr.medication.parser",
            ready_category="READY_RULE_BASED",
        ),
        _import_row(
            area="ocr_parser",
            item="Checkup OCR extractor",
            module="ai_worker.ocr.checkup.extractor",
            ready_category="READY_PROVIDER_CODE_ONLY",
        ),
    ]
    try:
        from ai_worker.ocr.medication.parser import parse_medication_text

        sample = "샘플약 10mg 하루 2회 3일분 식후 복용"
        parsed = parse_medication_text(sample)
        rows.append(
            AuditRow(
                area="ocr_parser",
                item="Medication parser sample",
                status="OK" if parsed.items else "FAIL",
                detail=f"items={len(parsed.items)}, warnings={','.join(parsed.warnings) or '-'}",
                category="READY_RULE_BASED" if parsed.items else "NOT_IMPLEMENTED",
            )
        )
    except Exception as exc:
        rows.append(
            AuditRow(
                area="ocr_parser",
                item="Medication parser sample",
                status="FAIL",
                detail=exc.__class__.__name__,
                category="NEEDS_DEPENDENCY",
            )
        )
    return rows


def _import_row(area: str, item: str, module: str, ready_category: str) -> AuditRow:
    try:
        importlib.import_module(module)
    except ImportError as exc:
        return AuditRow(
            area=area,
            item=item,
            status="FAIL",
            detail=f"ImportError:{exc.__class__.__name__}",
            category="NEEDS_DEPENDENCY",
        )
    except Exception as exc:
        return AuditRow(
            area=area,
            item=item,
            status="FAIL",
            detail=exc.__class__.__name__,
            category="NEEDS_DEPENDENCY",
        )
    return AuditRow(area=area, item=item, status="OK", detail=f"import={module}", category=ready_category)


def _print_table(rows: list[AuditRow]) -> None:
    headers = ["Area", "Item", "Status", "Detail", "Category"]
    values = [[row.area, row.item, row.status, row.detail, row.category] for row in rows]
    widths = [
        min(max(len(str(row[index])) for row in [headers, *values]), 110)
        for index in range(len(headers))
    ]
    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for value in values:
        print(_format_row(value, widths))


def _format_row(values: list[str], widths: list[int]) -> str:
    rendered = []
    for value, width in zip(values, widths, strict=True):
        text = str(value)
        if len(text) > width:
            text = f"{text[: width - 1]}…"
        rendered.append(text.ljust(width))
    return " | ".join(rendered)


def _print_summary(rows: list[AuditRow]) -> None:
    categories = [
        "READY_LOCAL_MODEL",
        "READY_RULE_BASED",
        "READY_PROVIDER_CODE_ONLY",
        "NEEDS_ENV",
        "NEEDS_DEPENDENCY",
        "NOT_IMPLEMENTED",
    ]
    print("\nSummary")
    for category in categories:
        count = sum(1 for row in rows if row.category == category)
        print(f"- {category}: {count}")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
