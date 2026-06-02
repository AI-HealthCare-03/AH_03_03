from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS = {
    "DM CatBoost": REPO_ROOT / "ai_runtime" / "ml" / "artifacts" / "dm" / "catboost",
    "HTN CatBoost": REPO_ROOT / "ai_runtime" / "ml" / "artifacts" / "htn" / "catboost",
    "DL CatBoost": REPO_ROOT / "ai_runtime" / "ml" / "artifacts" / "dl" / "catboost",
}
ENV_KEYS = [
    "OPENAI_API_KEY",
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
    parser = argparse.ArgumentParser(
        description="Audit ai_runtime runtime capability wiring without external API calls."
    )
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
    rows.extend(_audit_llm_runtime_scope())
    rows.extend(_audit_llm_prompt_locations())
    rows.extend(_audit_keyword_rag_poc())
    rows.extend(_audit_ocr_parsers())
    rows.extend(_audit_async_runtime_scope())
    rows.extend(_audit_intentional_backlog())

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
        from ai_runtime.ml.inference.disease_risk_service import warmup_chronic_disease_models

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
            module="ai_runtime.ml.X2.health_stage_classifier",
            ready_category="READY_RULE_BASED",
        ),
        _import_row(
            area="rule_based",
            item="DiseaseFoodScorer",
            module="ai_runtime.cv.food.nutrition.scoring.disease_food_scorer",
            ready_category="READY_RULE_BASED",
        ),
    ]

    try:
        from ai_runtime.cv.food.nutrition.scoring.disease_food_scorer import DiseaseFoodScorer

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
            module="ai_runtime.cv.providers.gpt_vision",
            ready_category="READY_PROVIDER_CODE_ONLY",
        ),
        _import_row(
            area="provider",
            item="Clova OCR client import",
            module="ai_runtime.ocr.providers.clova_ocr.clova_client",
            ready_category="DEFERRED_PROVIDER",
        ),
        AuditRow(
            area="provider",
            item="Clova OCR runtime status",
            status="DEFERRED",
            detail="preserved as PoC/deferred provider; official demo path does not call Clova OCR",
            category="DEFERRED_PROVIDER",
        ),
        _import_row(
            area="provider",
            item="OpenAI LLM client",
            module="ai_runtime.llm.llm_client",
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
    llm_dir = REPO_ROOT / "ai_runtime" / "llm"
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


def _audit_llm_runtime_scope() -> list[AuditRow]:
    rows: list[AuditRow] = [
        _import_row(
            area="llm_runtime",
            item="Analysis/Diet explanation service",
            module="ai_runtime.llm.explanation_service",
            ready_category="READY_RUNTIME",
        ),
        _import_row(
            area="llm_runtime",
            item="LLM shared schemas",
            module="ai_runtime.llm.schemas",
            ready_category="READY_RUNTIME",
        ),
        AuditRow(
            area="llm_runtime",
            item="Official analysis runtime wiring",
            status="OK",
            detail="app/services/analysis.py calls explanation_service + keyword RAG context",
            category="READY_RUNTIME",
        ),
        AuditRow(
            area="llm_runtime",
            item="Official diet runtime wiring",
            status="OK",
            detail="app/services/diets.py calls generate_diet_score_explanation",
            category="READY_RUNTIME",
        ),
        AuditRow(
            area="llm_runtime",
            item="Official chatbot runtime wiring",
            status="OK",
            detail="app/services/chatbot.py calls ai_runtime.llm.response_router with use_real_llm=False",
            category="READY_RUNTIME",
        ),
    ]
    rows.extend(
        [
            _import_row(
                area="llm_prepared",
                item="response_router",
                module="ai_runtime.llm.response_router",
                ready_category="PREPARED_NOT_WIRED",
            ),
            _import_row(
                area="llm_prepared",
                item="health_chatbot",
                module="ai_runtime.llm.health_chatbot",
                ready_category="PREPARED_NOT_WIRED",
            ),
            _import_row(
                area="llm_prepared",
                item="rule_engine",
                module="ai_runtime.llm.rule_engine",
                ready_category="PREPARED_NOT_WIRED",
            ),
            _import_row(
                area="llm_prepared",
                item="llm_generator",
                module="ai_runtime.llm.llm_generator",
                ready_category="PREPARED_NOT_WIRED",
            ),
            _import_row(
                area="llm_prepared",
                item="recommendation_message",
                module="ai_runtime.llm.recommendation_message",
                ready_category="PREPARED_NOT_WIRED",
            ),
            _import_row(
                area="llm_prepared",
                item="risk_mapper",
                module="ai_runtime.llm.risk_mapper",
                ready_category="PREPARED_NOT_WIRED",
            ),
            _import_row(
                area="llm_legacy_poc",
                item="rag_generator",
                module="ai_runtime.llm.rag_generator",
                ready_category="PREPARED_NOT_WIRED",
            ),
            _import_row(
                area="llm_legacy_poc",
                item="rag_sources",
                module="ai_runtime.llm.rag_sources",
                ready_category="PREPARED_NOT_WIRED",
            ),
        ]
    )
    rows.extend(
        [
            AuditRow(
                area="llm_backlog",
                item="Vector RAG / pgvector embedding search",
                status="P2",
                detail="keyword RAG PoC is ready; vector retrieval is intentionally deferred",
                category="P2_BACKLOG",
            ),
            AuditRow(
                area="llm_backlog",
                item="LangChain / LangGraph orchestration",
                status="P2",
                detail="not introduced for MVP; direct service functions remain the runtime boundary",
                category="P2_BACKLOG",
            ),
            AuditRow(
                area="llm_backlog",
                item="Operating LLM/RAG evaluation pipeline",
                status="P2",
                detail="Langfuse trace metadata exists; evaluation dataset/workflow is deferred",
                category="P2_BACKLOG",
            ),
        ]
    )
    return rows


def _audit_ocr_parsers() -> list[AuditRow]:
    rows = [
        _import_row(
            area="ocr_parser",
            item="Medication OCR parser",
            module="ai_runtime.ocr.medication.parser",
            ready_category="READY_RULE_BASED",
        ),
        _import_row(
            area="ocr_parser",
            item="Checkup OCR extractor",
            module="ai_runtime.ocr.checkup.extractor",
            ready_category="READY_PROVIDER_CODE_ONLY",
        ),
    ]
    try:
        from ai_runtime.ocr.medication.parser import parse_medication_text

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


def _audit_keyword_rag_poc() -> list[AuditRow]:
    rows: list[AuditRow] = [
        _import_row(
            area="rag_poc",
            item="Keyword RAG retriever",
            module="ai_runtime.llm.rag.keyword_retriever",
            ready_category="READY_RAG_POC",
        ),
        _import_row(
            area="rag_poc",
            item="RAG source loader",
            module="ai_runtime.llm.rag.source_loader",
            ready_category="READY_RAG_POC",
        ),
        _import_row(
            area="rag_trace",
            item="Keyword RAG Langfuse trace module",
            module="ai_runtime.llm.rag.tracing",
            ready_category="READY_PROVIDER_CODE_ONLY",
        ),
    ]
    try:
        from ai_runtime.llm.rag.keyword_retriever import retrieve_keyword_rag_contexts
        from ai_runtime.llm.rag.source_loader import load_rag_source_index
        from ai_runtime.llm.rag.tracing import build_keyword_rag_trace_metadata

        source_index = load_rag_source_index()
        status_counts = Counter(metadata.status for metadata in source_index)
        rows.append(
            AuditRow(
                area="rag_poc",
                item="RAG source registry",
                status="OK" if source_index else "FAIL",
                detail=(
                    f"source_count={len(source_index)}, "
                    f"status_distribution={','.join(f'{status}:{count}' for status, count in sorted(status_counts.items())) or '-'}"
                ),
                category="READY_RAG_POC" if source_index else "NOT_IMPLEMENTED",
            )
        )

        contexts = retrieve_keyword_rag_contexts(
            user_message="공복혈당 관리",
            disease_type="DIABETES",
            include_safety_disclaimer=True,
        )
        source_ids = [str(context.metadata.get("id")) for context in contexts]
        rows.append(
            AuditRow(
                area="rag_poc",
                item="Keyword RAG demo retrieval",
                status="OK" if {"diabetes", "safety_disclaimer"}.issubset(source_ids) else "FAIL",
                detail=f"sources={','.join(source_ids) or '-'}",
                category="READY_RAG_POC" if contexts else "NOT_IMPLEMENTED",
            )
        )
        metadata = build_keyword_rag_trace_metadata(
            query="공복혈당 관리",
            disease_type="DIABETES",
            contexts=contexts,
            top_k=3,
            include_safety_disclaimer=True,
        )
        rows.append(
            AuditRow(
                area="rag_trace",
                item="Keyword RAG trace metadata shape",
                status="OK"
                if {"prompt_version", "retrieved_source_ids", "source_status", "top_k", "fallback"}.issubset(metadata)
                else "FAIL",
                detail=(
                    f"prompt_version={metadata.get('prompt_version')}, "
                    f"sources={','.join(metadata.get('retrieved_source_ids', [])) or '-'}, "
                    f"fallback={metadata.get('fallback')}"
                ),
                category="READY_PROVIDER_CODE_ONLY",
            )
        )
    except Exception as exc:
        rows.append(
            AuditRow(
                area="rag_poc",
                item="Keyword RAG demo retrieval",
                status="FAIL",
                detail=exc.__class__.__name__,
                category="NEEDS_DEPENDENCY",
            )
        )
    return rows


def _audit_intentional_backlog() -> list[AuditRow]:
    return [
        AuditRow(
            area="backlog",
            item="자체 식단 CV 모델 artifact",
            status="P2",
            detail="no local food CV model artifact; MVP uses rule_based_food_detection + nutrition scorer",
            category="P2_BACKLOG",
        ),
        AuditRow(
            area="backlog",
            item="vector RAG / embedding search",
            status="P2",
            detail="RAG-ready interfaces exist, but vector DB retrieval/embedding search is not implemented",
            category="P2_BACKLOG",
        ),
        AuditRow(
            area="backlog",
            item="OCR/CV/ML/LLM production async workflow",
            status="P2",
            detail="DEMO_ECHO Redis Stream skeleton exists; real OCR/CV/ML/LLM jobs, retry/DLQ, heartbeat are deferred",
            category="P2_BACKLOG",
        ),
        AuditRow(
            area="backlog",
            item="약봉투/처방전 OCR 실제 provider 연결",
            status="P1",
            detail="medication parser skeleton exists; real provider integration for medication/prescription OCR is not wired",
            category="P1_BACKLOG",
        ),
        AuditRow(
            area="backlog",
            item="실제 LLM/RAG 운영 연결 고도화",
            status="P2",
            detail="provider code and rule-based fallback exist; production RAG grounding/observability policy is deferred",
            category="P2_BACKLOG",
        ),
    ]


def _audit_async_runtime_scope() -> list[AuditRow]:
    return [
        AuditRow(
            area="async_scope",
            item="FastAPI router and DB I/O",
            status="INFO",
            detail="FastAPI handlers and Tortoise/asyncpg DB access are async-based",
            category="READY_RUNTIME",
        ),
        AuditRow(
            area="async_scope",
            item="Redis runtime use",
            status="DEMO_QUEUE",
            detail="Redis is used for health checks and the DEMO_ECHO Redis Stream skeleton",
            category="READY_RUNTIME",
        ),
        AuditRow(
            area="async_scope",
            item="OCR/CV/ML/LLM workflow",
            status="SYNC",
            detail="official MVP workflows remain synchronous API flows before demo",
            category="READY_RUNTIME",
        ),
        AuditRow(
            area="async_scope",
            item="AI Worker consumer",
            status="READY_RUNTIME",
            detail="ai_runtime/main.py runs Redis Stream handlers for analysis, OCR, diet, medication, and service jobs",
            category="READY_RUNTIME",
        ),
        AuditRow(
            area="async_scope",
            item="async_jobs table/API",
            status="READY_RUNTIME",
            detail="async_jobs model, /api/v1/jobs status API, retry/DLQ, and AI/service stream job enqueue paths are implemented",
            category="READY_RUNTIME",
        ),
        AuditRow(
            area="async_scope",
            item="AnalysisResult.async_job_id",
            status="OPTIONAL",
            detail="analysis.run async job stores result ids in job result_payload; direct DB FK linkage remains optional",
            category="READY_RUNTIME",
        ),
    ]


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
    widths = [min(max(len(str(row[index])) for row in [headers, *values]), 110) for index in range(len(headers))]
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
        "READY_RUNTIME",
        "READY_LOCAL_MODEL",
        "READY_RULE_BASED",
        "READY_RAG_POC",
        "READY_PROVIDER_CODE_ONLY",
        "PREPARED_NOT_WIRED",
        "NEEDS_ENV",
        "NEEDS_DEPENDENCY",
        "NOT_IMPLEMENTED",
        "DEFERRED_PROVIDER",
        "P1_BACKLOG",
        "P2_BACKLOG",
    ]
    print("\nSummary")
    for category in categories:
        count = sum(1 for row in rows if row.category == category)
        print(f"- {category}: {count}")
    print(
        "\nThis audit reports current implemented/provider/backlog status. "
        "NOT_IMPLEMENTED/P2_BACKLOG items may be intentional MVP exclusions."
    )


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
