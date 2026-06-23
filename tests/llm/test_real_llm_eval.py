from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

from ai_runtime.llm.evaluation import evaluate_llm_case_quality

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_LLM_EVAL", "").lower() != "true",
    reason="Set RUN_REAL_LLM_EVAL=true to run opt-in real LLM black-box evaluation.",
)

FIXTURE_PATH = Path("tests/fixtures/llm_eval_cases.yaml")
REPORT_DIR = Path("reports/llm_eval")


@pytest.mark.asyncio
async def test_real_llm_blackbox_quality_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for RUN_REAL_LLM_EVAL=true.")

    from app.dtos.chatbot import ChatbotAskRequest, ChatbotContextType
    from app.services import chatbot as chatbot_service

    monkeypatch.setattr(chatbot_service.config, "CHATBOT_USE_REAL_LLM", True)
    monkeypatch.setattr(chatbot_service.config, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    monkeypatch.setattr(chatbot_service.config, "OPENAI_TEMPERATURE", 0.0)

    cases = _load_cases()
    assert 80 <= len(cases) <= 120

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for case in cases:
        started_at = perf_counter()
        response, runtime_trace = await chatbot_service.ask_chatbot_with_runtime_trace(
            ChatbotAskRequest(message=case["question"], context_type=ChatbotContextType.MAIN),
            user_id=None,
        )
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        metadata = {
            "token_usage_extracted": True,
            "langfuse_input_redacted": True,
            "openai_failed": response.source in {"rule_engine", "rule_engine_unmatched", "fallback"},
            "runtime_trace": runtime_trace,
        }
        evaluation = evaluate_llm_case_quality(
            answer=response.answer,
            source=response.source,
            category=case["category"],
            expected=case["expected"],
            temperature=0.0,
            repeated_answers=[response.answer],
            metadata=metadata,
        )
        row = {
            "case_id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": response.answer,
            "source": response.source,
            "score": evaluation.total_score,
            "passed": evaluation.passed,
            "issues": evaluation.issues,
            "latency_ms": latency_ms,
            "token_usage": runtime_trace.get("token_usage"),
        }
        results.append(row)
        if not evaluation.passed:
            failures.append(f"{case['id']} score={evaluation.total_score} issues={evaluation.issues}")
        if case["category"] == "static_intent":
            assert response.source.startswith("static_"), (
                f"{case['id']} expected static_* source, got {response.source}"
            )

    _write_reports(results)
    assert not failures, "LLM eval failures:\n" + "\n".join(failures[:20])


def _load_cases() -> list[dict[str, Any]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _write_reports(results: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "real_llm_eval_results.json"
    csv_path = REPORT_DIR / "real_llm_eval_results.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "case_id",
        "category",
        "question",
        "answer",
        "source",
        "score",
        "passed",
        "issues",
        "latency_ms",
        "token_usage",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            csv_row = dict(row)
            csv_row["issues"] = json.dumps(row["issues"], ensure_ascii=False)
            csv_row["token_usage"] = json.dumps(row["token_usage"], ensure_ascii=False)
            writer.writerow(csv_row)
