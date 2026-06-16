from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import asyncio
import json
import sys
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_runtime.llm.diet_recommendation_rewriter import rewrite_diet_rag_comment
from ai_runtime.llm.llm_client import get_openai_model
from app.core import config
from app.core.providers import has_openai_config
from scripts.qa import generate_diet_recommendation_qa_outputs as qa


SMOKE_CASE_IDS = ("A02", "A09", "A16", "A23", "A25", "B11", "A42", "B19", "C02", "C03")


class OpenAISmokeConfigurationError(RuntimeError):
    pass


class OpenAISmokeRewriteError(RuntimeError):
    pass


def smoke_cases() -> list[qa.DietQaCase]:
    cases_by_id = {case.case_id: case for case in qa.build_qa_cases()}
    return [cases_by_id[case_id] for case_id in SMOKE_CASE_IDS]


async def generate_smoke_outputs(
    *,
    confirm_openai_call: bool,
    rewrite_func: Callable[..., dict[str, Any] | None] = rewrite_diet_rag_comment,
) -> list[dict[str, Any]]:
    _ensure_openai_ready(confirm_openai_call=confirm_openai_call, case_count=len(SMOKE_CASE_IDS))
    outputs: list[dict[str, Any]] = []
    for case in smoke_cases():
        base_output = await qa.generate_case_output(case)
        original_response = base_output["response"]
        openai_response = deepcopy(original_response)
        rewritten_rag_comment = rewrite_func(
            rag_comment=deepcopy(original_response.get("rag_comment")),
            recommendation_payload=deepcopy(original_response),
            use_real_llm=True,
        )
        if rewritten_rag_comment is None:
            raise OpenAISmokeRewriteError(f"{case.case_id}: OpenAI rewrite returned empty response.")
        if rewritten_rag_comment.get("fallback_reason") == "llm_rewrite_failed":
            raise OpenAISmokeRewriteError(
                f"{case.case_id}: OpenAI rewrite call failed. Check OPENAI_API_KEY, model, network, and quota."
            )
        openai_response["rag_comment"] = rewritten_rag_comment
        original_checks = qa.run_response_checks(original_response)
        openai_checks = qa.run_response_checks(openai_response)
        outputs.append(
            {
                "case": _case_payload(case),
                "original_response": original_response,
                "openai_rewrite_response": openai_response,
                "checks": {
                    "passed": original_checks["passed"] and openai_checks["passed"],
                    "original": original_checks,
                    "openai_rewrite": openai_checks,
                },
                "openai": {
                    "called": True,
                    "model": get_openai_model(),
                    "rewrite_used": bool(rewritten_rag_comment.get("rewrite_used")),
                    "fallback_reason": rewritten_rag_comment.get("fallback_reason"),
                    "estimated_tokens": _estimate_tokens(original_response, openai_response),
                },
            }
        )
    return outputs


def _ensure_openai_ready(*, confirm_openai_call: bool, case_count: int) -> None:
    if not confirm_openai_call:
        raise OpenAISmokeConfigurationError(
            "OpenAI smoke test is disabled by default. Re-run with --confirm-openai-call to call OpenAI."
        )
    if case_count > 10:
        raise OpenAISmokeConfigurationError("OpenAI smoke test is limited to at most 10 cases.")
    if not has_openai_config(config):
        raise OpenAISmokeConfigurationError("OPENAI_API_KEY is not configured. Set it without printing the key.")
    model = get_openai_model()
    if not model:
        raise OpenAISmokeConfigurationError("OPENAI_MODEL is empty.")


def _case_payload(case: qa.DietQaCase) -> dict[str, Any]:
    payload = asdict(case)
    payload["analysis_types"] = [analysis_type.value for analysis_type in case.analysis_types]
    payload.pop("fake_vector_documents")
    payload.pop("rag_embedding_enabled", None)
    payload.pop("rag_embedding_provider", None)
    return payload


def _estimate_tokens(original_response: dict[str, Any], openai_response: dict[str, Any]) -> dict[str, int]:
    prompt_like_chars = len(json.dumps(original_response, ensure_ascii=False, sort_keys=True))
    output_chars = len(json.dumps(openai_response.get("rag_comment") or {}, ensure_ascii=False, sort_keys=True))
    input_tokens = max(1, prompt_like_chars // 4)
    output_tokens = max(1, output_chars // 4)
    return {
        "input_tokens_approx": input_tokens,
        "output_tokens_approx": output_tokens,
        "total_tokens_approx": input_tokens + output_tokens,
    }


def summarize_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    estimated_input = sum(item["openai"]["estimated_tokens"]["input_tokens_approx"] for item in outputs)
    estimated_output = sum(item["openai"]["estimated_tokens"]["output_tokens_approx"] for item in outputs)
    return {
        "total_cases": len(outputs),
        "case_ids": [item["case"]["case_id"] for item in outputs],
        "model": get_openai_model(),
        "openai_called": bool(outputs),
        "failed_cases": [item["case"]["case_id"] for item in outputs if not item["checks"]["passed"]],
        "estimated_tokens": {
            "input_tokens_approx": estimated_input,
            "output_tokens_approx": estimated_output,
            "total_tokens_approx": estimated_input + estimated_output,
        },
    }


def write_outputs(outputs: list[dict[str, Any]], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown_report(outputs), encoding="utf-8")


def render_markdown_report(outputs: list[dict[str, Any]]) -> str:
    summary = summarize_outputs(outputs)
    lines = [
        "# Diet Recommendation OpenAI Smoke Outputs",
        "",
        "대표 식단 추천 케이스에 대해 실제 OpenAI rewrite 경로를 확인하는 smoke 리포트입니다.",
        "API key와 토큰 값은 출력하지 않습니다.",
        "",
        f"- 총 케이스: {summary['total_cases']}",
        f"- 모델: {summary['model']}",
        f"- 추정 토큰: {json.dumps(summary['estimated_tokens'], ensure_ascii=False)}",
        f"- 자동 검사 실패 케이스: {summary['failed_cases'] or '없음'}",
        "",
    ]
    for output in outputs:
        case = output["case"]
        original = output["original_response"]
        rewritten = output["openai_rewrite_response"]
        original_rag = original.get("rag_comment") or {}
        rewritten_rag = rewritten.get("rag_comment") or {}
        lines.extend(
            [
                f"## {case['case_id']} {case['title']}",
                "",
                f"- case_id: {case['case_id']}",
                f"- 입력 요약: {case['user_context']} / {case['expected_focus']}",
                f"- disease_groups: {', '.join(case['disease_groups'])}",
                f"- diet_issue: {case['diet_issue']}",
                f"- rule/formatter 기반 원본 응답: {_response_summary(original)}",
                f"- OpenAI rewrite 응답: {_response_summary(rewritten)}",
                f"- recommended_challenges: {qa._challenge_details(rewritten.get('recommended_challenges'))}",
                f"- rag_comment 원본: {original_rag.get('summary') or '-'}",
                f"- rag_comment rewrite: {rewritten_rag.get('summary') or '-'}",
                f"- safety_notice: {rewritten.get('safety_notice')}",
                f"- 금지 표현 검사 결과: {output['checks']['openai_rewrite']['forbidden_phrases_found'] or '통과'}",
                f"- 내부값 노출 검사 결과: {output['checks']['openai_rewrite']['internal_terms_found'] or '통과'}",
                f"- 사람이 확인할 포인트: {case['expected_focus']}",
                "",
            ]
        )
    return "\n".join(lines)


def _response_summary(response: dict[str, Any]) -> str:
    rag_comment = response.get("rag_comment") or {}
    return " / ".join(
        [
            f"nutrition={qa._messages(response.get('nutrition_findings'))}",
            f"disease={qa._messages(response.get('disease_context'))}",
            f"rag={rag_comment.get('summary') or '-'}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate real-OpenAI diet recommendation smoke samples.")
    parser.add_argument("--confirm-openai-call", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/qa/diet_recommendation_openai_smoke_outputs.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/qa/diet_recommendation_openai_smoke_outputs.md"),
    )
    parser.add_argument("--summary-json", action="store_true")
    return parser.parse_args()


async def _main_async() -> None:
    args = parse_args()
    try:
        outputs = await generate_smoke_outputs(confirm_openai_call=args.confirm_openai_call)
    except (OpenAISmokeConfigurationError, OpenAISmokeRewriteError) as exc:
        print(f"OpenAI smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    write_outputs(outputs, args.output_json, args.output_md)
    summary = summarize_outputs(outputs) | {
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }
    if args.summary_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Generated {summary['total_cases']} OpenAI smoke outputs.")
        print(f"JSON: {args.output_json}")
        print(f"Markdown: {args.output_md}")
        print(f"Failed checks: {summary['failed_cases'] or 'none'}")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
