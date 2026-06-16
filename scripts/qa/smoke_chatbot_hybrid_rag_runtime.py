from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

QUESTIONS = [
    "고혈압이 있을 때 식단에서 뭘 조심해야 해?",
    "당뇨가 걱정될 때 탄수화물은 어떻게 관리하면 좋아?",
    "건강검진 결과에서 혈압과 혈당이 같이 높게 나오면 생활습관은 뭘 먼저 봐야 해?",
    "이상지질혈증이면 기름진 음식은 어떻게 조절해야 해?",
    "신장 관련 수치가 걱정될 때 식단은 어떻게 봐야 해?",
]
FORBIDDEN_ANSWER_TERMS = [
    '"answer":',
    '"intent":',
    '"source":',
    '"is_safe":',
    '"caution_message":',
    "```json",
    "```",
]
INTERNAL_TERMS = [
    "chunk_key",
    "score",
    "embedding",
    "pgvector",
    "text-embedding",
    "vector retriever",
    "similarity",
]


class SmokeConfigurationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small authenticated chatbot hybrid RAG runtime smoke test.")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--token-env", default="CHATBOT_SMOKE_TOKEN")
    parser.add_argument("--confirm-openai-call", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("reports/qa/chatbot_hybrid_rag_smoke_outputs.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/qa/chatbot_hybrid_rag_smoke_outputs.md"))
    parser.add_argument("--limit", type=int, default=len(QUESTIONS))
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> str:
    if not args.confirm_openai_call:
        raise SmokeConfigurationError("Refusing to run real OpenAI smoke without --confirm-openai-call.")
    token = os.environ.get(args.token_env)
    if not token:
        raise SmokeConfigurationError(f"{args.token_env} is not set. Export a valid JWT before running smoke.")
    if args.limit < 1 or args.limit > len(QUESTIONS):
        raise SmokeConfigurationError(f"--limit must be between 1 and {len(QUESTIONS)}.")
    return token


def run_smoke(
    *,
    base_url: str,
    token: str,
    limit: int,
    timeout: float,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    records = []
    for index, question in enumerate(QUESTIONS[:limit], start=1):
        records.append(
            call_chatbot(base_url=base_url, token=token, question=question, case_id=f"C{index:02d}", timeout=timeout)
        )

    return {
        "generated_at": started_at,
        "base_url": base_url.rstrip("/"),
        "question_count": len(records),
        "db_write_performed": False,
        "token_printed": False,
        "api_key_printed": False,
        "records": records,
        "summary": summarize_records(records),
    }


def call_chatbot(*, base_url: str, token: str, question: str, case_id: str, timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/chatbot/ask"
    payload = {"message": question, "context_type": "MAIN"}
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Chatbot-Smoke-Trace": "true",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            response_json = json.loads(body)
            trace = parse_trace_header(response.headers.get("X-Chatbot-Rag-Trace"))
            status_code = response.status
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chatbot smoke HTTP {exc.code}: {body[:300]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Chatbot smoke request failed: {exc.reason}") from exc

    answer = str(response_json.get("answer") or "")
    checks = check_answer(answer)
    return {
        "case_id": case_id,
        "question": question,
        "status_code": status_code,
        "source": response_json.get("source"),
        "answer": answer,
        "answer_preview": preview(answer),
        "recommended_actions": response_json.get("recommended_actions") or [],
        "safety_notice": response_json.get("safety_notice"),
        "rag_trace": trace,
        "checks": checks,
        "passed": status_code == 200 and all(checks.values()),
    }


def parse_trace_header(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"parse_error": True}
    return parsed if isinstance(parsed, dict) else {}


def check_answer(answer: str) -> dict[str, bool]:
    return {
        "no_json_code_block": not any(term in answer for term in FORBIDDEN_ANSWER_TERMS),
        "no_internal_terms": not any(term in answer for term in INTERNAL_TERMS),
        "has_natural_language": bool(answer.strip()) and "진단이 아니" in answer and "의료진 상담" in answer,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "passed_count": sum(1 for record in records if record["passed"]),
        "failed_count": sum(1 for record in records if not record["passed"]),
        "json_code_block_failures": [
            record["case_id"] for record in records if not record["checks"]["no_json_code_block"]
        ],
        "internal_term_failures": [
            record["case_id"] for record in records if not record["checks"]["no_internal_terms"]
        ],
        "trace_missing_cases": [record["case_id"] for record in records if not record.get("rag_trace")],
    }


def preview(value: str, limit: int = 180) -> str:
    compact = " ".join(value.split())
    return compact[:limit]


def write_outputs(result: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Chatbot Hybrid RAG Runtime Smoke",
        "",
        f"- generated_at: {result['generated_at']}",
        f"- question_count: {result['question_count']}",
        f"- db_write_performed: {result['db_write_performed']}",
        f"- passed_count: {result['summary']['passed_count']}",
        f"- failed_count: {result['summary']['failed_count']}",
        "",
    ]
    for record in result["records"]:
        trace = record.get("rag_trace") or {}
        lines.extend(
            [
                f"## {record['case_id']}",
                "",
                f"- question: {record['question']}",
                f"- status_code: {record['status_code']}",
                f"- source: {record['source']}",
                f"- rag_strategy: {trace.get('rag_strategy')}",
                f"- keyword_returned_count: {trace.get('keyword_returned_count')}",
                f"- vector_returned_count: {trace.get('vector_returned_count')}",
                f"- merged_count: {trace.get('merged_count')}",
                f"- final_count: {trace.get('final_count')}",
                f"- fallback_used: {trace.get('fallback_used')}",
                f"- fallback_reason: {trace.get('fallback_reason')}",
                f"- no_json_code_block: {record['checks']['no_json_code_block']}",
                f"- no_internal_terms: {record['checks']['no_internal_terms']}",
                "",
                "answer_preview:",
                "",
                record["answer_preview"],
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        token = validate_args(args)
        result = run_smoke(base_url=args.base_url, token=token, limit=args.limit, timeout=args.timeout)
        write_outputs(result, args.output_json, args.output_md)
    except SmokeConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0 if result["summary"]["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
