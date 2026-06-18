#!/usr/bin/env python3
"""Measure Health Ladder API latency and print P95-friendly summaries.

This script intentionally uses only Python standard library modules so it can
run on a production bastion or local laptop without installing dependencies.
Do not print or persist authorization tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_BASE_URL = "https://healthladder.duckdns.org"
P95_THRESHOLD_MS = 3000.0


@dataclass(frozen=True)
class Endpoint:
    name: str
    method: str
    path: str
    requires_auth: bool = False


@dataclass
class RequestResult:
    endpoint: str
    method: str
    path: str
    ok: bool
    status: int | None
    latency_ms: float
    error: str | None = None


PUBLIC_ENDPOINTS = [
    Endpoint("health", "GET", "/api/v1/system/health"),
]

SMOKE_AUTH_ENDPOINTS = [
    Endpoint("dashboard_trends_week", "GET", "/api/v1/dashboard/trends?period=week", True),
    Endpoint("dashboard_trends_today", "GET", "/api/v1/dashboard/trends?period=today", True),
    Endpoint("dashboard_risk_trend_week", "GET", "/api/v1/dashboard/risk-trend?period=week", True),
    Endpoint("settings_me", "GET", "/api/v1/settings/me", True),
    Endpoint("notification_logs", "GET", "/api/v1/notifications/logs?limit=10", True),
    Endpoint("challenges_list", "GET", "/api/v1/challenges?limit=10&offset=0", True),
    Endpoint("my_challenges", "GET", "/api/v1/challenges/my?limit=10&offset=0", True),
]

FULL_AUTH_ENDPOINTS = [
    *SMOKE_AUTH_ENDPOINTS,
    Endpoint("notifications_list", "GET", "/api/v1/notifications?limit=10&offset=0", True),
    Endpoint("health_records", "GET", "/api/v1/health/records?limit=10&offset=0", True),
    Endpoint("health_latest", "GET", "/api/v1/health/records/latest", True),
    Endpoint("health_readiness", "GET", "/api/v1/health/analysis-readiness", True),
    Endpoint("exams_list", "GET", "/api/v1/exams?limit=10&offset=0", True),
    Endpoint("diets_list", "GET", "/api/v1/diets?limit=10&offset=0", True),
    Endpoint("medications_list", "GET", "/api/v1/medications?limit=10&offset=0", True),
    Endpoint("analysis_results", "GET", "/api/v1/analysis/results?limit=10&offset=0", True),
    Endpoint("dashboard_summary", "GET", "/api/v1/dashboard/summary", True),
    Endpoint("dashboard_health", "GET", "/api/v1/dashboard/health", True),
    Endpoint("dashboard_challenges", "GET", "/api/v1/dashboard/challenges", True),
    Endpoint("dashboard_diets", "GET", "/api/v1/dashboard/diets", True),
    Endpoint("dashboard_medications", "GET", "/api/v1/dashboard/medications", True),
]

AI_AUTH_ENDPOINTS = [
    Endpoint("analysis_results", "GET", "/api/v1/analysis/results?limit=10&offset=0", True),
    Endpoint("analysis_results_latest", "GET", "/api/v1/analysis/results/latest", True),
    Endpoint("health_readiness", "GET", "/api/v1/health/analysis-readiness", True),
    Endpoint("exams_list", "GET", "/api/v1/exams?limit=10&offset=0", True),
    Endpoint("diets_list", "GET", "/api/v1/diets?limit=10&offset=0", True),
    Endpoint("dashboard_health", "GET", "/api/v1/dashboard/health", True),
    Endpoint("dashboard_diets", "GET", "/api/v1/dashboard/diets", True),
    Endpoint("dashboard_trends_week", "GET", "/api/v1/dashboard/trends?period=week", True),
    Endpoint("dashboard_risk_trend_week", "GET", "/api/v1/dashboard/risk-trend?period=week", True),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure Health Ladder API P95 latency.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API origin, default: {DEFAULT_BASE_URL}")
    parser.add_argument("--count", type=int, default=30, help="Requests per endpoint, default: 30")
    parser.add_argument("--interval", type=float, default=0.2, help="Sleep seconds between requests, default: 0.2")
    parser.add_argument("--timeout", type=float, default=20.0, help="Request timeout seconds, default: 20")
    parser.add_argument("--with-auth", action="store_true", help="Include authenticated endpoints using TOKEN env var")
    parser.add_argument(
        "--profile",
        choices=("smoke", "full", "ai"),
        default="smoke",
        help="Endpoint profile to measure, default: smoke",
    )
    parser.add_argument("--json-out", help="Write JSON result to this path")
    return parser


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = int(len(sorted_values) * ratio) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def format_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}ms"


def format_request_status(result: RequestResult) -> str:
    if result.ok:
        return f"{result.status}"
    if result.status is not None:
        return f"HTTP {result.status}"
    return result.error or "ERROR"


def make_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def request_once(endpoint: Endpoint, base_url: str, token: str | None, timeout: float) -> RequestResult:
    headers = {"User-Agent": "healthladder-p95-check/1.0"}
    if endpoint.requires_auth and token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(
        make_url(base_url, endpoint.path),
        method=endpoint.method,
        headers=headers,
    )
    started = time.perf_counter()
    status: int | None = None
    error: str | None = None
    ok = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            response.read()
            ok = 200 <= status < 400
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"HTTP {exc.code}"
        try:
            exc.read()
        except Exception:
            pass
    except urllib.error.URLError as exc:
        error = exc.reason.__class__.__name__ if not isinstance(exc.reason, str) else exc.reason
    except TimeoutError:
        error = "timeout"
    latency_ms = (time.perf_counter() - started) * 1000
    return RequestResult(
        endpoint=endpoint.name,
        method=endpoint.method,
        path=endpoint.path,
        ok=ok,
        status=status,
        latency_ms=latency_ms,
        error=error,
    )


def summarize_results(endpoint: Endpoint, results: list[RequestResult]) -> dict[str, Any]:
    success_results = [result for result in results if result.ok]
    latency_source = success_results if success_results else results
    latencies = [result.latency_ms for result in latency_source]
    avg = statistics.fmean(latencies) if latencies else None
    return {
        "endpoint": endpoint.name,
        "method": endpoint.method,
        "path": endpoint.path,
        "count": len(results),
        "success_count": len(success_results),
        "error_count": len(results) - len(success_results),
        "avg_ms": avg,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "min_ms": min(latencies) if latencies else None,
        "max_ms": max(latencies) if latencies else None,
    }


def print_summary(summaries: list[dict[str, Any]]) -> None:
    print("\n================ SUMMARY ================")
    print(
        f"{'endpoint':<30} {'ok':>9} {'err':>6} {'avg':>10} {'p50':>10} {'p95':>10} {'p99':>10} {'min':>10} {'max':>10}"
    )
    for summary in summaries:
        ok_text = f"{summary['success_count']}/{summary['count']}"
        print(
            f"{summary['endpoint']:<30} {ok_text:>9} {summary['error_count']:>6} "
            f"{format_ms(summary['avg_ms']):>10} {format_ms(summary['p50_ms']):>10} "
            f"{format_ms(summary['p95_ms']):>10} {format_ms(summary['p99_ms']):>10} "
            f"{format_ms(summary['min_ms']):>10} {format_ms(summary['max_ms']):>10}"
        )

    failed = [summary["endpoint"] for summary in summaries if summary["error_count"] > 0]
    slow = [
        summary["endpoint"]
        for summary in summaries
        if summary["p95_ms"] is not None and summary["p95_ms"] > P95_THRESHOLD_MS
    ]
    print("\n판정:")
    print("- 실패 요청 없음" if not failed else f"- 실패 요청 있음: {', '.join(failed)}")
    if slow:
        print(f"- p95 3초 초과 endpoint 있음: {', '.join(slow)}")
    else:
        print("- 모든 측정 endpoint p95 3초 이하")


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def select_endpoints(profile: str, with_auth: bool) -> list[Endpoint]:
    endpoints = [*PUBLIC_ENDPOINTS]
    if not with_auth:
        return endpoints
    if profile == "full":
        endpoints.extend(FULL_AUTH_ENDPOINTS)
    elif profile == "ai":
        endpoints.extend(AI_AUTH_ENDPOINTS)
    else:
        endpoints.extend(SMOKE_AUTH_ENDPOINTS)
    return endpoints


def run(args: argparse.Namespace) -> int:
    if args.count <= 0:
        raise SystemExit("--count는 1 이상이어야 합니다.")
    if args.interval < 0:
        raise SystemExit("--interval은 0 이상이어야 합니다.")

    token = os.environ.get("TOKEN")
    if args.with_auth and not token:
        raise SystemExit("--with-auth 사용 시 TOKEN 환경변수가 필요합니다. 예: export TOKEN='JWT_ACCESS_TOKEN'")

    endpoints = select_endpoints(args.profile, bool(args.with_auth))

    all_results: list[RequestResult] = []
    summaries: list[dict[str, Any]] = []
    print(f"Profile: {args.profile}")
    print(f"Base URL: {args.base_url.rstrip('/')}")
    print(f"Count per endpoint: {args.count}")
    print(f"Authenticated endpoints: {'enabled' if args.with_auth else 'disabled'}")

    for endpoint in endpoints:
        endpoint_results: list[RequestResult] = []
        print(f"\n[{endpoint.name}] {endpoint.method} {endpoint.path}")
        for index in range(args.count):
            result = request_once(endpoint, args.base_url, token, args.timeout)
            endpoint_results.append(result)
            all_results.append(result)
            status = "OK" if result.ok else "FAIL"
            print(
                f"{index + 1:03d}/{args.count:03d} {status:<7} "
                f"{result.latency_ms:8.1f} ms  {format_request_status(result)}"
            )
            if index < args.count - 1 and args.interval > 0:
                time.sleep(args.interval)
        summaries.append(summarize_results(endpoint, endpoint_results))

    print_summary(summaries)

    if args.json_out:
        write_json(
            args.json_out,
            {
                "base_url": args.base_url.rstrip("/"),
                "profile": args.profile,
                "count_per_endpoint": args.count,
                "with_auth": bool(args.with_auth),
                "threshold_ms": P95_THRESHOLD_MS,
                "summaries": summaries,
                "results": [asdict(result) for result in all_results],
            },
        )
        print(f"\nJSON 결과 저장: {args.json_out}")
    return 0


def main() -> int:
    parser = build_parser()
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
