"""Smoke test the Docker/API precision analysis path.

Environment variables:
- VERIFY_API_BASE_URL: defaults to http://localhost:8000/api/v1
- VERIFY_EMAIL: defaults to demo@example.com
- VERIFY_PASSWORD: defaults to Demo1234!
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from argparse import ArgumentParser
from typing import Any

API_BASE_URL = os.getenv("VERIFY_API_BASE_URL", "http://localhost:8000/api/v1").rstrip("/")
EMAIL = os.getenv("VERIFY_EMAIL", "demo@example.com")
PASSWORD = os.getenv("VERIFY_PASSWORD", "Demo1234!")
JOB_POLL_INTERVAL_SECONDS = float(os.getenv("VERIFY_JOB_POLL_INTERVAL_SECONDS", "1"))
JOB_TIMEOUT_SECONDS = float(os.getenv("VERIFY_JOB_TIMEOUT_SECONDS", "60"))

EXPECTED_CATBOOST = {
    "DIABETES": "dm_catboost_final",
    "HYPERTENSION": "htn_catboost_final",
    "DYSLIPIDEMIA": "dl_catboost_final",
}


def main() -> int:
    args = _parse_args()
    if args.warmup_ml:
        _warmup_ml_models()

    token = _login()
    readiness = _request_json("GET", "/health/analysis-readiness", token=token)
    if not readiness.get("precision_ready"):
        _fail(f"precision_ready=false: missing={readiness.get('missing_precision_fields')}")

    health_record_id = readiness.get("latest_health_record_id")
    if not health_record_id:
        _fail("latest_health_record_id가 없습니다.")

    job = _request_json(
        "POST",
        "/analysis/run-async",
        token=token,
        payload={"health_record_id": health_record_id, "mode": "PRECISION"},
    )
    job_id = job.get("id") if isinstance(job, dict) else None
    if not job_id:
        _fail("/analysis/run-async 응답에 job id가 없습니다.")

    completed_job = _wait_for_job_success(int(job_id), token)
    result_payload = completed_job.get("result_payload") or {}
    result_ids = result_payload.get("analysis_result_ids") if isinstance(result_payload, dict) else None
    if not isinstance(result_ids, list) or not result_ids:
        _fail("analysis.run job result_payload에 analysis_result_ids가 없습니다.")

    details = [_request_json("GET", f"/analysis/results/{result_id}", token=token) for result_id in result_ids]
    _assert_precision_results(details)

    print("PRECISION analysis smoke OK")
    for detail in sorted(details, key=lambda item: item["analysis_type"]):
        print(
            "- {analysis_type}: mode={analysis_mode}, model={model_name}, version={model_version}, score={risk_score}".format(
                **detail
            )
        )
    return 0


def _wait_for_job_success(job_id: int, token: str) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    last_job: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        job = _request_json("GET", f"/jobs/{job_id}", token=token)
        if not isinstance(job, dict):
            _fail(f"/jobs/{job_id} 응답이 올바르지 않습니다.")
        last_job = job
        status = job.get("status")
        if status == "SUCCESS":
            return job
        if status in {"FAILED", "CANCELED"}:
            _fail(f"analysis.run job failed: status={status}, error={job.get('error_message')}")
        time.sleep(JOB_POLL_INTERVAL_SECONDS)
    _fail(f"analysis.run job timeout: last={last_job}")


def _parse_args():
    parser = ArgumentParser(description="Verify PRECISION analysis API path.")
    parser.add_argument(
        "--warmup-ml",
        action="store_true",
        help="Load CatBoost artifacts before calling the API. No external API call is made.",
    )
    return parser.parse_args()


def _warmup_ml_models() -> None:
    from ai_runtime.ml.inference.disease_risk_service import warmup_chronic_disease_models

    results = warmup_chronic_disease_models()
    failed = [disease for disease, result in results.items() if result.get("status") == "failed"]
    for disease, result in results.items():
        print(
            "warmup {disease}: status={status}, models={models}, features={features}".format(
                disease=disease,
                status=result.get("status"),
                models=result.get("model_count", "-"),
                features=result.get("feature_count", "-"),
            )
        )
    if failed:
        _fail(f"ML warmup failed: {failed}")


def _login() -> str:
    response = _request_json("POST", "/auth/login", payload={"email": EMAIL, "password": PASSWORD})
    token = response.get("access_token")
    if not token:
        _fail("로그인 응답에 access_token이 없습니다.")
    return str(token)


def _assert_precision_results(details: list[dict[str, Any]]) -> None:
    by_type = {str(detail.get("analysis_type")): detail for detail in details}
    missing = [analysis_type for analysis_type in EXPECTED_CATBOOST if analysis_type not in by_type]
    if missing:
        _fail(f"CatBoost 검증 대상 분석 결과가 없습니다: {missing}")

    for analysis_type, expected_version in EXPECTED_CATBOOST.items():
        detail = by_type[analysis_type]
        if detail.get("analysis_mode") != "PRECISION":
            _fail(f"{analysis_type} analysis_mode가 PRECISION이 아닙니다: {detail.get('analysis_mode')}")
        if detail.get("model_name") != "catboost":
            _fail(f"{analysis_type} model_name이 catboost가 아닙니다: {detail.get('model_name')}")
        if detail.get("model_version") != expected_version:
            _fail(f"{analysis_type} model_version 불일치: {detail.get('model_version')}")
        if detail.get("risk_score") is None:
            _fail(f"{analysis_type} risk_score가 없습니다.")

    obesity = by_type.get("OBESITY")
    if obesity and obesity.get("model_name") != "rule_based":
        _fail(f"OBESITY는 현재 rule_based가 기대됩니다: {obesity.get('model_name')}")


def _request_json(
    method: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        _fail(f"{method} {path} failed: HTTP {exc.code} {error_body}")
    except urllib.error.URLError as exc:
        _fail(f"{method} {path} failed: {exc.reason}")

    return json.loads(body) if body else None


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
