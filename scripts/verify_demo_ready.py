"""Check local demo readiness without calling paid external providers.

This script is a broad pre-demo checklist. It does not run the authenticated
PRECISION API flow; use scripts/verify_precision_analysis_api.py for that E2E
check after Docker compose and seed data are ready.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

API_BASE_URL = os.getenv("DEMO_API_BASE_URL", "http://localhost:8000/api/v1").rstrip("/")

CATBOOST_ARTIFACTS = {
    "dm": "dm_catboost_final",
    "htn": "htn_catboost_final",
    "dl": "dl_catboost_final",
}

REQUIRED_ENV_KEYS = (
    "SECRET_KEY",
    "DB_HOST",
    "DB_PORT",
    "DB_USER",
    "DB_PASSWORD",
    "DB_NAME",
    "REDIS_HOST",
    "REDIS_PORT",
)

OPTIONAL_PROVIDER_ENV_KEYS = (
    "OPENAI_API_KEY",
    "EMAIL_ENABLED",
    "EMAIL_VERIFICATION_DEBUG",
    "PASSWORD_RESET_DEBUG",
    "PHONE_VERIFICATION_DEBUG",
    "TWILIO_ENABLED",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_VERIFY_SERVICE_SID",
    "SMTP_HOST",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
)


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def main() -> int:
    results = [
        _check_fastapi_import(),
        _check_ml_import(),
        _check_catboost_artifacts(),
        _check_nutrition_assets(),
        _check_disease_score_load(),
        _check_environment_keys(),
        _check_optional_provider_keys(),
        _check_auth_delivery_policy(),
        _check_deferred_provider_policy(),
        _check_system_health_if_running(),
    ]

    print("Demo readiness check")
    print("====================")
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")

    failures = [result for result in results if result.status == "FAIL"]
    if failures:
        print("\nFailed checks:")
        for failure in failures:
            print(f"- {failure.name}: {failure.detail}")
        return 1

    print("\nDemo readiness check finished without critical failures.")
    print("For authenticated PRECISION API E2E, run: uv run python scripts/verify_precision_analysis_api.py")
    return 0


def _check_fastapi_import() -> CheckResult:
    try:
        from app.main import app

        path_count = len(app.openapi().get("paths", {}))
    except Exception as exc:  # noqa: BLE001 - readiness check should report import errors clearly.
        return CheckResult("FastAPI import", "FAIL", f"{type(exc).__name__}: {exc}")
    return CheckResult("FastAPI import", "OK", f"title={app.title}, openapi_paths={path_count}")


def _check_ml_import() -> CheckResult:
    try:
        from catboost import CatBoostClassifier

        from ai_worker.ml.inference.disease_risk_service import predict_chronic_disease_risks
    except Exception as exc:  # noqa: BLE001 - readiness check should report import errors clearly.
        return CheckResult("ML import", "FAIL", f"{type(exc).__name__}: {exc}")

    _ = CatBoostClassifier
    _ = predict_chronic_disease_risks
    return CheckResult("ML import", "OK", "catboost and disease risk service import succeeded")


def _check_catboost_artifacts() -> CheckResult:
    artifact_root = REPO_ROOT / "ai_worker" / "ml" / "artifacts"
    missing: list[str] = []
    details: list[str] = []

    for disease_key in CATBOOST_ARTIFACTS:
        model_dir = artifact_root / disease_key / "catboost"
        model_files = sorted(model_dir.glob("model_fold*.cbm"))
        required_files = [
            model_dir / "feature_columns.json",
            model_dir / "threshold.json",
            model_dir / "metrics.json",
            model_dir / "model_params.json",
            model_dir / "experiment_config.json",
        ]
        missing.extend(str(path.relative_to(REPO_ROOT)) for path in required_files if not path.exists())
        if len(model_files) != 5:
            missing.append(f"{model_dir.relative_to(REPO_ROOT)}/model_fold*.cbm expected=5 actual={len(model_files)}")
        details.append(f"{disease_key}: models={len(model_files)}")

    if missing:
        return CheckResult("CatBoost artifacts", "FAIL", "missing " + ", ".join(missing[:5]))
    return CheckResult("CatBoost artifacts", "OK", "; ".join(details))


def _check_nutrition_assets() -> CheckResult:
    csv_path = REPO_ROOT / "ai_worker" / "cv" / "food" / "nutrition" / "data" / "food_disease_scores.csv"
    rules_path = REPO_ROOT / "ai_worker" / "cv" / "food" / "nutrition" / "rules" / "disease_score_rules.json"
    missing = [path for path in (csv_path, rules_path) if not path.exists()]
    if missing:
        return CheckResult(
            "Nutrition assets",
            "FAIL",
            "missing " + ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing),
        )
    return CheckResult("Nutrition assets", "OK", "food_disease_scores.csv and disease_score_rules.json exist")


def _check_disease_score_load() -> CheckResult:
    try:
        from ai_worker.cv.food.nutrition.scoring.disease_food_scorer import DiseaseFoodScorer
        from ai_worker.cv.food.nutrition.scoring.schemas import DISEASE_CODES

        records = DiseaseFoodScorer().load_runtime_scores()
    except Exception as exc:  # noqa: BLE001 - readiness check should report load errors clearly.
        return CheckResult("Disease score load", "FAIL", f"{type(exc).__name__}: {exc}")

    expected_codes = {"DM", "HTN", "DL", "OBE", "ANEM"}
    if set(DISEASE_CODES) != expected_codes:
        return CheckResult("Disease score load", "FAIL", f"unexpected disease codes={DISEASE_CODES}")
    if not records:
        return CheckResult("Disease score load", "FAIL", "no runtime food score records loaded")
    return CheckResult("Disease score load", "OK", f"records={len(records)}, disease_codes={','.join(DISEASE_CODES)}")


def _check_environment_keys() -> CheckResult:
    try:
        from app.core import config
    except Exception:
        missing = [key for key in REQUIRED_ENV_KEYS if not _has_value(os.getenv(key))]
    else:
        missing = [key for key in REQUIRED_ENV_KEYS if not _has_value(str(getattr(config, key, "")))]

    if missing:
        return CheckResult(
            "Required environment",
            "WARN",
            "missing or empty keys: " + ", ".join(missing),
        )
    return CheckResult("Required environment", "OK", "required local runtime keys are present")


def _check_optional_provider_keys() -> CheckResult:
    missing = [key for key in OPTIONAL_PROVIDER_ENV_KEYS if not _has_value(os.getenv(key))]
    if missing:
        return CheckResult(
            "External provider environment",
            "WARN",
            "not configured: " + ", ".join(missing) + " (no external API call is made)",
        )
    return CheckResult("External provider environment", "OK", "optional provider keys are configured")


def _check_auth_delivery_policy() -> CheckResult:
    try:
        from app.core import config
        from app.services.email_service import EmailService
    except Exception as exc:  # noqa: BLE001 - readiness check should report import/config errors clearly.
        return CheckResult("Auth delivery policy", "WARN", f"{type(exc).__name__}: {exc}")

    debug_flags = {
        "EMAIL_VERIFICATION_DEBUG": config.EMAIL_VERIFICATION_DEBUG,
        "PASSWORD_RESET_DEBUG": config.PASSWORD_RESET_DEBUG,
        "PHONE_VERIFICATION_DEBUG": config.PHONE_VERIFICATION_DEBUG,
    }
    enabled_debug_flags = [key for key, enabled in debug_flags.items() if enabled]
    if config.is_production and enabled_debug_flags:
        return CheckResult(
            "Auth delivery policy",
            "FAIL",
            "production must not enable debug auth responses: " + ", ".join(enabled_debug_flags),
        )

    email_status = EmailService().status()
    email_detail = (
        "Brevo/SMTP live email configured" if email_status == "configured" else f"SMTP live email status={email_status}"
    )
    if config.TWILIO_ENABLED and config.twilio_verify_status == "configured":
        phone_detail = "Twilio Verify configured; Trial Korean SMS may require verified recipient or paid account"
    elif config.PHONE_VERIFICATION_DEBUG and not config.is_production:
        phone_detail = "local/demo phone debug fallback enabled"
    else:
        phone_detail = "phone verification needs Twilio config or local/demo debug fallback"
    status = (
        "OK" if email_status == "configured" or config.PHONE_VERIFICATION_DEBUG or config.TWILIO_ENABLED else "WARN"
    )
    return CheckResult("Auth delivery policy", status, f"{email_detail}; {phone_detail}")


def _check_deferred_provider_policy() -> CheckResult:
    return CheckResult(
        "Deferred provider policy",
        "OK",
        "Clova OCR is excluded from official demo readiness; GPT Vision fallback remains off unless explicitly enabled",
    )


def _check_system_health_if_running() -> CheckResult:
    try:
        body = _request_json(f"{API_BASE_URL}/system/health")
    except urllib.error.URLError:
        return CheckResult(
            "Docker/API health",
            "WARN",
            f"API is not reachable at {API_BASE_URL}; start docker compose if API smoke is needed",
        )
    except Exception as exc:  # noqa: BLE001 - readiness check should keep going with readable output.
        return CheckResult("Docker/API health", "WARN", f"{type(exc).__name__}: {exc}")

    status = body.get("status", "unknown") if isinstance(body, dict) else "unknown"
    return CheckResult("Docker/API health", "OK" if status == "ok" else "WARN", f"status={status}")


def _request_json(url: str) -> Any:
    request = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _has_value(value: str | None) -> bool:
    if value is None:
        return False
    stripped = value.strip()
    return bool(stripped) and not stripped.lower().startswith(("your-", "change-me", "replace-me"))


if __name__ == "__main__":
    raise SystemExit(main())
