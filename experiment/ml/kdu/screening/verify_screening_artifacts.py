"""Verify screening_catboost staging artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from screening_config import ARTIFACT_MODEL_NAME, DISEASE_RUNTIME_KEYS, MAIN_SCREENING_TARGETS
from screening_feature_set import validate_feature_columns


REQUIRED_FILES = (
    "model_fold1.cbm",
    "model_fold2.cbm",
    "model_fold3.cbm",
    "model_fold4.cbm",
    "model_fold5.cbm",
    "feature_columns.json",
    "threshold.json",
    "metrics.json",
    "model_params.json",
    "experiment_config.json",
)

METRIC_REQUIRED_KEYS = (
    "disease",
    "label_policy",
    "feature_set",
    "data_version",
    "rows",
    "positive_rate",
    "service_policy_allowed",
    "threshold_metrics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify KDU screening_catboost staging artifacts.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("experiment/ml/kdu/screening/artifacts"),
        help="Artifact root. Defaults to experiment/ml/kdu/screening/artifacts.",
    )
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    raise RuntimeError("Cannot find repo root. Run this script inside AH_03_03.")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_target(artifact_root: Path, target: Any) -> dict[str, Any]:
    runtime_key = DISEASE_RUNTIME_KEYS[target.disease]
    artifact_dir = artifact_root / runtime_key / ARTIFACT_MODEL_NAME
    missing_files = [file_name for file_name in REQUIRED_FILES if not (artifact_dir / file_name).exists()]
    result: dict[str, Any] = {
        "disease": target.disease,
        "runtime_key": runtime_key,
        "label_policy": target.label_policy,
        "data_version": target.data_version,
        "artifact_dir": str(artifact_dir),
        "missing_files": missing_files,
        "required_files_present": not missing_files,
    }
    if missing_files:
        result["status"] = "fail"
        return result

    feature_columns = read_json(artifact_dir / "feature_columns.json")
    feature_match, feature_diff = validate_feature_columns(feature_columns)
    threshold = read_json(artifact_dir / "threshold.json")
    metrics = read_json(artifact_dir / "metrics.json")
    experiment_config = read_json(artifact_dir / "experiment_config.json")

    fold_files = sorted(artifact_dir.glob("model_fold*.cbm"))
    metric_missing = [key for key in METRIC_REQUIRED_KEYS if key not in metrics]
    threshold_value = threshold.get("threshold")
    model_name = experiment_config.get("model_name")

    checks = {
        "feature_columns_match": feature_match,
        "feature_diff": feature_diff,
        "model_name_is_screening_catboost": model_name == ARTIFACT_MODEL_NAME,
        "threshold_exists": threshold_value is not None,
        "metrics_required_keys_present": not metric_missing,
        "metrics_missing_keys": metric_missing,
        "fold_model_count": len(fold_files),
        "fold_model_count_is_5": len(fold_files) == 5,
        "service_policy_allowed": experiment_config.get("service_policy_allowed"),
    }
    result.update(checks)
    result["status"] = (
        "pass"
        if all(
            [
                result["required_files_present"],
                checks["feature_columns_match"],
                checks["model_name_is_screening_catboost"],
                checks["threshold_exists"],
                checks["metrics_required_keys_present"],
                checks["fold_model_count_is_5"],
            ]
        )
        else "fail"
    )
    return result


def write_reports(repo_root: Path, results: list[dict[str, Any]]) -> None:
    report_json = repo_root / "experiment" / "ml" / "kdu" / "screening" / "artifact_verification_report.json"
    report_md = repo_root / "experiment" / "ml" / "kdu" / "screening" / "artifact_verification_report.md"
    payload = {
        "model_name": ARTIFACT_MODEL_NAME,
        "all_passed": all(row["status"] == "pass" for row in results),
        "results": results,
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Screening Artifact Verification Report",
        "",
        f"- Model name: `{ARTIFACT_MODEL_NAME}`",
        f"- All passed: `{payload['all_passed']}`",
        "",
        "| disease | artifact | required files | feature match | folds | threshold | model_name | status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            "| {disease} | `{artifact}` | {required} | {feature} | {folds} | {threshold} | {model_name} | {status} |".format(
                disease=row["disease"],
                artifact=row["artifact_dir"],
                required=row["required_files_present"],
                feature=row.get("feature_columns_match"),
                folds=row.get("fold_model_count"),
                threshold=row.get("threshold_exists"),
                model_name=row.get("model_name_is_screening_catboost"),
                status=row["status"],
            )
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd().resolve())
    artifact_root = args.artifact_root
    if not artifact_root.is_absolute():
        artifact_root = repo_root / artifact_root

    results = [verify_target(artifact_root, target) for target in MAIN_SCREENING_TARGETS]
    write_reports(repo_root, results)
    for row in results:
        print(f"[VERIFY] {row['disease']} status={row['status']} artifact={row['artifact_dir']}", flush=True)
    if not all(row["status"] == "pass" for row in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
