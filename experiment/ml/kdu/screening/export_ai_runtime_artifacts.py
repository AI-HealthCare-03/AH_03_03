"""Export KDU screening CatBoost models in ai_runtime artifact format.

The default output is a staging directory under experiment/ml/kdu/screening/artifacts.
It intentionally does not overwrite ai_runtime/ml/artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from screening_config import (
    ARTIFACT_MODEL_NAME,
    DISEASE_RUNTIME_KEYS,
    FEATURE_SET,
    MAIN_SCREENING_TARGETS,
    RANDOM_STATE,
    SERVICE_POLICY_ALLOWED,
)
from screening_feature_set import FORBIDDEN_LEAKAGE_COLUMNS, X1_SERVICE_RUNTIME_FEATURES, validate_feature_columns


REQUIRED_ARTIFACT_FILES = (
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


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    raise RuntimeError("Cannot find repo root. Run this script inside AH_03_03.")


def add_src_path(repo_root: Path) -> None:
    src_dir = repo_root / "experiment" / "ml" / "kdu" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export KDU screening models as ai_runtime staging artifacts.")
    parser.add_argument(
        "--target",
        choices=["all", "hypertension", "diabetes", "dyslipidemia"],
        default="all",
        help="Disease target to export.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiment/ml/kdu/screening/artifacts"),
        help="Staging artifact root. Defaults to experiment/ml/kdu/screening/artifacts.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of StratifiedKFold folds.")
    parser.add_argument("--dry-run", action="store_true", help="Print export plan without model training.")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=False):
        if right == 1.0:
            mask = (proba >= left) & (proba <= right)
        else:
            mask = (proba >= left) & (proba < right)
        if not mask.any():
            continue
        confidence = float(proba[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (mask.sum() / total) * abs(confidence - accuracy)
    return float(ece)


def evaluate_binary(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(y_true, pred, labels=[0, 1]).ravel()]
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, proba)),
        "ece": expected_calibration_error(y_true, proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_gray_zone(y_true: np.ndarray, proba: np.ndarray, low: float, high: float) -> dict[str, Any]:
    low_mask = proba < low
    high_mask = proba >= high
    gray_mask = ~(low_mask | high_mask)
    positive = y_true == 1
    negative = y_true == 0
    return {
        "low_threshold": float(low),
        "high_threshold": float(high),
        "low_rate": float(low_mask.mean()),
        "gray_rate": float(gray_mask.mean()),
        "high_rate": float(high_mask.mean()),
        "low_npv": safe_div(float((low_mask & negative).sum()), float(low_mask.sum())),
        "high_precision": safe_div(float((high_mask & positive).sum()), float(high_mask.sum())),
        "high_recall": safe_div(float((high_mask & positive).sum()), float(positive.sum())),
        "gray_positive_rate": safe_div(float((gray_mask & positive).sum()), float(gray_mask.sum())),
    }


def model_params(seed: int) -> dict[str, Any]:
    return {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "random_seed": seed,
        "auto_class_weights": "Balanced",
        "verbose": False,
        "allow_writing_files": False,
    }


def load_dataset(repo_root: Path, target: Any) -> pd.DataFrame:
    path = repo_root / "experiment" / "ml" / "kdu" / "data" / "processed" / target.data_version / target.file_name
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run: uv run python experiment/ml/kdu/screening/build_screening_datasets.py --main-only"
        )
    return pd.read_csv(path)


def select_features(frame: pd.DataFrame, disease: str) -> list[str]:
    missing = [column for column in X1_SERVICE_RUNTIME_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"{disease} {FEATURE_SET} missing requested features: {missing}")
    leaked = [column for column in FORBIDDEN_LEAKAGE_COLUMNS if column in X1_SERVICE_RUNTIME_FEATURES]
    if leaked:
        raise ValueError(f"{disease} {FEATURE_SET} includes forbidden leakage features: {leaked}")
    return list(X1_SERVICE_RUNTIME_FEATURES)


def verify_exported_artifact(artifact_dir: Path, feature_columns: list[str]) -> dict[str, Any]:
    missing_files = [file_name for file_name in REQUIRED_ARTIFACT_FILES if not (artifact_dir / file_name).exists()]
    feature_match, feature_diff = validate_feature_columns(feature_columns)
    if missing_files or not feature_match:
        raise RuntimeError(
            f"Artifact verification failed for {artifact_dir}: "
            f"missing_files={missing_files}, feature_diff={feature_diff}"
        )
    return {
        "artifact_dir": str(artifact_dir),
        "required_files_present": True,
        "feature_columns_match_final_definition": True,
    }


def export_target(repo_root: Path, output_root: Path, target: Any, folds: int, dry_run: bool) -> dict[str, Any]:
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import StratifiedKFold

    frame = load_dataset(repo_root, target)
    frame = frame.dropna(subset=["target"]).reset_index(drop=True)
    feature_columns = select_features(frame, target.disease)
    x = frame[feature_columns].copy()
    y = frame["target"].astype(int).reset_index(drop=True)
    runtime_key = DISEASE_RUNTIME_KEYS[target.disease]
    artifact_dir = output_root / runtime_key / ARTIFACT_MODEL_NAME

    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    print(
        f"[EXPORT] {target.disease}/{target.label_policy} rows={len(frame)} "
        f"positive_rate={positive_count / len(frame):.4f} features={len(feature_columns)} -> {artifact_dir}",
        flush=True,
    )
    if dry_run:
        return {
            "disease": target.disease,
            "label_policy": target.label_policy,
            "artifact_dir": str(artifact_dir),
            "rows": int(len(frame)),
            "positive_rate": positive_count / len(frame),
            "status": "dry_run",
        }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    oof_proba = np.zeros(len(y), dtype=float)
    oof_row_index = np.arange(len(y), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    importance_accumulator: list[np.ndarray] = []
    params = model_params(RANDOM_STATE)
    start_time = time.perf_counter()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_index, valid_index) in enumerate(skf.split(x, y), start=1):
        print(
            f"[EXPORT] {target.disease} fold {fold}/{folds} "
            f"train={len(train_index)} valid={len(valid_index)}",
            flush=True,
        )
        x_train = x.iloc[train_index].reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        x_valid = x.iloc[valid_index].reset_index(drop=True)
        y_valid = y.iloc[valid_index].reset_index(drop=True)
        model = CatBoostClassifier(**params)
        model.fit(Pool(x_train, y_train), eval_set=Pool(x_valid, y_valid))
        valid_proba = model.predict_proba(x_valid)[:, 1]
        oof_proba[valid_index] = valid_proba
        model.save_model(str(artifact_dir / f"model_fold{fold}.cbm"))
        fold_metric = evaluate_binary(y_valid.to_numpy(dtype=int), valid_proba, target.recommended_threshold or 0.5)
        fold_rows.append({"fold": fold, **fold_metric})
        importance_accumulator.append(np.asarray(model.get_feature_importance(), dtype=float))

    duration_sec = time.perf_counter() - start_time
    y_array = y.to_numpy(dtype=int)
    threshold = float(target.recommended_threshold or 0.5)
    oof_metrics = evaluate_binary(y_array, oof_proba, threshold)
    threshold_0_5_metrics = evaluate_binary(y_array, oof_proba, 0.5)
    gray_zone = None
    if target.recommended_gray_zone:
        gray_zone = evaluate_gray_zone(y_array, oof_proba, *target.recommended_gray_zone)

    feature_importance = []
    if importance_accumulator:
        mean_importance = np.vstack(importance_accumulator).mean(axis=0)
        feature_importance = sorted(
            [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(feature_columns, mean_importance, strict=False)
            ],
            key=lambda row: row["importance"],
            reverse=True,
        )

    threshold_payload = {
        "strategy": "diagnostic_screening_threshold",
        "threshold": threshold,
        "source": "KDU OOF diagnostic candidate; future-period validation required before service use.",
        "service_policy_allowed": SERVICE_POLICY_ALLOWED,
        "gray_zone": {
            "low_threshold": target.recommended_gray_zone[0],
            "high_threshold": target.recommended_gray_zone[1],
        }
        if target.recommended_gray_zone
        else None,
        "label_policy": target.label_policy,
        "user_probability_display_allowed": False,
    }
    metrics_payload = {
        "disease": target.disease,
        "label_policy": target.label_policy,
        "feature_set": FEATURE_SET,
        "data_version": target.data_version,
        "rows": int(len(frame)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": float(positive_count / len(frame)),
        "service_policy_allowed": SERVICE_POLICY_ALLOWED,
        "threshold_metrics": oof_metrics,
        "threshold_0_5_metrics": threshold_0_5_metrics,
        "gray_zone_metrics": gray_zone,
        "folds": fold_rows,
        "feature_importance": feature_importance,
        "duration_sec": round(duration_sec, 4),
    }
    experiment_config = {
        "experiment_id": f"kdu_{runtime_key}_{FEATURE_SET}_{target.label_policy}_{ARTIFACT_MODEL_NAME}",
        "disease": target.disease,
        "runtime_disease_key": runtime_key,
        "label_policy": target.label_policy,
        "role": target.role,
        "feature_set": FEATURE_SET,
        "data_version": target.data_version,
        "target_column": "target",
        "source_years": "HN13-HN24",
        "model_name": ARTIFACT_MODEL_NAME,
        "artifact_format": "ai_runtime_catboost_fold_ensemble",
        "service_policy_allowed": SERVICE_POLICY_ALLOWED,
        "diagnostic_only": True,
        "user_probability_display_allowed": False,
        "medical_safety_note": "Screening/check-needed model only. Do not use confirmed diagnosis wording.",
    }

    write_json(artifact_dir / "feature_columns.json", feature_columns)
    write_json(artifact_dir / "threshold.json", threshold_payload)
    write_json(artifact_dir / "metrics.json", metrics_payload)
    write_json(artifact_dir / "model_params.json", params)
    write_json(artifact_dir / "experiment_config.json", experiment_config)
    np.save(artifact_dir / "oof_pred_proba.npy", oof_proba)
    np.save(artifact_dir / "oof_y_true.npy", y_array)
    np.save(artifact_dir / "oof_row_index.npy", oof_row_index)
    pd.DataFrame(fold_rows).to_csv(artifact_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(feature_importance).to_csv(artifact_dir / "feature_importance.csv", index=False)
    artifact_check = verify_exported_artifact(artifact_dir, feature_columns)

    return {
        "disease": target.disease,
        "runtime_key": runtime_key,
        "label_policy": target.label_policy,
        "data_version": target.data_version,
        "artifact_dir": str(artifact_dir),
        "rows": int(len(frame)),
        "positive_rate": positive_count / len(frame),
        "feature_count": len(feature_columns),
        "threshold": threshold,
        "precision": oof_metrics["precision"],
        "recall": oof_metrics["recall"],
        "f1": oof_metrics["f1"],
        "f2": oof_metrics["f2"],
        "roc_auc": oof_metrics["roc_auc"],
        "pr_auc": oof_metrics["pr_auc"],
        "ece": oof_metrics["ece"],
        "artifact_check": artifact_check["required_files_present"],
        "status": "exported",
    }


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd().resolve())
    add_src_path(repo_root)
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    targets = [target for target in MAIN_SCREENING_TARGETS if args.target in {"all", target.disease}]
    print(f"[EXPORT] targets={len(targets)} output_root={output_root}", flush=True)
    rows = [export_target(repo_root, output_root, target, args.folds, args.dry_run) for target in targets]

    summary = pd.DataFrame(rows)
    summary_path = output_root / "export_summary.csv"
    if not args.dry_run:
        summary.to_csv(summary_path, index=False)
        print(f"[EXPORT] summary={summary_path}", flush=True)
        print(
            summary[
                ["disease", "runtime_key", "rows", "positive_rate", "threshold", "precision", "recall", "f1", "roc_auc"]
            ].to_string(index=False),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
