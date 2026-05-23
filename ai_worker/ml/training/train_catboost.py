from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_worker.ml.common.artifacts import save_training_metadata, write_json
from ai_worker.ml.common.features import apply_feature_engineering
from ai_worker.ml.common.metrics import evaluate
from ai_worker.ml.common.threshold import build_threshold_table, tune_threshold
from ai_worker.ml.datasets.loaders import load_csv_dataset

TARGET_COLUMNS = {"고혈압유병", "당뇨유병", "이상지질혈증유병", "비만단계"}


def train_catboost_from_config(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    sample_size: int | None = None,
) -> dict[str, Any]:
    import pandas as pd

    df = load_csv_dataset(config, sample_size=sample_size)
    disease = str(config["disease"])
    target_column = str(config["target_column"])
    if target_column not in df.columns:
        raise ValueError(f"target_column이 데이터셋에 없습니다: {target_column}")

    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    base_feature_columns = list(config.get("feature_columns") or [])
    missing_base_features = [column for column in base_feature_columns if column not in df.columns]
    if missing_base_features:
        raise ValueError(f"feature_columns가 데이터셋에 없습니다: {missing_base_features}")

    if base_feature_columns:
        df = df[base_feature_columns + [target_column]].copy()

    df = apply_feature_engineering(
        df,
        disease=disease,
        fe_keys_override=list(config.get("fe_keys") or []),
        extra_fe=list(config.get("extra_fe") or []),
        verbose=not dry_run,
    )
    drop_columns = [column for column in TARGET_COLUMNS if column in df.columns]
    X = df.drop(columns=drop_columns)
    y = df[target_column].astype(int)

    artifact_dir = Path(str(config.get("artifact_dir") or f"ai_worker/ml/artifacts/{disease.lower()}/catboost"))
    if dry_run:
        return {
            "status": "dry_run",
            "disease": disease,
            "rows": int(len(df)),
            "target_column": target_column,
            "feature_count": int(X.shape[1]),
            "feature_columns": list(X.columns),
            "artifact_dir": str(artifact_dir),
        }

    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, train_test_split

    seed = int(config.get("random_seed") or config.get("seed") or 42)
    folds = int(config.get("folds") or config.get("n_splits") or 5)
    test_size = float(config.get("test_size") or 0.2)
    train_df, test_df, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    pos_weight = float(neg_count / max(pos_count, 1))
    model_params = _build_catboost_params(config, seed, pos_weight)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    oof_proba = np.zeros(len(y_train), dtype=float)
    fold_scores: list[dict[str, Any]] = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (train_index, valid_index) in enumerate(skf.split(train_df, y_train), 1):
        X_train = train_df.iloc[train_index]
        X_valid = train_df.iloc[valid_index]
        fold_y_train = y_train.iloc[train_index]
        fold_y_valid = y_train.iloc[valid_index]
        model = CatBoostClassifier(**model_params)
        model.fit(Pool(X_train, fold_y_train), eval_set=Pool(X_valid, fold_y_valid))
        valid_proba = model.predict_proba(X_valid)[:, 1]
        oof_proba[valid_index] = valid_proba
        valid_pred = (valid_proba >= 0.5).astype(int)
        model.save_model(str(artifact_dir / f"model_fold{fold}.cbm"))
        fold_scores.append(
            {
                "fold": fold,
                "auc": float(roc_auc_score(fold_y_valid, valid_proba)),
                "recall": float(recall_score(fold_y_valid, valid_pred, zero_division=0)),
                "precision": float(precision_score(fold_y_valid, valid_pred, zero_division=0)),
                "f1": float(f1_score(fold_y_valid, valid_pred, zero_division=0)),
                "best_iteration": getattr(model, "best_iteration_", None),
            }
        )

    threshold_config = dict(config.get("threshold_strategy") or {})
    recall_min = float(threshold_config.get("recall_min") or config.get("recall_min") or 0.87)
    threshold_range = _threshold_range(threshold_config)
    best_threshold, best_f1, threshold_recall, threshold_precision = tune_threshold(
        oof_proba,
        y_train.to_numpy(dtype=int),
        recall_min=recall_min,
        thr_range=threshold_range,
    )

    test_proba_matrix = []
    for model_path in sorted(artifact_dir.glob("model_fold*.cbm")):
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        test_proba_matrix.append(model.predict_proba(test_df)[:, 1])
    test_proba = np.column_stack(test_proba_matrix).mean(axis=1)
    test_metrics = evaluate(y_test.to_numpy(dtype=int), test_proba, best_threshold, label=f"{disease} Test")
    feature_importance = _feature_importance(model, list(train_df.columns))

    threshold_payload = {
        "strategy": threshold_config.get("name") or "recall_min_f1",
        "threshold": best_threshold,
        "recall_min": recall_min,
        "oof_f1": best_f1,
        "oof_recall": threshold_recall,
        "oof_precision": threshold_precision,
    }
    metrics_payload = {
        "folds": fold_scores,
        "test": test_metrics,
        "positive_class_weight": pos_weight,
        "rows": int(len(df)),
        "feature_importance": feature_importance,
    }
    save_training_metadata(
        artifact_dir,
        feature_columns=list(train_df.columns),
        threshold=threshold_payload,
        metrics=metrics_payload,
        experiment_config=config,
    )
    np.save(artifact_dir / "oof_proba.npy", oof_proba)
    np.save(artifact_dir / "oof_y_true.npy", y_train.to_numpy(dtype=int))
    pd.DataFrame(build_threshold_table(oof_proba, y_train.to_numpy(dtype=int), threshold_range)).to_csv(
        artifact_dir / "threshold_tuning.csv",
        index=False,
    )
    write_json(artifact_dir / "model_params.json", model_params)
    return {
        "status": "trained",
        "disease": disease,
        "artifact_dir": str(artifact_dir),
        "threshold": best_threshold,
        "metrics": test_metrics,
        "feature_count": int(train_df.shape[1]),
    }


def _feature_importance(model: Any, feature_columns: list[str]) -> list[dict[str, Any]]:
    try:
        importances = model.get_feature_importance()
    except Exception:
        return []
    rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_columns, importances, strict=False)
    ]
    return sorted(rows, key=lambda item: item["importance"], reverse=True)


def _build_catboost_params(config: dict[str, Any], seed: int, pos_weight: float) -> dict[str, Any]:
    params = {
        "iterations": 500,
        "learning_rate": 0.03,
        "depth": 5,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    params.update({key: value for key, value in dict(config.get("model_params") or {}).items() if value is not None})
    class_weight_policy = str(config.get("class_weight") or "balanced")
    if class_weight_policy == "balanced":
        params["class_weights"] = [1.0, pos_weight]
    return params


def _threshold_range(threshold_config: dict[str, Any]) -> np.ndarray:
    values = threshold_config.get("range")
    if not values:
        return np.arange(0.30, 0.71, 0.01)
    start, stop, step = [float(value) for value in values]
    return np.arange(start, stop + (step / 2), step)
