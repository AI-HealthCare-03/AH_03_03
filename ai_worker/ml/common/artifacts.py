"""
공통 산출물 저장 모듈
fold_scores, feature_importance, threshold_tuning, oof_proba, best_params 저장
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def save_artifacts(
    model_dir: str,
    oof_proba: NDArray[np.float64],
    oof_y_true: NDArray[np.int64],
    best_threshold: float,
    fold_scores: list[dict],
    feature_importance: pd.DataFrame,
    threshold_table: list[dict],
    best_params: dict | None = None,
    optuna_trials: pd.DataFrame | None = None,
) -> None:
    """
    실험 산출물 일괄 저장

    Parameters
    ----------
    model_dir          : 저장 경로
    oof_proba          : OOF 예측 확률
    oof_y_true         : OOF 실제 레이블
    best_threshold     : 최적 threshold
    fold_scores        : fold별 성능 dict 리스트
    feature_importance : feature importance DataFrame (feature, importance 컬럼)
    threshold_table    : threshold 탐색 결과 리스트
    best_params        : Optuna best params (없으면 저장 생략)
    optuna_trials      : Optuna trials DataFrame (없으면 저장 생략)
    """
    os.makedirs(model_dir, exist_ok=True)

    np.save(os.path.join(model_dir, "oof_proba.npy"), oof_proba)
    np.save(os.path.join(model_dir, "oof_y_true.npy"), oof_y_true)
    np.save(os.path.join(model_dir, "best_threshold.npy"), np.array([best_threshold]))

    pd.DataFrame(fold_scores).to_csv(os.path.join(model_dir, "fold_scores.csv"), index=False)
    feature_importance.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
    pd.DataFrame(threshold_table).to_csv(os.path.join(model_dir, "threshold_tuning.csv"), index=False)

    if best_params is not None:
        with open(os.path.join(model_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

    if optuna_trials is not None:
        optuna_trials.to_csv(os.path.join(model_dir, "optuna_trials.csv"), index=False)

    print(f"\n[저장] 완료 → {model_dir}")
