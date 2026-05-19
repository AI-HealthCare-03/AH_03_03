"""
공통 Threshold 탐색 모듈
Recall >= RECALL_MIN 조건에서 F1 최대화
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score, precision_score, recall_score


def tune_threshold(
    proba: NDArray[np.float64],
    y_true: NDArray[np.int64],
    recall_min: float = 0.87,
    thr_range: NDArray[np.float64] | None = None,
) -> tuple[float, float, float, float]:
    """
    Recall >= recall_min 조건에서 F1이 최대가 되는 threshold 탐색

    Parameters
    ----------
    proba      : 예측 확률 (1D array)
    y_true     : 실제 레이블 (1D array)
    recall_min : 최소 Recall 기준 (default: 0.87)
    thr_range  : 탐색할 threshold 범위 (default: 0.30~0.70, step 0.01)

    Returns
    -------
    (best_thr, best_f1, recall_at_best, precision_at_best)
    """
    if thr_range is None:
        thr_range = np.arange(0.30, 0.71, 0.01)

    best_f1: float = 0.0
    best_thr: float = 0.30

    for thr in thr_range:
        pred = (proba >= thr).astype(int)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        if r >= recall_min and f > best_f1:
            best_f1 = f
            best_thr = round(float(thr), 2)

    final_pred = (proba >= best_thr).astype(int)
    final_recall = float(recall_score(y_true, final_pred, zero_division=0))
    final_prec = float(precision_score(y_true, final_pred, zero_division=0))

    return best_thr, best_f1, final_recall, final_prec


def build_threshold_table(
    proba: NDArray[np.float64],
    y_true: NDArray[np.int64],
    thr_range: NDArray[np.float64] | None = None,
) -> list[dict[str, float]]:
    """
    전체 threshold 범위에 대한 Recall / Precision / F1 테이블 생성
    threshold_tuning.csv 저장용

    Returns
    -------
    list of {"threshold": float, "recall": float, "precision": float, "f1": float}
    """
    if thr_range is None:
        thr_range = np.arange(0.30, 0.71, 0.01)

    rows: list[dict[str, float]] = []
    for thr in thr_range:
        pred = (proba >= thr).astype(int)
        rows.append({
            "threshold": round(float(thr), 2),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        })
    return rows
