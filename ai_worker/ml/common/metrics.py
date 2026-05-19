"""
공통 평가 지표 모듈
Test / OOF 평가, Classification Report, Confusion Matrix
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(
    y_true: NDArray[np.int64],
    proba: NDArray[np.float64],
    threshold: float,
    target_names: list[str] | None = None,
    label: str = "Test",
) -> dict[str, float]:
    """
    단일 threshold 기준 전체 평가 출력 + 결과 dict 반환

    Parameters
    ----------
    y_true       : 실제 레이블
    proba        : 예측 확률
    threshold    : 분류 기준 threshold
    target_names : classification report 레이블명 (예: ["정상(0)", "당뇨(1)"])
    label        : 출력 헤더용 레이블 (예: "Test", "OOF")

    Returns
    -------
    {"auc": float, "recall": float, "precision": float, "f1": float}
    """
    pred = (proba >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, proba))
    recall = float(recall_score(y_true, pred, zero_division=0))
    prec = float(precision_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    cm = confusion_matrix(y_true, pred)

    print(f"\n[{label}] 평가 (threshold={threshold:.2f})")
    print("=" * 60)
    print(f"    AUC       : {auc:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    if target_names:
        print(f"\n[{label}] Classification Report")
        print(classification_report(y_true, pred, target_names=target_names))

    return {"auc": auc, "recall": recall, "precision": prec, "f1": f1}


def evaluate_oof(
    y_true: NDArray[np.int64],
    proba: NDArray[np.float64],
    fold_scores: list[dict[str, float]],
    threshold_05: float = 0.5,
) -> dict[str, float]:
    """
    OOF 전체 성능 출력 (threshold=0.5 기준 + fold avg/std)

    Parameters
    ----------
    y_true        : OOF 실제 레이블
    proba         : OOF 예측 확률
    fold_scores   : 각 fold 결과 dict 리스트 (auc, recall, f1 포함)
    threshold_05  : 출력용 threshold (default 0.5)

    Returns
    -------
    {"auc": float, "recall": float, "f1": float}
    """
    pred = (proba >= threshold_05).astype(int)
    auc = float(roc_auc_score(y_true, proba))
    recall = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))

    auc_arr = np.array([s["auc"] for s in fold_scores])
    recall_arr = np.array([s["recall"] for s in fold_scores])
    f1_arr = np.array([s["f1"] for s in fold_scores])

    print(f"\n[OOF] 전체 성능 (threshold={threshold_05})")
    print(f"    AUC    : {auc:.4f}  (fold avg: {auc_arr.mean():.4f} ± {auc_arr.std():.4f})")
    print(f"    Recall : {recall:.4f}  (fold avg: {recall_arr.mean():.4f} ± {recall_arr.std():.4f})")
    print(f"    F1     : {f1:.4f}  (fold avg: {f1_arr.mean():.4f} ± {f1_arr.std():.4f})")

    return {"auc": auc, "recall": recall, "f1": f1}
