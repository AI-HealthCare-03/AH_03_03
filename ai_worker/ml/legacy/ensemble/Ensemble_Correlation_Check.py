"""
LGBM v3 + CatBoost FE OOF Correlation 확인 및 앙상블 분석
HTN / DM / DL 각각 단순 평균 앙상블 후 OOF 성능 비교

Python 3.13 | numpy | scikit-learn
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR: str = str(Path(__file__).parent.parent.parent.parent / "ai_worker" / "ml" / "LGB18~24" / "outputs")
CAT_DIR: str = str(Path(__file__).parent.parent.parent.parent / "ai_worker" / "ml" / "CAT18~24" / "outputs")

RECALL_MIN: float = 0.87
THRESHOLD_RANGE: NDArray[np.float64] = np.arange(0.30, 0.71, 0.01)

# ── 타겟별 설정 ───────────────────────────────────────────────
TARGETS: dict = {
    "HTN": {
        "lgbm_dir": f"{BASE_DIR}/optuna_HTN_FE_v3",
        "catboost_dir": f"{CAT_DIR}/catboost_HTN_FE",
    },
    "DM": {
        "lgbm_dir": f"{BASE_DIR}/optuna_DM_FE_v3",
        "catboost_dir": f"{CAT_DIR}/catboost_DM_FE",
    },
    "DL": {
        "lgbm_dir": f"{BASE_DIR}/optuna_DL_FE_v3",
        "catboost_dir": f"{CAT_DIR}/catboost_DL_FE",
    },
}


def tune_threshold_f1(
    proba: NDArray[np.float64],
    y_true: NDArray[np.int64],
) -> tuple[float, float, float, float]:
    best_f1: float = 0.0
    best_thr: float = 0.30
    for thr in THRESHOLD_RANGE:
        pred = (proba >= thr).astype(int)
        r = recall_score(y_true, pred)
        f = f1_score(y_true, pred, zero_division=0)
        if r >= RECALL_MIN and f > best_f1:
            best_f1 = f
            best_thr = round(float(thr), 2)
    final_pred = (proba >= best_thr).astype(int)
    final_recall = recall_score(y_true, final_pred)
    final_prec = precision_score(y_true, final_pred, zero_division=0)
    return best_thr, best_f1, final_recall, final_prec


def evaluate(
    proba: NDArray[np.float64],
    y_true: NDArray[np.int64],
    thr: float,
    label: str,
) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "model": label,
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "threshold": thr,
    }


def run_correlation_check(target_name: str, cfg: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{target_name}] LGBM v3 + CatBoost FE Correlation 확인")
    print(f"{'=' * 60}")

    # ── OOF proba 로드 ────────────────────────────────────────
    lgbm_oof = np.load(f"{cfg['lgbm_dir']}/oof_proba.npy")
    cat_oof = np.load(f"{cfg['catboost_dir']}/oof_proba.npy")
    y_train = np.load(f"{cfg['lgbm_dir']}/oof_y_true.npy")

    lgbm_thr = float(np.load(f"{cfg['lgbm_dir']}/best_threshold.npy")[0])
    cat_thr = float(np.load(f"{cfg['catboost_dir']}/best_threshold.npy")[0])

    # ── OOF Correlation ───────────────────────────────────────
    corr = float(np.corrcoef(lgbm_oof, cat_oof)[0, 1])
    print(f"\n[OOF Correlation] LGBM v3 vs CatBoost FE : {corr:.4f}")
    if corr > 0.95:
        print("  ⚠️  상관관계 높음 (>0.95) → 앙상블 다양성 제한적")
    elif corr > 0.90:
        print("  △  상관관계 보통 (0.90~0.95) → 앙상블 효과 소폭 기대")
    else:
        print("  ✅  상관관계 낮음 (<0.90) → 앙상블 효과 기대")

    # ── 앙상블 proba ──────────────────────────────────────────
    ens_oof = (lgbm_oof + cat_oof) / 2.0
    ens_thr, ens_f1, ens_recall, ens_prec = tune_threshold_f1(ens_oof, y_train)

    # ── OOF 성능 비교 ─────────────────────────────────────────
    rows = [
        evaluate(lgbm_oof, y_train, lgbm_thr, "LGBM v3 (OOF)"),
        evaluate(cat_oof, y_train, cat_thr, "CatBoost FE (OOF)"),
        evaluate(ens_oof, y_train, ens_thr, "앙상블 평균 (OOF)"),
    ]
    result_df = pd.DataFrame(rows)
    print("\n[OOF 성능 비교]")
    print(result_df.to_string(index=False))


def main() -> None:
    print("=" * 60)
    print("LGBM v3 + CatBoost FE 앙상블 Correlation 분석")
    print(f"RECALL_MIN: {RECALL_MIN} | 목표함수: F1 최대화")
    print("=" * 60)

    for target_name, cfg in TARGETS.items():
        run_correlation_check(target_name, cfg)

    print(f"\n{'=' * 60}")
    print("판단 기준: correlation < 0.90 → 앙상블 진행")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
