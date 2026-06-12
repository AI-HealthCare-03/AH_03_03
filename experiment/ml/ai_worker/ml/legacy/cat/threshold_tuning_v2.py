"""
3개 질환 통합 Threshold 튜닝 — 50 trial vs 100 trial 비교
Python 3.9 | scikit-learn>=1.4
OOF 확률값 기준 최적 threshold 탐색
목적: 스크리닝 서비스 → Recall >= 0.85 조건에서 F1 최대
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "threshold_tuning_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 질환별 설정 ───────────────────────────────────────────────
CONFIGS = {
    "고혈압": {
        "dir_50": "tuned_v2_HTN",
        "dir_100": "tuned_v2_HTN_100",
    },
    "당뇨": {
        "dir_50": "tuned_v2_DM",
        "dir_100": "tuned_v2_DM_100",
    },
    "이상지질혈증": {
        "dir_50": "tuned_v2_HL",
        "dir_100": "tuned_v2_HL_100",
    },
}


def find_best_threshold(y_true, oof_proba, recall_min=0.85):
    thresholds = np.arange(0.1, 0.91, 0.01)
    results = []
    for thr in thresholds:
        pred = (oof_proba >= thr).astype(int)
        cm = confusion_matrix(y_true, pred)
        results.append(
            {
                "threshold": round(thr, 2),
                "recall": recall_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "auc": roc_auc_score(y_true, oof_proba),
                "fn": cm[1, 0],
                "fp": cm[0, 1],
                "tn": cm[0, 0],
                "tp": cm[1, 1],
            }
        )
    results_df = pd.DataFrame(results)

    cond = results_df[results_df["recall"] >= recall_min]
    if len(cond) > 0:
        best = cond.loc[cond["f1"].idxmax()]
    else:
        best = results_df.loc[results_df["recall"].idxmax()]

    return best, results_df


print("=" * 65)
print("3개 질환 Threshold 튜닝 — 50 trial vs 100 trial 비교")
print("=" * 65)

final_results = {}

for disease, cfg in CONFIGS.items():
    print(f"\n{'─' * 65}")
    print(f"[{disease}]")

    results = {}
    for trial_ver, dir_key in [("50", "dir_50"), ("100", "dir_100")]:
        oof_dir = os.path.join(BASE_DIR, "outputs", cfg[dir_key])
        y_true = np.load(os.path.join(oof_dir, "oof_y_true.npy"))
        oof_proba = np.load(os.path.join(oof_dir, "oof_proba.npy"))

        best, results_df = find_best_threshold(y_true, oof_proba)
        results[trial_ver] = {"best": best, "df": results_df}

        # 저장
        results_df.to_csv(os.path.join(OUTPUT_DIR, f"threshold_results_{disease}_{trial_ver}.csv"), index=False)

    # 비교 출력
    b50 = results["50"]["best"]
    b100 = results["100"]["best"]

    print(f"\n  {'구분':<20} {'Threshold':>10} {'AUC':>8} {'Recall':>8} {'F1':>8} {'FN':>6} {'FP':>6}")
    print("  " + "-" * 68)
    print(
        f"  {'50 trial':<20} {b50['threshold']:>10.2f} {b50['auc']:>8.4f} {b50['recall']:>8.4f} {b50['f1']:>8.4f} {b50['fn']:>6.0f} {b50['fp']:>6.0f}"
    )
    print(
        f"  {'100 trial':<20} {b100['threshold']:>10.2f} {b100['auc']:>8.4f} {b100['recall']:>8.4f} {b100['f1']:>8.4f} {b100['fn']:>6.0f} {b100['fp']:>6.0f}"
    )

    # 채택 기준: Recall 우선, 같으면 F1
    if b50["recall"] > b100["recall"]:
        adopted = "50"
        adopted_best = b50
    elif b100["recall"] > b50["recall"]:
        adopted = "100"
        adopted_best = b100
    else:
        adopted = "50" if b50["f1"] >= b100["f1"] else "100"
        adopted_best = b50 if adopted == "50" else b100

    print(
        f"\n  ★ 채택: {adopted} trial | Threshold: {adopted_best['threshold']} | Recall: {adopted_best['recall']:.4f} | F1: {adopted_best['f1']:.4f} | FN: {adopted_best['fn']:.0f}"
    )

    final_results[disease] = {
        "adopted_trial": adopted,
        "threshold": adopted_best["threshold"],
        "auc": round(adopted_best["auc"], 4),
        "recall": round(adopted_best["recall"], 4),
        "f1": round(adopted_best["f1"], 4),
        "fn": int(adopted_best["fn"]),
        "fp": int(adopted_best["fp"]),
    }

# ── 최종 요약 ─────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("최종 채택 요약")
print(f"{'질환':<15} {'Trial':>6} {'Threshold':>10} {'AUC':>8} {'Recall':>8} {'F1':>8} {'FN':>6}")
print("-" * 65)
for disease, res in final_results.items():
    print(
        f"{disease:<15} {res['adopted_trial']:>6} {res['threshold']:>10.2f} "
        f"{res['auc']:>8.4f} {res['recall']:>8.4f} {res['f1']:>8.4f} {res['fn']:>6}"
    )

# 저장
pd.DataFrame(final_results).T.to_csv(os.path.join(OUTPUT_DIR, "final_threshold_summary.csv"))
print(f"\n저장 완료 → {OUTPUT_DIR}")
