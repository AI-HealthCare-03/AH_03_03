import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

BASE = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml"

# 내 모델 OOF proba + y_true 로드
targets = {
    "HTN": {"thr_mine": 0.46, "thr_common": 0.50},
    "DM": {"thr_mine": 0.45, "thr_common": 0.50},
    "DL": {"thr_mine": 0.47, "thr_common": 0.45},
}

# 조원 BASE 수치 (전체 OOF, threshold 고정 기준)
teammate = {
    "HTN": {"auc": 0.8566, "recall": 0.8399, "precision": 0.5807, "f1": 0.6866, "fp": 3807, "fn": 1005},
    "DM": {"auc": 0.8067, "recall": 0.8139, "precision": 0.4798, "f1": 0.6037, "fp": 5576, "fn": 1176},
    "DL": {"auc": 0.7782, "recall": 0.8776, "precision": 0.4662, "f1": 0.6089, "fp": 6823, "fn": 831},
}

for t, cfg in targets.items():
    proba = np.load(f"{BASE}/CAT15~24/outputs/optuna_{t}_FE/oof_proba.npy")
    y = np.load(f"{BASE}/CAT15~24/outputs/optuna_{t}_FE/oof_y_true.npy")

    # 내 threshold로
    pred_mine = (proba >= cfg["thr_mine"]).astype(int)
    cm_mine = confusion_matrix(y, pred_mine)

    # 공통 threshold로
    pred_common = (proba >= cfg["thr_common"]).astype(int)
    cm_common = confusion_matrix(y, pred_common)

    auc = roc_auc_score(y, proba)

    print(f"\n{'=' * 60}")
    print(f"[{t}]")
    print(f"{'=' * 60}")
    print(f"{'지표':<15} {'조원(고정thr)':>15} {'내모델(튜닝thr)':>15} {'내모델(공통thr)':>15}")
    print(f"{'-' * 60}")
    print(f"{'threshold':<15} {cfg['thr_common']:>15.2f} {cfg['thr_mine']:>15.2f} {cfg['thr_common']:>15.2f}")
    print(f"{'AUC':<15} {teammate[t]['auc']:>15.4f} {auc:>15.4f} {auc:>15.4f}")
    print(
        f"{'Recall':<15} {teammate[t]['recall']:>15.4f} {recall_score(y, pred_mine):>15.4f} {recall_score(y, pred_common):>15.4f}"
    )
    print(
        f"{'Precision':<15} {teammate[t]['precision']:>15.4f} {precision_score(y, pred_mine, zero_division=0):>15.4f} {precision_score(y, pred_common, zero_division=0):>15.4f}"
    )
    print(f"{'F1':<15} {teammate[t]['f1']:>15.4f} {f1_score(y, pred_mine):>15.4f} {f1_score(y, pred_common):>15.4f}")
    print(f"{'FP':<15} {teammate[t]['fp']:>15} {cm_mine[0, 1]:>15} {cm_common[0, 1]:>15}")
    print(f"{'FN':<15} {teammate[t]['fn']:>15} {cm_mine[1, 0]:>15} {cm_common[1, 0]:>15}")
    print("\n  ※ FN = 실제 환자인데 정상으로 예측한 수 (낮을수록 좋음)")
