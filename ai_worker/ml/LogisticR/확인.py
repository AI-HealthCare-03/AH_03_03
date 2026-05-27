import numpy as np
from scipy.stats import pearsonr

BASE = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml"
targets = ["HTN", "DM", "DL"]

for t in targets:
    cat_proba = np.load(f"{BASE}/CAT15~24/outputs/optuna_{t}_FE/oof_proba.npy")
    log_proba = np.load(f"{BASE}/LogisticR/outputs/logistic_{t}/oof_proba.npy")

    min_len = min(len(cat_proba), len(log_proba))
    corr, _ = pearsonr(cat_proba[:min_len], log_proba[:min_len])
    print(f"{t} | CatBoost vs Logistic correlation: {corr:.4f}")


import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, recall_score, roc_auc_score

BASE = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml"
RECALL_MIN = 0.85
THRESHOLD_RANGE = np.arange(0.30, 0.71, 0.01)
targets = ["HTN", "DM", "DL"]


def tune_threshold(proba, y_true):
    best_f1, best_thr = 0.0, 0.30
    for thr in THRESHOLD_RANGE:
        pred = (proba >= thr).astype(int)
        r = recall_score(y_true, pred)
        f = f1_score(y_true, pred, zero_division=0)
        if r >= RECALL_MIN and f > best_f1:
            best_f1 = f
            best_thr = round(float(thr), 2)
    return best_thr, best_f1


for t in targets:
    print(f"\n{'=' * 60}")
    print(f"[{t}] 앙상블 실험")
    print("=" * 60)

    cat_proba = np.load(f"{BASE}/CAT15~24/outputs/optuna_{t}_FE/oof_proba.npy")
    log_proba = np.load(f"{BASE}/LogisticR/outputs/logistic_{t}/oof_proba.npy")
    y_true = np.load(f"{BASE}/CAT15~24/outputs/optuna_{t}_FE/oof_y_true.npy")

    min_len = min(len(cat_proba), len(log_proba), len(y_true))
    cat_proba = cat_proba[:min_len]
    log_proba = log_proba[:min_len]
    y_true = y_true[:min_len]

    # 단순 평균
    ens_proba = (cat_proba + log_proba) / 2
    thr, f1 = tune_threshold(ens_proba, y_true)
    auc = roc_auc_score(y_true, ens_proba)
    recall = recall_score(y_true, (ens_proba >= thr).astype(int))
    print(f"단순 평균 (0.5:0.5) | AUC: {auc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | thr: {thr:.2f}")

    # 가중 평균 탐색
    best = {"f1": 0, "w": 0.5, "auc": 0, "recall": 0, "thr": 0.5}
    for w in np.arange(0.1, 1.0, 0.1):
        ens = cat_proba * w + log_proba * (1 - w)
        thr, f1 = tune_threshold(ens, y_true)
        if f1 > best["f1"]:
            best = {
                "f1": f1,
                "w": round(w, 1),
                "auc": roc_auc_score(y_true, ens),
                "recall": recall_score(y_true, (ens >= thr).astype(int)),
                "thr": thr,
            }
    print(
        f"최적 가중 평균     (Cat {best['w']:.1f} : Log {1 - best['w']:.1f}) | AUC: {best['auc']:.4f} | Recall: {best['recall']:.4f} | F1: {best['f1']:.4f} | thr: {best['thr']:.2f}"
    )

    # CatBoost 단독 참고
    thr_cat, f1_cat = tune_threshold(cat_proba, y_true)
    auc_cat = roc_auc_score(y_true, cat_proba)
    recall_cat = recall_score(y_true, (cat_proba >= thr_cat).astype(int))
    print(
        f"CatBoost 단독     (참고)           | AUC: {auc_cat:.4f} | Recall: {recall_cat:.4f} | F1: {f1_cat:.4f} | thr: {thr_cat:.2f}"
    )
