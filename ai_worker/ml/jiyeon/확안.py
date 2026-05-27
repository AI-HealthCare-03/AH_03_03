import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score

BASE = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml"
targets = {"HTN": 0.50, "DM": 0.50, "DL": 0.47}

for t, thr in targets.items():
    proba = np.load(f"{BASE}/CAT15~24/outputs/optuna_{t}_FE/oof_proba.npy")
    y = np.load(f"{BASE}/CAT15~24/outputs/optuna_{t}_FE/oof_y_true.npy")
    pred = (proba >= thr).astype(int)
    print(
        f"{t} (thr={thr}) | AUC: {roc_auc_score(y, proba):.4f} | Recall: {recall_score(y, pred):.4f} | F1: {f1_score(y, pred):.4f}"
    )
