import numpy as np
from scipy.stats import pearsonr

BASE = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml"
targets = ["HTN", "DM", "DL"]

for t in targets:
    cat = np.load(f"{BASE}/CAT15~24/outputs/optuna_{t}_FE/oof_proba.npy")
    lgb = np.load(f"{BASE}/LGB15~24/outputs/baseline_lgbm_{t}_FE/oof_proba.npy")
    min_len = min(len(cat), len(lgb))
    corr, _ = pearsonr(cat[:min_len], lgb[:min_len])
    print(f"{t} | CatBoost vs LGBM correlation: {corr:.4f}")
