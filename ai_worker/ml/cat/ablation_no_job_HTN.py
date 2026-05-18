"""
고혈압유병 예측 — 직업 컬럼 제거 Ablation Test
Python 3.9 | catboost>=1.2 | scikit-learn>=1.4
기존 튜닝 파라미터(50 trial) 그대로 사용
직업 OHE 8개 컬럼 제거 후 성능 비교
"""

import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "hn24_file1_preprocessed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "ablation_no_job_HTN")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET = "고혈압유병"
N_SPLITS = 5
SEED = 42

# ── 50 trial 최적 파라미터 ────────────────────────────────────
BEST_PARAMS = dict(
    iterations=388,
    learning_rate=0.06174532779902198,
    depth=4,
    l2_leaf_reg=2.066200719991522,
    bagging_temperature=0.6005954120104386,
    random_strength=0.16725169601770631,
    border_count=198,
    loss_function="Logloss",
    eval_metric="AUC",
    early_stopping_rounds=50,
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
)

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

drop_cols = ["고혈압유병", "당뇨유병", "이상지질혈증유병", "비만단계"]
X = df.drop(columns=drop_cols)
y = df[TARGET].astype(int)

# ── 직업 컬럼 제거 ────────────────────────────────────────────
job_cols = [c for c in X.columns if c.startswith("직업_")]
X_no_job = X.drop(columns=job_cols)

neg, pos = (y == 0).sum(), (y == 1).sum()
ratio = neg / pos
BEST_PARAMS["class_weights"] = {0: 1.0, 1: ratio}

print(f"[0] 데이터 로드 완료 | shape: {df.shape}")
print(f"[0] 제거된 직업 컬럼 ({len(job_cols)}개): {job_cols}")
print(f"[0] 기존 피처 수: {X.shape[1]} → 제거 후: {X_no_job.shape[1]}")

# ── 5-Fold CV ─────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pred_proba = np.zeros(len(y))
oof_pred_label = np.zeros(len(y))
fold_scores = []

print(f"\n[1] {N_SPLITS}-Fold CV (직업 제거)")
print("=" * 55)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_no_job, y), 1):
    X_tr, X_val = X_no_job.iloc[tr_idx], X_no_job.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**BEST_PARAMS)
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val))

    val_proba = model.predict_proba(X_val)[:, 1]
    val_label = (val_proba >= 0.48).astype(int)  # 최적 threshold 적용

    oof_pred_proba[val_idx] = val_proba
    oof_pred_label[val_idx] = val_label

    auc = roc_auc_score(y_val, val_proba)
    f1 = f1_score(y_val, val_label)
    recall = recall_score(y_val, val_label)

    fold_scores.append({"fold": fold, "auc": auc, "f1": f1, "recall": recall, "best_iter": model.best_iteration_})
    print(f"  Fold {fold} | AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | best_iter: {model.best_iteration_}")

# ── OOF 전체 성능 ─────────────────────────────────────────────
print("=" * 55)
scores_df = pd.DataFrame(fold_scores)
oof_auc = roc_auc_score(y, oof_pred_proba)
oof_f1 = f1_score(y, oof_pred_label)
oof_recall = recall_score(y, oof_pred_label)

print("\n[2] OOF 성능 (직업 제거 + threshold=0.48)")
print(f"    AUC-ROC : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
print(f"    F1      : {oof_f1:.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})")
print(f"    Recall  : {oof_recall:.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})")

print("\n[3] Classification Report")
print(classification_report(y, oof_pred_label, target_names=["정상(0)", "고혈압(1)"]))

cm = confusion_matrix(y, oof_pred_label)
print("[4] Confusion Matrix")
print(f"    TN={cm[0, 0]} FP={cm[0, 1]}")
print(f"    FN={cm[1, 0]} TP={cm[1, 1]}")

# ── 기존 결과와 비교 ──────────────────────────────────────────
print("\n[5] 직업 포함 vs 직업 제거 비교 (threshold=0.48 기준)")
print(f"    {'구분':<15} {'AUC':>8} {'Recall':>8} {'F1':>8} {'FN':>6} {'FP':>6}")
print("    " + "-" * 48)
print(f"    {'직업 포함':<15} {0.8596:>8.4f} {0.8524:>8.4f} {0.6459:>8.4f} {248:>6} {1322:>6}")
print(f"    {'직업 제거':<15} {oof_auc:>8.4f} {oof_recall:>8.4f} {oof_f1:>8.4f} {cm[1, 0]:>6} {cm[0, 1]:>6}")
print(
    f"    {'변화':<15} {oof_auc - 0.8596:>+8.4f} {oof_recall - 0.8524:>+8.4f} {oof_f1 - 0.6459:>+8.4f} {cm[1, 0] - 248:>+6} {cm[0, 1] - 1322:>+6}"
)

# ── Feature Importance ────────────────────────────────────────
print("\n[6] Feature Importance Top 15 (마지막 fold)")
fi = pd.DataFrame(
    {
        "feature": X_no_job.columns,
        "importance": model.get_feature_importance(),
    }
).sort_values("importance", ascending=False)
print(fi.head(15).to_string(index=False))

# ── 저장 ──────────────────────────────────────────────────────
scores_df.to_csv(os.path.join(MODEL_DIR, "fold_scores.csv"), index=False)
fi.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
np.save(os.path.join(MODEL_DIR, "oof_y_true.npy"), y.values)
np.save(os.path.join(MODEL_DIR, "oof_proba.npy"), oof_pred_proba)
np.save(os.path.join(MODEL_DIR, "oof_label.npy"), oof_pred_label)

print(f"\n[7] 저장 완료 → {MODEL_DIR}")
