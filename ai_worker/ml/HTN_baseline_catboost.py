"""
고혈압유병 예측 베이스라인 — CatBoost
Python 3.9 | catboost>=1.2 | scikit-learn>=1.4 | pandas>=2.2
검증: Stratified 5-Fold CV
지표: AUC-ROC / F1 / Recall
"""

import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "hn24_file1_preprocessed.csv")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "baseline_catboost")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET = "고혈압유병"
N_SPLITS = 5
SEED = 42

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"[0] 데이터 로드 | shape: {df.shape}")

df = df.dropna(subset=[TARGET]).reset_index(drop=True)
print(f"[0] {TARGET} 결측 제거 후 | shape: {df.shape}")
print(f"[0] 클래스 분포:\n{df[TARGET].value_counts().to_string()}")

# ── X / Y 분리 ────────────────────────────────────────────────
drop_cols = ["고혈압유병", "당뇨유병", "이상지질혈증유병", "비만단계"]
X = df.drop(columns=drop_cols)
y = df[TARGET].astype(int)

print(f"\n[1] Feature 수: {X.shape[1]}")
print(f"    Feature 목록: {list(X.columns)}")

# ── 클래스 불균형 확인 ────────────────────────────────────────
# CatBoost: class_weights 파라미터로 전달 (딕셔너리 형태)
neg, pos = (y == 0).sum(), (y == 1).sum()
ratio = neg / pos
print(f"\n[2] 클래스 불균형 | 음성(0): {neg} / 양성(1): {pos}")
print(f"    class_weights: {{0: 1.0, 1: {ratio:.4f}}}")

# ── CatBoost 파라미터 (베이스라인) ────────────────────────────
# CatBoost는 NaN 자체 처리 가능 → 음주량_enc NaN 그대로 전달
params = dict(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights={0: 1.0, 1: ratio},
    early_stopping_rounds=50,
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,  # 불필요한 catboost_info 폴더 생성 방지
)

# ── Stratified 5-Fold CV ──────────────────────────────────────
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pred_proba = np.zeros(len(y))
oof_pred_label = np.zeros(len(y))
fold_scores = []

print(f"\n[3] {N_SPLITS}-Fold CV 시작")
print("=" * 55)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_label = (val_proba >= 0.5).astype(int)

    oof_pred_proba[val_idx] = val_proba
    oof_pred_label[val_idx] = val_label

    auc = roc_auc_score(y_val, val_proba)
    f1 = f1_score(y_val, val_label)
    recall = recall_score(y_val, val_label)

    fold_scores.append({"fold": fold, "auc": auc, "f1": f1, "recall": recall, "best_iter": model.best_iteration_})
    print(f"  Fold {fold} | AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | best_iter: {model.best_iteration_}")

# ── OOF 전체 성능 ─────────────────────────────────────────────
print("=" * 55)
oof_auc = roc_auc_score(y, oof_pred_proba)
oof_f1 = f1_score(y, oof_pred_label)
oof_recall = recall_score(y, oof_pred_label)

scores_df = pd.DataFrame(fold_scores)
print("\n[4] OOF 전체 성능")
print(f"    AUC-ROC : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
print(f"    F1      : {oof_f1:.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})")
print(f"    Recall  : {oof_recall:.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})")

print("\n[5] Classification Report (OOF)")
print(classification_report(y, oof_pred_label, target_names=["정상(0)", "고혈압(1)"]))

print("[6] Confusion Matrix (OOF)")
cm = confusion_matrix(y, oof_pred_label)
print(f"    TN={cm[0, 0]} FP={cm[0, 1]}")
print(f"    FN={cm[1, 0]} TP={cm[1, 1]}")

# ── Feature Importance (마지막 fold 기준) ─────────────────────
print("\n[7] Feature Importance Top 15 (마지막 fold)")
fi = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.get_feature_importance(),
    }
).sort_values("importance", ascending=False)
print(fi.head(15).to_string(index=False))

# ── 결과 저장 ─────────────────────────────────────────────────
scores_df.to_csv(os.path.join(MODEL_DIR, "fold_scores.csv"), index=False)
fi.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "y_true": y,
        "oof_proba": oof_pred_proba,
        "oof_label": oof_pred_label,
    }
)
oof_df.to_csv(os.path.join(MODEL_DIR, "oof_predictions.csv"), index=False)

print(f"\n[8] 저장 완료 → {MODEL_DIR}")
