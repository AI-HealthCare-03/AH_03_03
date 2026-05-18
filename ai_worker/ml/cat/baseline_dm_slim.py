"""
당뇨유병 예측 — 피처 정리 버전
Python 3.9 | catboost>=1.2 | scikit-learn>=1.4
제거: 직업 8개 + 고혈압가족력 3개 + 고지혈증가족력 3개
사용 피처: 성별, 나이, 체중, BMI, 당뇨가족력_부/모/형제, 걷기일수, 근력운동일수, 현재흡연, 과거음주_현재금주, 음주빈도_enc, 음주량_enc, 키
"""

import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    classification_report, confusion_matrix,
)

warnings.filterwarnings('ignore')

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'hn24_file1_preprocessed.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'outputs', 'baseline_dm_slim')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET   = '당뇨유병'
N_SPLITS = 5
SEED     = 42

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

drop_cols = ['고혈압유병', '당뇨유병', '이상지질혈증유병', '비만단계']
X = df.drop(columns=drop_cols)
y = df[TARGET].astype(int)

# ── 피처 정리 ─────────────────────────────────────────────────
remove_cols = [
    # 직업 OHE (8개)
    '직업_관리전문', '직업_기능노무', '직업_농림어업', '직업_무직',
    '직업_사무', '직업_서비스판매', '직업_작업미상', '직업_주부학생',
    # 고혈압 가족력 (3개) — 당뇨 예측과 직접 무관
    '고혈압가족력_부', '고혈압가족력_모', '고혈압가족력_형제',
    # 고지혈증 가족력 (3개) — 당뇨 예측과 직접 무관
    '고지혈증가족력_부', '고지혈증가족력_모', '고지혈증가족력_형제',
]

X_slim = X.drop(columns=remove_cols)

neg, pos = (y == 0).sum(), (y == 1).sum()
ratio = neg / pos

print(f"[0] 데이터 로드 완료 | shape: {df.shape}")
print(f"[0] 클래스 분포 | 정상: {neg} / 당뇨: {pos} | ratio: {ratio:.4f}")
print(f"[0] 피처 수: {X.shape[1]}개 → {X_slim.shape[1]}개")
print(f"[0] 사용 피처: {list(X_slim.columns)}")

# ── CatBoost 파라미터 (베이스라인) ────────────────────────────
params = dict(
    iterations            = 500,
    learning_rate         = 0.05,
    depth                 = 6,
    loss_function         = 'Logloss',
    eval_metric           = 'AUC',
    class_weights         = {0: 1.0, 1: ratio},
    early_stopping_rounds = 50,
    random_seed           = SEED,
    verbose               = False,
    allow_writing_files   = False,
)

# ── Stratified 5-Fold CV ──────────────────────────────────────
skf            = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pred_proba = np.zeros(len(y))
oof_pred_label = np.zeros(len(y))
fold_scores    = []

print(f"\n[1] {N_SPLITS}-Fold CV 시작")
print("=" * 55)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_slim, y), 1):
    X_tr, X_val = X_slim.iloc[tr_idx], X_slim.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**params)
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val))

    val_proba = model.predict_proba(X_val)[:, 1]
    val_label = (val_proba >= 0.5).astype(int)

    oof_pred_proba[val_idx] = val_proba
    oof_pred_label[val_idx] = val_label

    auc    = roc_auc_score(y_val, val_proba)
    f1     = f1_score(y_val, val_label)
    recall = recall_score(y_val, val_label)

    fold_scores.append({'fold': fold, 'auc': auc, 'f1': f1, 'recall': recall,
                        'best_iter': model.best_iteration_})
    print(f"  Fold {fold} | AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} "
          f"| best_iter: {model.best_iteration_}")

# ── OOF 전체 성능 ─────────────────────────────────────────────
print("=" * 55)
scores_df  = pd.DataFrame(fold_scores)
oof_auc    = roc_auc_score(y, oof_pred_proba)
oof_f1     = f1_score(y, oof_pred_label)
oof_recall = recall_score(y, oof_pred_label)

print(f"\n[2] OOF 전체 성능")
print(f"    AUC-ROC : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
print(f"    F1      : {oof_f1:.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})")
print(f"    Recall  : {oof_recall:.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})")

print(f"\n[3] Classification Report (OOF)")
print(classification_report(y, oof_pred_label, target_names=['정상(0)', '당뇨(1)']))

cm = confusion_matrix(y, oof_pred_label)
print(f"[4] Confusion Matrix")
print(f"    TN={cm[0,0]} FP={cm[0,1]}")
print(f"    FN={cm[1,0]} TP={cm[1,1]}")

# ── Feature Importance ────────────────────────────────────────
print(f"\n[5] Feature Importance Top 15 (마지막 fold)")
fi = pd.DataFrame({
    'feature':    X_slim.columns,
    'importance': model.get_feature_importance(),
}).sort_values('importance', ascending=False)
print(fi.to_string(index=False))

# ── 저장 ──────────────────────────────────────────────────────
scores_df.to_csv(os.path.join(MODEL_DIR, 'fold_scores.csv'), index=False)
fi.to_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'), index=False)
np.save(os.path.join(MODEL_DIR, 'oof_y_true.npy'),  y.values)
np.save(os.path.join(MODEL_DIR, 'oof_proba.npy'),   oof_pred_proba)
np.save(os.path.join(MODEL_DIR, 'oof_label.npy'),   oof_pred_label)

print(f"\n[6] 저장 완료 → {MODEL_DIR}")
