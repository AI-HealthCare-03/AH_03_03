"""
고혈압유병 예측 — 관련 가족력만 사용 Ablation Test
Python 3.9 | catboost>=1.2 | scikit-learn>=1.4
직업 제거 + 당뇨/고지혈증 가족력 제거 → 고혈압 가족력만 유지
피처: 28개 → 14개
"""

import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'hn24_file1_preprocessed.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'outputs', 'ablation_htn_only_HTN')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET   = '고혈압유병'
N_SPLITS = 5
SEED     = 42

# 50 trial 최적 파라미터
BEST_PARAMS = dict(
    iterations            = 388,
    learning_rate         = 0.06174532779902198,
    depth                 = 4,
    l2_leaf_reg           = 2.066200719991522,
    bagging_temperature   = 0.6005954120104386,
    random_strength       = 0.16725169601770631,
    border_count          = 198,
    loss_function         = 'Logloss',
    eval_metric           = 'AUC',
    early_stopping_rounds = 50,
    random_seed           = SEED,
    verbose               = False,
    allow_writing_files   = False,
)

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

drop_cols = ['고혈압유병', '당뇨유병', '이상지질혈증유병', '비만단계']
X = df.drop(columns=drop_cols)
y = df[TARGET].astype(int)

# ── 제거할 컬럼 정의 ──────────────────────────────────────────
remove_cols = [
    # 직업 OHE (8개)
    '직업_관리전문', '직업_기능노무', '직업_농림어업', '직업_무직',
    '직업_사무', '직업_서비스판매', '직업_작업미상', '직업_주부학생',
    # 당뇨 가족력 (3개) — 고혈압 예측과 직접 무관
    '당뇨가족력_부', '당뇨가족력_모', '당뇨가족력_형제',
    # 고지혈증 가족력 (3개) — 고혈압 예측과 직접 무관
    '고지혈증가족력_부', '고지혈증가족력_모', '고지혈증가족력_형제',
]

X_slim = X.drop(columns=remove_cols)

neg, pos = (y == 0).sum(), (y == 1).sum()
BEST_PARAMS['class_weights'] = {0: 1.0, 1: neg / pos}

print(f"[0] 데이터 로드 완료 | shape: {df.shape}")
print(f"[0] 제거 컬럼 ({len(remove_cols)}개): 직업 8개 + 당뇨가족력 3개 + 고지혈증가족력 3개")
print(f"[0] 피처 수: {X.shape[1]}개 → {X_slim.shape[1]}개")
print(f"[0] 사용 피처: {list(X_slim.columns)}")

# ── 5-Fold CV ─────────────────────────────────────────────────
skf            = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pred_proba = np.zeros(len(y))
oof_pred_label = np.zeros(len(y))
fold_scores    = []

print(f"\n[1] {N_SPLITS}-Fold CV")
print("=" * 55)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_slim, y), 1):
    X_tr, X_val = X_slim.iloc[tr_idx], X_slim.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**BEST_PARAMS)
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val))

    val_proba = model.predict_proba(X_val)[:, 1]
    val_label = (val_proba >= 0.48).astype(int)

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

print(f"\n[2] OOF 성능 (피처 {X_slim.shape[1]}개 + threshold=0.48)")
print(f"    AUC-ROC : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
print(f"    F1      : {oof_f1:.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})")
print(f"    Recall  : {oof_recall:.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})")

print(f"\n[3] Classification Report")
print(classification_report(y, oof_pred_label, target_names=['정상(0)', '고혈압(1)']))

cm = confusion_matrix(y, oof_pred_label)
print(f"[4] Confusion Matrix")
print(f"    TN={cm[0,0]} FP={cm[0,1]}")
print(f"    FN={cm[1,0]} TP={cm[1,1]}")

# ── 단계별 비교 ───────────────────────────────────────────────
print(f"\n[5] 단계별 비교 (threshold=0.48 기준)")
print(f"    {'구분':<20} {'피처':>5} {'AUC':>8} {'Recall':>8} {'F1':>8} {'FN':>6} {'FP':>6}")
print("    " + "-" * 58)
print(f"    {'베이스라인':<20} {28:>5} {0.8596:>8.4f} {0.8524:>8.4f} {0.6459:>8.4f} {248:>6} {1322:>6}")
print(f"    {'직업 제거':<20} {20:>5} {0.8574:>8.4f} {0.8583:>8.4f} {0.6455:>8.4f} {238:>6} {1346:>6}")
print(f"    {'관련가족력만':<20} {X_slim.shape[1]:>5} {oof_auc:>8.4f} {oof_recall:>8.4f} {oof_f1:>8.4f} {cm[1,0]:>6} {cm[0,1]:>6}")

# ── Feature Importance ────────────────────────────────────────
print(f"\n[6] Feature Importance (마지막 fold)")
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

print(f"\n[7] 저장 완료 → {MODEL_DIR}")
