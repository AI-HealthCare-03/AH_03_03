"""
이상지질혈증유병 예측 — CatBoost Ablation (검증구조 통일)
Python 3.9 | catboost>=1.2 | scikit-learn>=1.4

피처    : 12개 (흡연/음주 플래그 제거, ablation 최종)
검증구조 : Train/Test 8:2 Hold-out → Train 내부 5-Fold OOF (LGB 동일 구조)
          Hold-out test는 최종 1회만 사용
앙상블  : fold별 모델 5개 앙상블 평균
Threshold: OOF 기반 자동 탐색 (0.30~0.70, step 0.01)
          Recall >= 0.85 만족 시 Precision 최대 / 미달 시 Recall 최대 fallback
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings('ignore')

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'hn24_file1_preprocessed.csv')
MODEL_DIR  = os.path.join(ML_DIR, 'outputs', 'verif_ablation_cat_DL_12feat')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET          = '이상지질혈증유병'
N_SPLITS        = 5
SEED            = 42
RECALL_MIN      = 0.85
THRESHOLD_RANGE = np.arange(0.30, 0.71, 0.01)

# ablation_hl_12feat.py 최적 파라미터 (그대로 유지)
BEST_PARAMS = dict(
    iterations=527, learning_rate=0.048239349659174445, depth=4,
    l2_leaf_reg=2.3609821250119802, bagging_temperature=0.2596297886337092,
    random_strength=2.997735571048697, border_count=49,
    loss_function='Logloss', eval_metric='AUC',
    early_stopping_rounds=50, random_seed=SEED,
    verbose=False, allow_writing_files=False,
)

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

X = df.drop(columns=['고혈압유병', '당뇨유병', '이상지질혈증유병', '비만단계'])
y = df[TARGET].astype(int)

# 12개 피처 선택 (ablation 최종)
remove_cols = [
    '직업_관리전문', '직업_기능노무', '직업_농림어업', '직업_무직',
    '직업_사무', '직업_서비스판매', '직업_작업미상', '직업_주부학생',
    '고혈압가족력_부', '고혈압가족력_모', '고혈압가족력_형제',
    '당뇨가족력_부', '당뇨가족력_모', '당뇨가족력_형제',
    '현재흡연', '과거음주_현재금주',
]
X = X.drop(columns=remove_cols)

print(f"[0] 데이터 로드 | shape: {df.shape} | Feature 수: {X.shape[1]}")
print(f"[0] 사용 피처: {list(X.columns)}")
print(f"[0] 클래스 분포:\n{df[TARGET].value_counts().to_string()}")

# ── Train / Test split (Hold-out) ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n[1] Train: {X_train.shape} | Test: {X_test.shape}")
print(f"    Train 양성 비율: {y_train.mean():.4f} | Test 양성 비율: {y_test.mean():.4f}")

# ── Stratified 5-Fold OOF CV (Train 내부) ─────────────────────
skf         = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_proba   = np.zeros(len(y_train))
fold_models = []
fold_scores = []

print(f"\n[2] {N_SPLITS}-Fold OOF CV 시작")
print("=" * 60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    # fold마다 train 기준 class_weights 재계산
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    params = {**BEST_PARAMS, 'class_weights': {0: 1.0, 1: pos_weight}}

    model = CatBoostClassifier(**params)
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val))

    val_proba = model.predict_proba(X_val)[:, 1]
    val_label = (val_proba >= 0.5).astype(int)
    oof_proba[val_idx] = val_proba
    fold_models.append(model)

    auc    = roc_auc_score(y_val, val_proba)
    recall = recall_score(y_val, val_label)
    prec   = precision_score(y_val, val_label, zero_division=0)
    f1     = f1_score(y_val, val_label)

    fold_scores.append({'fold': fold, 'auc': auc, 'recall': recall,
                        'precision': prec, 'f1': f1,
                        'best_iter': model.best_iteration_,
                        'class_weight_pos': round(pos_weight, 4)})
    print(f"  Fold {fold} | AUC: {auc:.4f} | Recall: {recall:.4f} | "
          f"Prec: {prec:.4f} | F1: {f1:.4f} | iter: {model.best_iteration_}")

# ── OOF 전체 성능 (threshold=0.5 기준) ───────────────────────
print("=" * 60)
oof_label_05 = (oof_proba >= 0.5).astype(int)
oof_auc      = roc_auc_score(y_train, oof_proba)
scores_df    = pd.DataFrame(fold_scores)

print("\n[3] OOF 전체 성능 (threshold=0.5)")
print(f"    AUC    : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
print(f"    Recall : {recall_score(y_train, oof_label_05):.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})")
print(f"    F1     : {f1_score(y_train, oof_label_05):.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})")

# ── OOF 기반 Threshold Tuning ─────────────────────────────────
print("\n[4] OOF Threshold Tuning (범위: 0.30~0.70, step=0.01)")
print(f"    기준: Recall >= {RECALL_MIN} 만족 시 Precision 최대")
print("-" * 60)

tuning_rows = []
for thr in THRESHOLD_RANGE:
    pred = (oof_proba >= thr).astype(int)
    r    = recall_score(y_train, pred)
    p    = precision_score(y_train, pred, zero_division=0)
    f    = f1_score(y_train, pred, zero_division=0)
    tuning_rows.append({'threshold': round(thr, 2), 'recall': r, 'precision': p, 'f1': f})

tuning_df  = pd.DataFrame(tuning_rows)
candidates = tuning_df[tuning_df['recall'] >= RECALL_MIN]

if len(candidates) > 0:
    best_row = candidates.loc[candidates['precision'].idxmax()]
    best_thr = best_row['threshold']
    print(f"    ✅ 선택된 threshold: {best_thr:.2f}")
    print(f"       Recall: {best_row['recall']:.4f} | Precision: {best_row['precision']:.4f} | F1: {best_row['f1']:.4f}")
else:
    best_row = tuning_df.loc[tuning_df['recall'].idxmax()]
    best_thr = best_row['threshold']
    print(f"    ⚠️  Recall >= {RECALL_MIN} 만족 threshold 없음 → Recall 최대 threshold 사용: {best_thr:.2f}")
    print(f"       Recall: {best_row['recall']:.4f} | Precision: {best_row['precision']:.4f} | F1: {best_row['f1']:.4f}")

# ── 최종 Hold-out Test 평가 (1회) ─────────────────────────────
print(f"\n[5] 최종 Hold-out Test 평가 (threshold={best_thr:.2f}, 1회 평가)")
print("=" * 60)

test_probas = np.column_stack([
    m.predict_proba(X_test)[:, 1] for m in fold_models
])
test_proba_ensemble = test_probas.mean(axis=1)
test_label          = (test_proba_ensemble >= best_thr).astype(int)

test_auc    = roc_auc_score(y_test, test_proba_ensemble)
test_recall = recall_score(y_test, test_label)
test_prec   = precision_score(y_test, test_label, zero_division=0)
test_f1     = f1_score(y_test, test_label)
cm          = confusion_matrix(y_test, test_label)

print(f"    AUC       : {test_auc:.4f}")
print(f"    Recall    : {test_recall:.4f}")
print(f"    Precision : {test_prec:.4f}")
print(f"    F1        : {test_f1:.4f}")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
print("\n[5] Classification Report")
print(classification_report(y_test, test_label, target_names=['정상(0)', '이상지질혈증(1)']))

# ── Feature Importance (5-fold 평균) ──────────────────────────
print("[6] Feature Importance (5-fold 평균)")
fi_matrix = np.column_stack([m.get_feature_importance() for m in fold_models])
fi_mean   = fi_matrix.mean(axis=1)
fi_df = pd.DataFrame({'feature': X.columns, 'importance': fi_mean}
                     ).sort_values('importance', ascending=False)
print(fi_df.to_string(index=False))

# ── 저장 ─────────────────────────────────────────────────────
scores_df.to_csv(os.path.join(MODEL_DIR, 'fold_scores.csv'), index=False)
fi_df.to_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'), index=False)
tuning_df.to_csv(os.path.join(MODEL_DIR, 'threshold_tuning.csv'), index=False)

np.save(os.path.join(MODEL_DIR, 'oof_y_true.npy'),    y_train.values)
np.save(os.path.join(MODEL_DIR, 'oof_proba.npy'),      oof_proba)
np.save(os.path.join(MODEL_DIR, 'best_threshold.npy'), np.array([best_thr]))

print(f"\n[7] 저장 완료 → {MODEL_DIR}")
