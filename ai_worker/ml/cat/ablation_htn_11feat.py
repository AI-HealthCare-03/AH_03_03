"""
고혈압유병 예측 — 걷기일수 + 근력운동일수 추가 Ablation
피처: 9개 → 11개
"""

import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix

warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'hn24_file1_preprocessed.csv')

TARGET   = '고혈압유병'
N_SPLITS = 5
SEED     = 42

BEST_PARAMS = dict(
    iterations=388, learning_rate=0.06174532779902198, depth=4,
    l2_leaf_reg=2.066200719991522, bagging_temperature=0.6005954120104386,
    random_strength=0.16725169601770631, border_count=198,
    loss_function='Logloss', eval_metric='AUC',
    early_stopping_rounds=50, random_seed=SEED,
    verbose=False, allow_writing_files=False,
)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
X  = df.drop(columns=['고혈압유병', '당뇨유병', '이상지질혈증유병', '비만단계'])
y  = df[TARGET].astype(int)

# 9개 확정 피처 + 걷기일수 + 근력운동일수
features = [
    '성별', '나이', '키', '체중', 'BMI',
    '고혈압가족력_부', '고혈압가족력_모', '고혈압가족력_형제',
    '음주빈도_enc', '음주량_enc',
    '걷기일수', '근력운동일수',  # 추가
]

X_slim = df[features]
BEST_PARAMS['class_weights'] = {0: 1.0, 1: (y==0).sum() / (y==1).sum()}

print(f"[0] 피처 수: 9개 → {len(features)}개")
print(f"[0] 사용 피처: {features}")

skf            = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pred_proba = np.zeros(len(y))
oof_pred_label = np.zeros(len(y))

print(f"\n[1] {N_SPLITS}-Fold CV")
print("=" * 55)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_slim, y), 1):
    X_tr, X_val = X_slim.iloc[tr_idx], X_slim.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**BEST_PARAMS)
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val))

    val_proba = model.predict_proba(X_val)[:, 1]
    oof_pred_proba[val_idx] = val_proba
    oof_pred_label[val_idx] = (val_proba >= 0.48).astype(int)

    auc    = roc_auc_score(y_val, val_proba)
    f1     = f1_score(y_val, oof_pred_label[val_idx])
    recall = recall_score(y_val, oof_pred_label[val_idx])
    print(f"  Fold {fold} | AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")

print("=" * 55)
oof_auc    = roc_auc_score(y, oof_pred_proba)
oof_f1     = f1_score(y, oof_pred_label)
oof_recall = recall_score(y, oof_pred_label)
cm         = confusion_matrix(y, oof_pred_label)

print(f"\n[2] 비교 (threshold=0.48)")
print(f"    {'구분':<20} {'피처':>5} {'AUC':>8} {'Recall':>8} {'F1':>8} {'FN':>6}")
print("    " + "-" * 54)
print(f"    {'확정 (9개)':<20} {9:>5} {0.8580:>8.4f} {0.8571:>8.4f} {0.6459:>8.4f} {249:>6}")
print(f"    {'걷기+근력 추가 (11개)':<20} {len(features):>5} {oof_auc:>8.4f} {oof_recall:>8.4f} {oof_f1:>8.4f} {cm[1,0]:>6}")
print(f"    변화                 {'':>5} {oof_auc-0.8580:>+8.4f} {oof_recall-0.8571:>+8.4f} {oof_f1-0.6459:>+8.4f} {cm[1,0]-249:>+6}")

print(f"\n[3] Feature Importance")
fi = pd.DataFrame({
    'feature':    features,
    'importance': model.get_feature_importance(),
}).sort_values('importance', ascending=False)
print(fi.to_string(index=False))
