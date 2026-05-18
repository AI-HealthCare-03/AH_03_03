"""
이상지질혈증유병 예측 — 근력운동일수 추가 제거 Ablation
피처: 12개 → 11개
"""

import os
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'hn24_file1_preprocessed.csv')

TARGET   = '이상지질혈증유병'
N_SPLITS = 5
SEED     = 42

BEST_PARAMS = dict(
    iterations=527, learning_rate=0.048239349659174445, depth=4,
    l2_leaf_reg=2.3609821250119802, bagging_temperature=0.2596297886337092,
    random_strength=2.997735571048697, border_count=49,
    loss_function='Logloss', eval_metric='AUC',
    early_stopping_rounds=50, random_seed=SEED,
    verbose=False, allow_writing_files=False,
)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
X  = df.drop(columns=['고혈압유병', '당뇨유병', '이상지질혈증유병', '비만단계'])
y  = df[TARGET].astype(int)

remove_cols = [
    '직업_관리전문', '직업_기능노무', '직업_농림어업', '직업_무직',
    '직업_사무', '직업_서비스판매', '직업_작업미상', '직업_주부학생',
    '고혈압가족력_부', '고혈압가족력_모', '고혈압가족력_형제',
    '당뇨가족력_부', '당뇨가족력_모', '당뇨가족력_형제',
    '현재흡연', '과거음주_현재금주',
    '근력운동일수',  # 추가 제거
]

X_slim = X.drop(columns=remove_cols)
BEST_PARAMS['class_weights'] = {0: 1.0, 1: (y==0).sum() / (y==1).sum()}

print(f"[0] 피처 수: {X.shape[1]}개 → {X_slim.shape[1]}개")
print(f"[0] 사용 피처: {list(X_slim.columns)}")

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

print("\n[2] 단계별 비교 (threshold=0.48)")
print(f"    {'구분':<22} {'피처':>5} {'AUC':>8} {'Recall':>8} {'F1':>8} {'FN':>6}")
print("    " + "-" * 56)
print(f"    {'튜닝+threshold':<22} {14:>5} {0.8035:>8.4f} {0.8588:>8.4f} {0.5705:>8.4f} {220:>6}")
print(f"    {'흡연/음주 제거':<22} {12:>5} {0.7923:>8.4f} {0.8774:>8.4f} {0.5601:>8.4f} {191:>6}")
print(f"    {'근력운동 제거':<22} {X_slim.shape[1]:>5} {oof_auc:>8.4f} {oof_recall:>8.4f} {oof_f1:>8.4f} {cm[1,0]:>6}")

print("\n[3] Feature Importance")
fi = pd.DataFrame({
    'feature':    X_slim.columns,
    'importance': model.get_feature_importance(),
}).sort_values('importance', ascending=False)
print(fi.to_string(index=False))
