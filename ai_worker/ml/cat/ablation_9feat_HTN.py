"""
고혈압유병 예측 — 걷기일수 추가 제거 Ablation Test
피처: 28개 → 9개
"""

import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score

warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'hn24_file1_preprocessed.csv')

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

remove_cols = [
    '직업_관리전문', '직업_기능노무', '직업_농림어업', '직업_무직',
    '직업_사무', '직업_서비스판매', '직업_작업미상', '직업_주부학생',
    '당뇨가족력_부', '당뇨가족력_모', '당뇨가족력_형제',
    '고지혈증가족력_부', '고지혈증가족력_모', '고지혈증가족력_형제',
    '현재흡연', '과거음주_현재금주', '키', '근력운동일수',
    '걷기일수',  # 추가 제거
]

X_slim = X.drop(columns=remove_cols)
BEST_PARAMS['class_weights'] = {0: 1.0, 1: (y == 0).sum() / (y == 1).sum()}

print(f"[0] 피처 수: {X.shape[1]}개 → {X_slim.shape[1]}개")
print(f"[0] 사용 피처: {list(X_slim.columns)}")

skf            = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pred_proba = np.zeros(len(y))
fold_scores    = []

print(f"\n[1] {N_SPLITS}-Fold CV")
print("=" * 55)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_slim, y), 1):
    X_tr, X_val = X_slim.iloc[tr_idx], X_slim.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**BEST_PARAMS)
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val))

    val_proba = model.predict_proba(X_val)[:, 1]
    oof_pred_proba[val_idx] = val_proba

    auc    = roc_auc_score(y_val, val_proba)
    f1     = f1_score(y_val, (val_proba >= 0.48).astype(int))
    recall = recall_score(y_val, (val_proba >= 0.48).astype(int))
    fold_scores.append({'fold': fold, 'auc': auc, 'f1': f1, 'recall': recall})
    print(f"  Fold {fold} | AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")

print("=" * 55)
oof_label  = (oof_pred_proba >= 0.48).astype(int)
oof_auc    = roc_auc_score(y, oof_pred_proba)
oof_f1     = f1_score(y, oof_label)
oof_recall = recall_score(y, oof_label)

print(f"\n[2] 전체 단계별 비교 (threshold=0.48)")
print(f"    {'구분':<20} {'피처':>5} {'AUC':>8} {'Recall':>8} {'F1':>8}")
print("    " + "-" * 48)
print(f"    {'베이스라인':<20} {28:>5} {0.8596:>8.4f} {0.8524:>8.4f} {0.6459:>8.4f}")
print(f"    {'직업 제거':<20} {20:>5} {0.8574:>8.4f} {0.8583:>8.4f} {0.6455:>8.4f}")
print(f"    {'관련가족력만':<20} {14:>5} {0.8577:>8.4f} {0.8530:>8.4f} {0.6458:>8.4f}")
print(f"    {'slim 10개':<20} {10:>5} {0.8572:>8.4f} {0.8518:>8.4f} {0.6418:>8.4f}")
print(f"    {'걷기일수 제거 9개':<20} {X_slim.shape[1]:>5} {oof_auc:>8.4f} {oof_recall:>8.4f} {oof_f1:>8.4f}")
