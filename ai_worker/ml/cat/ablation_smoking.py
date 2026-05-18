"""
흡연(현재흡연) 추가 Ablation — 고혈압 / 당뇨 / 이상지질혈증
Python 3.9 | catboost>=1.2 | scikit-learn>=1.4
기존 확정 피처셋에 현재흡연 1개만 추가해서 성능 비교
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
SEED      = 42
N_SPLITS  = 5

# ── 질환별 설정 ───────────────────────────────────────────────
CONFIGS = {
    '고혈압': {
        'target':     '고혈압유병',
        'threshold':  0.48,
        'features':   ['성별', '나이', '키', '체중', 'BMI',
                       '고혈압가족력_부', '고혈압가족력_모', '고혈압가족력_형제',
                       '음주빈도_enc', '음주량_enc'],
        'baseline':   {'auc': 0.8580, 'recall': 0.8571, 'f1': 0.6459, 'fn': 249},
        'params': dict(
            iterations=388, learning_rate=0.06174532779902198, depth=4,
            l2_leaf_reg=2.066200719991522, bagging_temperature=0.6005954120104386,
            random_strength=0.16725169601770631, border_count=198,
        ),
    },
    '당뇨': {
        'target':     '당뇨유병',
        'threshold':  0.45,
        'features':   ['성별', '나이', '키', '체중', 'BMI',
                       '당뇨가족력_부', '당뇨가족력_모', '당뇨가족력_형제',
                       '걷기일수', '음주빈도_enc', '음주량_enc'],
        'baseline':   {'auc': 0.8086, 'recall': 0.8767, 'f1': 0.3832, 'fn': 100},
        'params': dict(
            iterations=899, learning_rate=0.07751106059121869, depth=4,
            l2_leaf_reg=4.326173555866765, bagging_temperature=0.6424222998382222,
            random_strength=0.017824706823968917, border_count=69,
        ),
    },
    '이상지질혈증': {
        'target':     '이상지질혈증유병',
        'threshold':  0.48,
        'features':   ['성별', '나이', '키', '체중', 'BMI',
                       '고지혈증가족력_부', '고지혈증가족력_모', '고지혈증가족력_형제',
                       '걷기일수', '근력운동일수', '음주빈도_enc', '음주량_enc'],
        'baseline':   {'auc': 0.7923, 'recall': 0.8774, 'f1': 0.5601, 'fn': 191},
        'params': dict(
            iterations=527, learning_rate=0.048239349659174445, depth=4,
            l2_leaf_reg=2.3609821250119802, bagging_temperature=0.2596297886337092,
            random_strength=2.997735571048697, border_count=49,
        ),
    },
}

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['고혈압유병', '당뇨유병', '이상지질혈증유병']).reset_index(drop=True)

print("=" * 60)
print("흡연(현재흡연) 추가 Ablation — 3개 질환 비교")
print("=" * 60)

for disease, cfg in CONFIGS.items():
    target    = cfg['target']
    threshold = cfg['threshold']
    features  = cfg['features']
    baseline  = cfg['baseline']

    y = df[target].astype(int)
    neg, pos = (y == 0).sum(), (y == 1).sum()

    params = cfg['params'].copy()
    params.update({
        'class_weights':         {0: 1.0, 1: neg / pos},
        'loss_function':         'Logloss',
        'eval_metric':           'AUC',
        'early_stopping_rounds': 50,
        'random_seed':           SEED,
        'verbose':               False,
        'allow_writing_files':   False,
    })

    # 흡연 추가 피처셋
    features_with_smoking = features + ['현재흡연']
    X_slim = df[features_with_smoking]

    skf            = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_pred_proba = np.zeros(len(y))
    oof_pred_label = np.zeros(len(y))

    for tr_idx, val_idx in skf.split(X_slim, y):
        X_tr, X_val = X_slim.iloc[tr_idx], X_slim.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val))

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_pred_proba[val_idx] = val_proba
        oof_pred_label[val_idx] = (val_proba >= threshold).astype(int)

    oof_auc    = roc_auc_score(y, oof_pred_proba)
    oof_f1     = f1_score(y, oof_pred_label)
    oof_recall = recall_score(y, oof_pred_label)
    cm         = confusion_matrix(y, oof_pred_label)

    # Feature Importance
    fi = pd.DataFrame({
        'feature':    features_with_smoking,
        'importance': model.get_feature_importance(),
    }).sort_values('importance', ascending=False)
    smoking_importance = fi[fi['feature'] == '현재흡연']['importance'].values[0]

    print(f"\n[{disease}] 피처 {len(features)}개 → {len(features_with_smoking)}개 (흡연 추가)")
    print(f"    {'구분':<15} {'AUC':>8} {'Recall':>8} {'F1':>8} {'FN':>6}")
    print("    " + "-" * 44)
    print(f"    {'흡연 제외':<15} {baseline['auc']:>8.4f} {baseline['recall']:>8.4f} {baseline['f1']:>8.4f} {baseline['fn']:>6}")
    print(f"    {'흡연 포함':<15} {oof_auc:>8.4f} {oof_recall:>8.4f} {oof_f1:>8.4f} {cm[1,0]:>6}")
    print(f"    변화           {oof_auc-baseline['auc']:>+8.4f} {oof_recall-baseline['recall']:>+8.4f} {oof_f1-baseline['f1']:>+8.4f} {cm[1,0]-baseline['fn']:>+6}")
    print(f"    현재흡연 importance: {smoking_importance:.4f}%")

print("\n" + "=" * 60)
print("완료")
