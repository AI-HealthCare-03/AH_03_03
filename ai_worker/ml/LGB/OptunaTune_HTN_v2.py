"""
고혈압유병 예측 — Optuna 하이퍼파라미터 튜닝
Python 3.13 | lightgbm>=4.0 | optuna>=3.0 | scikit-learn>=1.4

피처: 20개 (직업 OHE 8개 제거)
Primary Objective : Recall >= 0.85 유지
Optimization Target: OOF Precision 최대화
Monitoring         : F1, ROC-AUC 함께 기록
Trial 수           : 50
검증 구조          : Train/Test 8:2 Hold-out → Train 내부 5-Fold OOF
"""

import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'hn24_file1_preprocessed.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'optuna_HTN_v2')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET     = '고혈압유병'
N_SPLITS   = 5
SEED       = 42
N_TRIALS   = 50
RECALL_MIN = 0.85
THRESHOLD_RANGE = np.arange(0.30, 0.71, 0.01)

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

drop_cols = ['고혈압유병', '당뇨유병', '이상지질혈증유병', '비만단계']
X = df.drop(columns=drop_cols)
y = df[TARGET].astype(int)

# ── 직업 OHE 피처 제거 (v2) ──────────────────────────────────
remove_cols = [
    '직업_관리전문', '직업_기능노무', '직업_농림어업', '직업_무직',
    '직업_사무', '직업_서비스판매', '직업_작업미상', '직업_주부학생',
]
X = X.drop(columns=remove_cols)

print(f"[0] 데이터 로드 | shape: {df.shape} | Feature 수: {X.shape[1]}")
print(f"[0] 클래스 분포:\n{df[TARGET].value_counts().to_string()}")

# ── Train / Test split (Hold-out) ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n[1] Train: {X_train.shape} | Test: {X_test.shape}")

# ── Objective 함수 ────────────────────────────────────────────
def objective(trial):
    params = {
        'objective':         'binary',
        'metric':            'auc',
        'boosting_type':     'gbdt',
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 16, 64),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state':      SEED,
        'n_jobs':            -1,
        'verbose':           -1,
    }

    skf       = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba = np.zeros(len(y_train))
    fold_recalls = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        cw  = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
        spw = cw[1] / cw[0]
        params['scale_pos_weight'] = spw

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_proba
        fold_recalls.append(recall_score(y_val, (val_proba >= 0.5).astype(int)))

    # Fold Recall 편차 너무 크면 trial 무효 (DM 불안정 방지용)
    if np.std(fold_recalls) > 0.2:
        raise optuna.exceptions.TrialPruned()

    # OOF 기반 threshold tuning
    best_prec = 0.0
    best_thr  = 0.30
    for thr in THRESHOLD_RANGE:
        pred = (oof_proba >= thr).astype(int)
        r    = recall_score(y_train, pred)
        p    = precision_score(y_train, pred, zero_division=0)
        if r >= RECALL_MIN and p > best_prec:
            best_prec = p
            best_thr  = round(thr, 2)

    # Recall 조건 미달 시 패널티
    oof_recall = recall_score(y_train, (oof_proba >= best_thr).astype(int))
    if oof_recall < RECALL_MIN:
        return 0.0

    # 부가 지표 기록
    trial.set_user_attr('oof_auc',       round(roc_auc_score(y_train, oof_proba), 4))
    trial.set_user_attr('oof_recall',    round(oof_recall, 4))
    trial.set_user_attr('oof_f1',        round(f1_score(y_train, (oof_proba >= best_thr).astype(int)), 4))
    trial.set_user_attr('best_thr',      best_thr)
    trial.set_user_attr('fold_recall_std', round(np.std(fold_recalls), 4))

    return best_prec  # Precision 최대화


# ── Optuna Study ──────────────────────────────────────────────
print(f"\n[2] Optuna 시작 | {N_TRIALS} trials | Recall >= {RECALL_MIN} → Precision 최대")
print("=" * 60)

study = optuna.create_study(
    direction = 'maximize',
    pruner    = MedianPruner(n_startup_trials=10),
    sampler   = optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ── Best Trial 결과 ───────────────────────────────────────────
print("=" * 60)
best = study.best_trial
print(f"\n[3] Best Trial #{best.number}")
print(f"    OOF Precision : {best.value:.4f}")
print(f"    OOF Recall    : {best.user_attrs.get('oof_recall', '-')}")
print(f"    OOF AUC       : {best.user_attrs.get('oof_auc', '-')}")
print(f"    OOF F1        : {best.user_attrs.get('oof_f1', '-')}")
print(f"    Best Threshold: {best.user_attrs.get('best_thr', '-')}")
print(f"    Fold Recall std: {best.user_attrs.get('fold_recall_std', '-')}")
print("\n    Best Params:")
for k, v in best.params.items():
    print(f"      {k}: {v}")

# ── Best 파라미터로 최종 OOF 재학습 ──────────────────────────
print("\n[4] Best 파라미터로 최종 OOF 재학습")
print("=" * 60)

best_params = best.params.copy()
best_thr    = best.user_attrs.get('best_thr', 0.44)

skf         = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_proba   = np.zeros(len(y_train))
fold_models = []
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    cw  = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_tr)
    spw = cw[1] / cw[0]

    params = {
        'objective':        'binary',
        'metric':           'auc',
        'boosting_type':    'gbdt',
        'scale_pos_weight': spw,
        'random_state':     SEED,
        'n_jobs':           -1,
        'verbose':          -1,
        **best_params,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

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
                        'scale_pos_weight': round(spw, 4)})
    print(f"  Fold {fold} | AUC: {auc:.4f} | Recall: {recall:.4f} | "
          f"Prec: {prec:.4f} | F1: {f1:.4f} | iter: {model.best_iteration_}")

# ── OOF 전체 성능 ─────────────────────────────────────────────
print("=" * 60)
oof_label_05 = (oof_proba >= 0.5).astype(int)
oof_auc      = roc_auc_score(y_train, oof_proba)
scores_df    = pd.DataFrame(fold_scores)

print("\n[5] OOF 전체 성능 (threshold=0.5)")
print(f"    AUC    : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
print(f"    Recall : {recall_score(y_train, oof_label_05):.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})")
print(f"    F1     : {f1_score(y_train, oof_label_05):.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})")

# ── OOF Threshold 재확인 ──────────────────────────────────────
print(f"\n[6] OOF Threshold 재확인 (best_thr={best_thr:.2f})")
tuning_rows = []
for thr in THRESHOLD_RANGE:
    pred = (oof_proba >= thr).astype(int)
    r    = recall_score(y_train, pred)
    p    = precision_score(y_train, pred, zero_division=0)
    f    = f1_score(y_train, pred, zero_division=0)
    tuning_rows.append({'threshold': round(thr, 2), 'recall': r, 'precision': p, 'f1': f})
tuning_df = pd.DataFrame(tuning_rows)

final_row = tuning_df[tuning_df['threshold'] == best_thr].iloc[0]
print(f"    Recall: {final_row['recall']:.4f} | Precision: {final_row['precision']:.4f} | F1: {final_row['f1']:.4f}")

# ── 최종 Hold-out Test 평가 (1회) ─────────────────────────────
print(f"\n[7] 최종 Hold-out Test 평가 (threshold={best_thr:.2f}, 1회 평가)")
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
print("\n[7] Classification Report")
print(classification_report(y_test, test_label, target_names=['정상(0)', '고혈압(1)']))

# ── Feature Importance (fold 평균) ────────────────────────────
print("[8] Feature Importance Top 15 (gain 평균, 5-fold)")
fi_matrix = np.column_stack([m.feature_importances_ for m in fold_models])
fi_mean   = fi_matrix.mean(axis=1)
fi_df = pd.DataFrame({'feature': X.columns, 'importance': fi_mean}
                     ).sort_values('importance', ascending=False)
print(fi_df.head(15).to_string(index=False))

# ── 저장 ─────────────────────────────────────────────────────
scores_df.to_csv(os.path.join(MODEL_DIR, 'fold_scores.csv'), index=False)
fi_df.to_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'), index=False)
tuning_df.to_csv(os.path.join(MODEL_DIR, 'threshold_tuning.csv'), index=False)

np.save(os.path.join(MODEL_DIR, 'oof_y_true.npy'),    y_train.values)
np.save(os.path.join(MODEL_DIR, 'oof_proba.npy'),      oof_proba)
np.save(os.path.join(MODEL_DIR, 'best_threshold.npy'), np.array([best_thr]))

# best params JSON 저장
with open(os.path.join(MODEL_DIR, 'best_params.json'), 'w') as f:
    json.dump(best_params, f, indent=2, ensure_ascii=False)

# trial 전체 결과 저장
trials_df = study.trials_dataframe()
trials_df.to_csv(os.path.join(MODEL_DIR, 'optuna_trials.csv'), index=False)

# 실험 로그
try:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from db_logger import log_experiment
    log_experiment(
        model_name      = 'hypertension_model',
        version         = 'v2.0-optuna-lgbm',
        feature_columns = list(X.columns),
        oof_auc         = oof_auc,
        oof_f1          = f1_score(y_train, oof_label_05),
        oof_recall      = recall_score(y_train, oof_label_05),
        fold_scores     = scores_df.to_dict('records'),
        best_params     = best_params,
        file_path       = MODEL_DIR,
    )
except Exception as e:
    print(f"[주의] DB 로그 저장 실패 (실험 결과에 영향 없음): {e}")

print(f"\n[9] 저장 완료 → {MODEL_DIR}")
