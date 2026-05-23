"""
고혈압유병 예측 — CatBoost Optuna 하이퍼파라미터 튜닝
Python 3.9 | catboost>=1.2 | optuna>=3.0 | scikit-learn>=1.4
최적화 목표: OOF AUC-ROC 최대화
"""

import os
import warnings

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "hn24_file1_preprocessed.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "tuned_catboost_HTN_50")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET = "고혈압유병"
N_SPLITS = 5
N_TRIALS = 50  # 시간 여유 있으면 100으로 늘려도 됨
SEED = 42

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

drop_cols = ["고혈압유병", "당뇨유병", "이상지질혈증유병", "비만단계"]
X = df.drop(columns=drop_cols)
y = df[TARGET].astype(int)

neg, pos = (y == 0).sum(), (y == 1).sum()
ratio = neg / pos

print(f"[0] 데이터 로드 완료 | shape: {X.shape}")
print(f"[0] 클래스 불균형 | 음성: {neg} / 양성: {pos} | ratio: {ratio:.4f}")
print(f"[1] Optuna 튜닝 시작 | n_trials: {N_TRIALS}")
print("=" * 55)


# ── Objective 함수 ────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = dict(
        iterations=trial.suggest_int("iterations", 200, 1000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        border_count=trial.suggest_int("border_count", 32, 255),
        class_weights={0: 1.0, 1: ratio},
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=50,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba = np.zeros(len(y))

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val))
        oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]

    return roc_auc_score(y, oof_proba)


# ── Optuna 실행 ───────────────────────────────────────────────
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n[2] 튜닝 완료")
print(f"    Best AUC : {study.best_value:.4f}")
print("    Best params:\n")
for k, v in study.best_params.items():
    print(f"      {k}: {v}")


# ── 최적 파라미터로 최종 학습 및 OOF 평가 ─────────────────────
print("\n[3] 최적 파라미터로 최종 5-Fold CV")
print("=" * 55)

best_params = study.best_params
best_params.update(
    {
        "class_weights": {0: 1.0, 1: ratio},
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
    }
)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pred_proba = np.zeros(len(y))
oof_pred_label = np.zeros(len(y))
fold_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**best_params)
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val))

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
scores_df = pd.DataFrame(fold_scores)
oof_auc = roc_auc_score(y, oof_pred_proba)
oof_f1 = f1_score(y, oof_pred_label)
oof_recall = recall_score(y, oof_pred_label)

print("\n[4] OOF 최종 성능 (튜닝 후)")
print(f"    AUC-ROC : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
print(f"    F1      : {oof_f1:.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})")
print(f"    Recall  : {oof_recall:.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})")

# ── 베이스라인 대비 비교 ──────────────────────────────────────
print("\n[5] 베이스라인 vs 튜닝 후")
print(f"    {'지표':<10} {'베이스라인':>12} {'튜닝 후':>10} {'변화':>8}")
print(f"    {'AUC-ROC':<10} {0.8585:>12.4f} {oof_auc:>10.4f} {oof_auc - 0.8585:>+8.4f}")
print(f"    {'F1':<10} {0.6472:>12.4f} {oof_f1:>10.4f} {oof_f1 - 0.6472:>+8.4f}")
print(f"    {'Recall':<10} {0.8298:>12.4f} {oof_recall:>10.4f} {oof_recall - 0.8298:>+8.4f}")

# ── 결과 저장 ─────────────────────────────────────────────────
scores_df.to_csv(os.path.join(OUTPUT_DIR, "fold_scores.csv"), index=False)

# OOF npy 저장 (앙상블 시 csv보다 빠름)
np.save(os.path.join(OUTPUT_DIR, "oof_y_true.npy"), y.values)
np.save(os.path.join(OUTPUT_DIR, "oof_proba.npy"), oof_pred_proba)
np.save(os.path.join(OUTPUT_DIR, "oof_label.npy"), oof_pred_label)

# best params 저장
pd.DataFrame([study.best_params]).to_csv(os.path.join(OUTPUT_DIR, "best_params.csv"), index=False)

# optuna trial 히스토리 저장
trials_df = study.trials_dataframe()
trials_df.to_csv(os.path.join(OUTPUT_DIR, "optuna_trials.csv"), index=False)


# ── 실험 로그 저장 (model_versions) ──────────────────────────
try:
    import os as _os
    import sys

    sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    from db_logger import log_experiment

    log_experiment(
        model_name="hypertension_model",
        version="v1.1-tuned-catboost",
        feature_columns=list(X.columns),
        oof_auc=oof_auc,
        oof_f1=oof_f1,
        oof_recall=oof_recall,
        fold_scores=scores_df.to_dict("records"),
        best_params=study.best_params,
        file_path=OUTPUT_DIR,
    )
except Exception as e:
    print(f"[주의] DB 로그 저장 실패 (실험 결과에 영향 없음): {e}")

print(f"\n[6] 저장 완료 → {OUTPUT_DIR}")
