"""
이상지질혈증유병 예측 — CatBoost Optuna 하이퍼파라미터 튜닝
Python 3.9 | catboost>=1.2 | optuna>=3.0 | scikit-learn>=1.4
최적화 목표: OOF AUC-ROC 최대화
피처: 직업 8개 + 고혈압가족력 3개 + 당뇨가족력 3개 제거 (14개)
"""

import os
import warnings

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 경로 설정 ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "hn24_file1_preprocessed.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "tuned_catboost_HL_100")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET = "이상지질혈증유병"
N_SPLITS = 5
N_TRIALS = 100
SEED = 42

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)

drop_cols = ["고혈압유병", "당뇨유병", "이상지질혈증유병", "비만단계"]
X = df.drop(columns=drop_cols)
y = df[TARGET].astype(int)

# ── 피처 정리 ─────────────────────────────────────────────────
remove_cols = [
    "직업_관리전문",
    "직업_기능노무",
    "직업_농림어업",
    "직업_무직",
    "직업_사무",
    "직업_서비스판매",
    "직업_작업미상",
    "직업_주부학생",
    "고혈압가족력_부",
    "고혈압가족력_모",
    "고혈압가족력_형제",
    "당뇨가족력_부",
    "당뇨가족력_모",
    "당뇨가족력_형제",
]
X = X.drop(columns=remove_cols)

neg, pos = (y == 0).sum(), (y == 1).sum()
ratio = neg / pos

print(f"[0] 데이터 로드 완료 | shape: {X.shape}")
print(f"[0] 클래스 불균형 | 정상: {neg} / 이상지질혈증: {pos} | ratio: {ratio:.4f}")
print(f"[0] 사용 피처 ({X.shape[1]}개): {list(X.columns)}")
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
print("    Best params:")
for k, v in study.best_params.items():
    print(f"      {k}: {v}")


# ── 최적 파라미터로 최종 5-Fold CV ────────────────────────────
print("\n[3] 최적 파라미터로 최종 5-Fold CV")
print("=" * 55)

best_params = study.best_params.copy()
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
cm = confusion_matrix(y, oof_pred_label)

print("\n[4] OOF 최종 성능 (튜닝 후)")
print(f"    AUC-ROC : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
print(f"    F1      : {oof_f1:.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})")
print(f"    Recall  : {oof_recall:.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})")

# ── 베이스라인 대비 비교 ──────────────────────────────────────
print("\n[5] 단계별 비교")
print(f"    {'구분':<20} {'AUC':>8} {'Recall':>8} {'F1':>8} {'FN':>6}")
print("    " + "-" * 50)
print(f"    {'베이스라인 (28개)':<20} {0.8026:>8.4f} {0.8261:>8.4f} {0.5709:>8.4f} {271:>6}")
print(f"    {'피처정리 (14개)':<20} {0.7893:>8.4f} {0.8395:>8.4f} {0.5689:>8.4f} {250:>6}")
print(f"    {'튜닝 후 (14개)':<20} {oof_auc:>8.4f} {oof_recall:>8.4f} {oof_f1:>8.4f} {cm[1, 0]:>6}")

# ── Feature Importance ────────────────────────────────────────
print("\n[6] Feature Importance (마지막 fold)")
fi = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.get_feature_importance(),
    }
).sort_values("importance", ascending=False)
print(fi.to_string(index=False))

# ── 저장 ──────────────────────────────────────────────────────
scores_df.to_csv(os.path.join(OUTPUT_DIR, "fold_scores.csv"), index=False)
fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
pd.DataFrame([study.best_params]).to_csv(os.path.join(OUTPUT_DIR, "best_params.csv"), index=False)
study.trials_dataframe().to_csv(os.path.join(OUTPUT_DIR, "optuna_trials.csv"), index=False)
np.save(os.path.join(OUTPUT_DIR, "oof_y_true.npy"), y.values)
np.save(os.path.join(OUTPUT_DIR, "oof_proba.npy"), oof_pred_proba)
np.save(os.path.join(OUTPUT_DIR, "oof_label.npy"), oof_pred_label)

print(f"\n[7] 저장 완료 → {OUTPUT_DIR}")
