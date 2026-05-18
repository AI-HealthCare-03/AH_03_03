"""
3개 질환 통합 Optuna 튜닝 — 유니크 피처 14개 고정
Python 3.9 | catboost>=1.2 | optuna>=3.0 | scikit-learn>=1.4
질환별 사용 피처: 공통 11개 + 질환별 가족력 3개 = 14개
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

N_SPLITS = 5
N_TRIALS = 100
SEED = 42

# ── 질환별 설정 ───────────────────────────────────────────────
CONFIGS = {
    "고혈압": {
        "target": "고혈압유병",
        "features": [
            "성별",
            "나이",
            "키",
            "체중",
            "BMI",
            "고혈압가족력_부",
            "고혈압가족력_모",
            "고혈압가족력_형제",
            "걷기일수",
            "근력운동일수",
            "음주빈도_enc",
            "음주량_enc",
        ],
        "output_dir": os.path.join(BASE_DIR, "outputs", "tuned_v2_HTN_100"),
    },
    "당뇨": {
        "target": "당뇨유병",
        "features": [
            "성별",
            "나이",
            "키",
            "체중",
            "BMI",
            "당뇨가족력_부",
            "당뇨가족력_모",
            "당뇨가족력_형제",
            "걷기일수",
            "근력운동일수",
            "음주빈도_enc",
            "음주량_enc",
        ],
        "output_dir": os.path.join(BASE_DIR, "outputs", "tuned_v2_DM_100"),
    },
    "이상지질혈증": {
        "target": "이상지질혈증유병",
        "features": [
            "성별",
            "나이",
            "키",
            "체중",
            "BMI",
            "고지혈증가족력_부",
            "고지혈증가족력_모",
            "고지혈증가족력_형제",
            "걷기일수",
            "근력운동일수",
            "음주빈도_enc",
            "음주량_enc",
        ],
        "output_dir": os.path.join(BASE_DIR, "outputs", "tuned_v2_HL_100"),
    },
}

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

all_results = {}

for disease, cfg in CONFIGS.items():
    target = cfg["target"]
    features = cfg["features"]
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    data = df.dropna(subset=[target]).reset_index(drop=True)
    X = data[features]
    y = data[target].astype(int)

    neg, pos = (y == 0).sum(), (y == 1).sum()
    ratio = neg / pos

    print(f"\n{'=' * 60}")
    print(f"[{disease}] 피처 {len(features)}개 | 정상: {neg} / 양성: {pos} | ratio: {ratio:.4f}")
    print(f"Optuna 튜닝 시작 | n_trials: {N_TRIALS}")
    print("=" * 60)

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
            model = CatBoostClassifier(**params)
            model.fit(Pool(X.iloc[tr_idx], y.iloc[tr_idx]), eval_set=Pool(X.iloc[val_idx], y.iloc[val_idx]))
            oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

        return roc_auc_score(y, oof_proba)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n[{disease}] 튜닝 완료 | Best AUC: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # ── 최적 파라미터로 최종 CV ───────────────────────────────
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

    print("\n최적 파라미터로 최종 5-Fold CV")
    print("-" * 55)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model = CatBoostClassifier(**best_params)
        model.fit(Pool(X.iloc[tr_idx], y.iloc[tr_idx]), eval_set=Pool(X.iloc[val_idx], y.iloc[val_idx]))

        val_proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        val_label = (val_proba >= 0.5).astype(int)

        oof_pred_proba[val_idx] = val_proba
        oof_pred_label[val_idx] = val_label

        auc = roc_auc_score(y.iloc[val_idx], val_proba)
        f1 = f1_score(y.iloc[val_idx], val_label)
        recall = recall_score(y.iloc[val_idx], val_label)

        fold_scores.append({"fold": fold, "auc": auc, "f1": f1, "recall": recall, "best_iter": model.best_iteration_})
        print(
            f"  Fold {fold} | AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | best_iter: {model.best_iteration_}"
        )

    scores_df = pd.DataFrame(fold_scores)
    oof_auc = roc_auc_score(y, oof_pred_proba)
    oof_f1 = f1_score(y, oof_pred_label)
    oof_recall = recall_score(y, oof_pred_label)

    print(f"\n  OOF | AUC: {oof_auc:.4f} | F1: {oof_f1:.4f} | Recall: {oof_recall:.4f}")
    print(f"  fold avg | AUC: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f}")

    all_results[disease] = {
        "auc": oof_auc,
        "f1": oof_f1,
        "recall": oof_recall,
        "best_params": study.best_params,
    }

    # 저장
    scores_df.to_csv(os.path.join(output_dir, "fold_scores.csv"), index=False)
    pd.DataFrame([study.best_params]).to_csv(os.path.join(output_dir, "best_params.csv"), index=False)
    study.trials_dataframe().to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)
    np.save(os.path.join(output_dir, "oof_y_true.npy"), y.values)
    np.save(os.path.join(output_dir, "oof_proba.npy"), oof_pred_proba)
    np.save(os.path.join(output_dir, "oof_label.npy"), oof_pred_label)
    print(f"  저장 완료 → {output_dir}")

# ── 전체 요약 ─────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("전체 요약 (threshold=0.5 기준)")
print(f"{'질환':<15} {'AUC':>8} {'Recall':>8} {'F1':>8}")
print("-" * 42)
for disease, res in all_results.items():
    print(f"{disease:<15} {res['auc']:>8.4f} {res['recall']:>8.4f} {res['f1']:>8.4f}")
