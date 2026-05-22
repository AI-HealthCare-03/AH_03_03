"""
당뇨유병 예측 — CatBoost + Optuna 하이퍼파라미터 튜닝 + FE v5 (hn18~24 통합)
Python 3.13 | catboost>=1.2 | optuna>=3.0 | scikit-learn>=1.4

v3 대비 변경점:
  - class_weight_1 Optuna 탐색 추가 (기존 고정값 3.23 → 1.5~5.0 탐색)

FE 조합: 나이구간 + BMI구간 + 가족력합산 + BMI_X_나이 (v3 확정, 38개)
Primary Objective : Recall >= 0.87 유지
Optimization Target: OOF F1 최대화
Trial 수           : 50
검증 구조          : Train/Test 8:2 Hold-out → Train 내부 5-Fold OOF
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from numpy.typing import NDArray
from optuna.pruners import MedianPruner
from optuna.trial import Trial
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 경로 설정 ─────────────────────────────────────────────────
DATA_PATH: str = str(Path(__file__).parent.parent.parent.parent / "ai_worker" / "data" / "hn_all_preprocessed.csv")
MODEL_DIR: str = str(Path(__file__).parent.parent.parent.parent / "ai_worker" / "ml" / "CAT18~24" / "outputs" / "optuna_DM_FE_v5")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET: str = "당뇨유병"
N_SPLITS: int = 5
SEED: int = 42
N_TRIALS: int = 50
RECALL_MIN: float = 0.87
THRESHOLD_RANGE: NDArray[np.float64] = np.arange(0.30, 0.71, 0.01)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FE 플래그 (v3 확정 조합)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_AGE_BIN: bool = True
USE_BMI_BIN: bool = True
USE_FAMILY_SUM: bool = True
USE_BMI_X_AGE: bool = True


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    added: list[str] = []

    if USE_AGE_BIN:
        age_bins = [0, 40, 50, 60, 70, 80, np.inf]
        age_labels = ["나이_19_39", "나이_40대", "나이_50대", "나이_60대", "나이_70대", "나이_80이상"]
        df["_나이구간"] = pd.cut(df["나이"], bins=age_bins, labels=age_labels, right=False)
        for label in age_labels:
            df[label] = (df["_나이구간"] == label).astype(int)
        df = df.drop(columns=["_나이구간"])
        added += age_labels
        print("  [ON] 나이 구간화")

    if USE_BMI_BIN:
        df["BMI_구간"] = pd.cut(df["BMI"], bins=[0, 23, 25, 30, np.inf], labels=[0, 1, 2, 3], right=False).astype(float)
        added += ["BMI_구간"]
        print("  [ON] BMI 구간화")

    if USE_FAMILY_SUM:
        df["고혈압가족력_합산"] = (
            df["고혈압가족력_부"].fillna(0) + df["고혈압가족력_모"].fillna(0) + df["고혈압가족력_형제"].fillna(0)
        ).clip(0, 3)
        df["당뇨가족력_합산"] = (
            df["당뇨가족력_부"].fillna(0) + df["당뇨가족력_모"].fillna(0) + df["당뇨가족력_형제"].fillna(0)
        ).clip(0, 3)
        df["고지혈증가족력_합산"] = (
            df["고지혈증가족력_부"].fillna(0) + df["고지혈증가족력_모"].fillna(0) + df["고지혈증가족력_형제"].fillna(0)
        ).clip(0, 3)
        added += ["고혈압가족력_합산", "당뇨가족력_합산", "고지혈증가족력_합산"]
        print("  [ON] 가족력 합산")

    if USE_BMI_X_AGE:
        df["BMI_X_나이"] = df["BMI"] * df["나이"]
        added += ["BMI_X_나이"]
        print("  [ON] BMI × 나이")

    print(f"\n[FE] 추가 피처 수: {len(added)} | 목록: {added}")
    return df


def tune_threshold(
    proba: NDArray[np.float64],
    y_true: NDArray[np.int64],
) -> tuple[float, float, float, float]:
    best_f1: float = 0.0
    best_thr: float = 0.30
    for thr in THRESHOLD_RANGE:
        pred = (proba >= thr).astype(int)
        r = recall_score(y_true, pred)
        f = f1_score(y_true, pred, zero_division=0)
        if r >= RECALL_MIN and f > best_f1:
            best_f1 = f
            best_thr = round(float(thr), 2)
    final_pred = (proba >= best_thr).astype(int)
    final_recall = recall_score(y_true, final_pred)
    final_prec = precision_score(y_true, final_pred, zero_division=0)
    return best_thr, best_f1, final_recall, final_prec


def objective(
    trial: Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    # ── v5 핵심: class_weight_1도 Optuna가 탐색 ──────────────
    class_weight_1 = trial.suggest_float("class_weight_1", 1.5, 5.0)

    params: dict[str, Any] = {
        "iterations": trial.suggest_int("iterations", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "class_weights": {0: 1.0, 1: class_weight_1},
        "early_stopping_rounds": 50,
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba: NDArray[np.float64] = np.zeros(len(y_train))
    fold_recalls: list[float] = []

    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        train_pool = Pool(X_tr, y_tr)
        val_pool = Pool(X_val, y_val)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        val_proba: NDArray[np.float64] = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_proba
        fold_recalls.append(float(recall_score(y_val, (val_proba >= 0.5).astype(int))))

    if np.std(fold_recalls) > 0.2:
        raise optuna.exceptions.TrialPruned()

    best_thr, best_f1, oof_recall, oof_prec = tune_threshold(oof_proba, y_train.values)

    if oof_recall < RECALL_MIN:
        return 0.0

    trial.set_user_attr("oof_auc", round(float(roc_auc_score(y_train, oof_proba)), 4))
    trial.set_user_attr("oof_recall", round(oof_recall, 4))
    trial.set_user_attr("oof_precision", round(oof_prec, 4))
    trial.set_user_attr("oof_f1", round(best_f1, 4))
    trial.set_user_attr("best_thr", best_thr)
    trial.set_user_attr("class_weight_1", round(class_weight_1, 4))
    trial.set_user_attr("fold_recall_std", round(float(np.std(fold_recalls)), 4))

    return best_f1


def retrain_with_best_params(
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[NDArray[np.float64], list[CatBoostClassifier], pd.DataFrame]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba: NDArray[np.float64] = np.zeros(len(y_train))
    fold_models: list[CatBoostClassifier] = []
    fold_scores: list[dict[str, Any]] = []

    # class_weight_1은 best_params에서 꺼내서 class_weights로 변환
    class_weight_1 = best_params.pop("class_weight_1")

    params: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "class_weights": {0: 1.0, 1: class_weight_1},
        "early_stopping_rounds": 50,
        "random_seed": SEED,
        "verbose": False,
        "allow_writing_files": False,
        **best_params,
    }

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        train_pool = Pool(X_tr, y_tr)
        val_pool = Pool(X_val, y_val)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        val_proba: NDArray[np.float64] = model.predict_proba(X_val)[:, 1]
        val_label = (val_proba >= 0.5).astype(int)
        oof_proba[val_idx] = val_proba
        fold_models.append(model)

        fold_scores.append(
            {
                "fold": fold,
                "auc": round(float(roc_auc_score(y_val, val_proba)), 4),
                "recall": round(float(recall_score(y_val, val_label)), 4),
                "precision": round(float(precision_score(y_val, val_label, zero_division=0)), 4),
                "f1": round(float(f1_score(y_val, val_label)), 4),
                "best_iter": model.best_iteration_,
            }
        )
        print(
            f"  Fold {fold} | AUC: {fold_scores[-1]['auc']:.4f} | "
            f"Recall: {fold_scores[-1]['recall']:.4f} | "
            f"Prec: {fold_scores[-1]['precision']:.4f} | "
            f"F1: {fold_scores[-1]['f1']:.4f} | "
            f"iter: {model.best_iteration_}"
        )

    return oof_proba, fold_models, pd.DataFrame(fold_scores)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"[0] 데이터 로드 | shape: {df.shape}")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"[0] {TARGET} 결측 제거 후 | shape: {df.shape}")
    print(f"[0] 클래스 분포:\n{df[TARGET].value_counts().to_string()}")

    print("\n[FE] 피처 엔지니어링 시작")
    df = apply_feature_engineering(df)

    drop_cols = [c for c in ["고혈압유병", "당뇨유병", "이상지질혈증유병", "비만단계"] if c in df.columns]
    X: pd.DataFrame = df.drop(columns=drop_cols)
    y: pd.Series = df[TARGET].astype(int)
    print(f"\n[1] Feature 수: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"\n[2] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Train 양성 비율: {y_train.mean():.4f} | Test 양성 비율: {y_test.mean():.4f}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    default_ratio = float(neg / pos)
    print(f"    기본 class_ratio (neg/pos): {default_ratio:.4f} → Optuna 탐색 범위: 1.5 ~ 5.0")

    print(f"\n[3] Optuna 시작 | {N_TRIALS} trials | Recall >= {RECALL_MIN} → F1 최대화")
    print("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=10),
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("=" * 60)
    best = study.best_trial
    print(f"\n[4] Best Trial #{best.number}")
    print(f"    OOF F1           : {best.value:.4f}")
    print(f"    OOF Recall       : {best.user_attrs.get('oof_recall', '-')}")
    print(f"    OOF Precision    : {best.user_attrs.get('oof_precision', '-')}")
    print(f"    OOF AUC          : {best.user_attrs.get('oof_auc', '-')}")
    print(f"    Best Threshold   : {best.user_attrs.get('best_thr', '-')}")
    print(f"    Best class_weight_1: {best.user_attrs.get('class_weight_1', '-')}  (기본값: {default_ratio:.4f})")
    print(f"    Fold Recall std  : {best.user_attrs.get('fold_recall_std', '-')}")
    print("\n    Best Params:")
    for k, v in best.params.items():
        print(f"      {k}: {v}")

    print("\n[5] Best 파라미터로 최종 OOF 재학습")
    print("=" * 60)

    best_params: dict[str, Any] = best.params.copy()
    best_thr: float = float(best.user_attrs.get("best_thr", 0.44))

    oof_proba, fold_models, scores_df = retrain_with_best_params(best_params, X_train, y_train)

    print("=" * 60)
    oof_label_05 = (oof_proba >= 0.5).astype(int)
    oof_auc = float(roc_auc_score(y_train, oof_proba))

    print("\n[6] OOF 전체 성능 (threshold=0.5)")
    print(f"    AUC    : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
    print(
        f"    Recall : {recall_score(y_train, oof_label_05):.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})"
    )
    print(
        f"    F1     : {f1_score(y_train, oof_label_05):.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})"
    )

    print(f"\n[7] OOF Threshold 재확인 (best_thr={best_thr:.2f})")
    tuning_rows: list[dict[str, float]] = []
    for thr in THRESHOLD_RANGE:
        pred = (oof_proba >= thr).astype(int)
        r = recall_score(y_train, pred)
        p = precision_score(y_train, pred, zero_division=0)
        f = f1_score(y_train, pred, zero_division=0)
        tuning_rows.append({"threshold": round(float(thr), 2), "recall": r, "precision": p, "f1": f})
    tuning_df = pd.DataFrame(tuning_rows)
    final_row = tuning_df[tuning_df["threshold"] == best_thr].iloc[0]
    print(
        f"    Recall: {final_row['recall']:.4f} | Precision: {final_row['precision']:.4f} | F1: {final_row['f1']:.4f}"
    )

    print(f"\n[8] 최종 Hold-out Test 평가 (threshold={best_thr:.2f})")
    print("=" * 60)

    test_probas = np.column_stack([m.predict_proba(X_test)[:, 1] for m in fold_models])
    test_proba_ens: NDArray[np.float64] = test_probas.mean(axis=1)
    test_label = (test_proba_ens >= best_thr).astype(int)

    test_auc = float(roc_auc_score(y_test, test_proba_ens))
    test_recall = float(recall_score(y_test, test_label))
    test_prec = float(precision_score(y_test, test_label, zero_division=0))
    test_f1 = float(f1_score(y_test, test_label))
    cm = confusion_matrix(y_test, test_label)

    print(f"    AUC       : {test_auc:.4f}")
    print(f"    Recall    : {test_recall:.4f}")
    print(f"    Precision : {test_prec:.4f}")
    print(f"    F1        : {test_f1:.4f}")
    print(f"    TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"    FN={cm[1, 0]}  TP={cm[1, 1]}")
    print("\n[8] Classification Report")
    print(classification_report(y_test, test_label, target_names=["정상(0)", "당뇨(1)"]))

    print("[9] Feature Importance Top 20 (마지막 fold)")
    fi_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": fold_models[-1].get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    print(fi_df.head(20).to_string(index=False))

    scores_df.to_csv(os.path.join(MODEL_DIR, "fold_scores.csv"), index=False)
    fi_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    tuning_df.to_csv(os.path.join(MODEL_DIR, "threshold_tuning.csv"), index=False)
    np.save(os.path.join(MODEL_DIR, "oof_y_true.npy"), y_train.values)
    np.save(os.path.join(MODEL_DIR, "oof_proba.npy"), oof_proba)
    np.save(os.path.join(MODEL_DIR, "best_threshold.npy"), np.array([best_thr]))

    with open(os.path.join(MODEL_DIR, "best_params.json"), "w") as f:
        json.dump(best.params, f, indent=2, ensure_ascii=False)

    study.trials_dataframe().to_csv(os.path.join(MODEL_DIR, "optuna_trials.csv"), index=False)

    print(f"\n[10] 저장 완료 → {MODEL_DIR}")


if __name__ == "__main__":
    main()
