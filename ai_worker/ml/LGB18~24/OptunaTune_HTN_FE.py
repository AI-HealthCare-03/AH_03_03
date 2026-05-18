"""
고혈압유병 예측 — Optuna 하이퍼파라미터 튜닝 + 피처 엔지니어링 (hn18~24 통합)
Python 3.13 | lightgbm>=4.0 | optuna>=3.0 | scikit-learn>=1.4

FE 블록을 True/False로 켜고 끄면서 ablation 가능
Primary Objective : Recall >= 0.85 유지
Optimization Target: OOF Precision 최대화
Monitoring         : F1, ROC-AUC 함께 기록
Trial 수           : 50
검증 구조          : Train/Test 8:2 Hold-out → Train 내부 5-Fold OOF
"""

import json
import os
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
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
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 경로 설정 ─────────────────────────────────────────────────
DATA_PATH: str = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/data/hn_all_preprocessed.csv"
MODEL_DIR: str = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml/LGB18~24/outputs/optuna_HTN_FE"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET: str = "고혈압유병"
N_SPLITS: int = 5
SEED: int = 42
N_TRIALS: int = 50
RECALL_MIN: float = 0.85
THRESHOLD_RANGE: NDArray[np.float64] = np.arange(0.30, 0.71, 0.01)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 피처 엔지니어링 ON/OFF 스위치 (True=사용 / False=미사용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_AGE_BIN: bool = True    # 나이 구간화 (19~39/40대/50대/60대/70대/80+)
USE_BMI_BIN: bool = False   # BMI 구간화 (한국인 기준)
USE_WEIGHT_BIN: bool = False  # 체중 구간화
USE_ALCOHOL_RISK: bool = True  # 음주위험군 (비음주/저위험/고위험)
USE_WALK_LEVEL: bool = True  # 걷기 활동량
USE_STRENGTH: bool = False  # 근력운동 활동량
USE_FAMILY_SUM: bool = True  # 가족력 합산 스코어
USE_BMI_X_AGE: bool = True  # BMI × 나이 상호작용
USE_OBESITY_FLAG: bool = False  # 비만여부 (BMI >= 25)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """FE 블록 적용 함수. ON/OFF 스위치에 따라 피처 추가."""
    added: list[str] = []

    # 1. 나이 구간화
    if USE_AGE_BIN:
        age_bins: list[int] = [0, 40, 50, 60, 70, 80, 999]
        age_labels: list[str] = ["나이_19_39", "나이_40대", "나이_50대", "나이_60대", "나이_70대", "나이_80이상"]
        df["_나이구간"] = pd.cut(df["나이"], bins=age_bins, labels=age_labels, right=False)
        for label in age_labels:
            df[label] = (df["_나이구간"] == label).astype(int)
        df = df.drop(columns=["_나이구간"])
        added += age_labels
        print(f"  [ON] 나이 구간화: {age_labels}")

    # 2. BMI 구간화
    if USE_BMI_BIN:
        df["BMI_구간"] = pd.cut(
            df["BMI"], bins=[0, 23, 25, 30, 999], labels=[0, 1, 2, 3], right=False
        ).astype(float)
        added += ["BMI_구간"]
        print("  [ON] BMI 구간화: 0=정상/1=과체중/2=비만1/3=비만2")

    # 3. 체중 구간화
    if USE_WEIGHT_BIN:
        wt_bins: list[int] = [0, 50, 70, 90, 999]
        wt_labels: list[str] = ["체중_저체중", "체중_정상", "체중_과체중", "체중_비만"]
        df["_체중구간"] = pd.cut(df["체중"], bins=wt_bins, labels=wt_labels, right=False)
        for label in wt_labels:
            df[label] = (df["_체중구간"] == label).astype(float)
        df = df.drop(columns=["_체중구간"])
        added += wt_labels
        print(f"  [ON] 체중 구간화: {wt_labels}")

    # 4. 음주위험군
    if USE_ALCOHOL_RISK:
        df["음주위험군"] = pd.cut(
            df["음주빈도"], bins=[-1, 0, 2, 99], labels=[0, 1, 2], right=True
        ).astype(float)
        added += ["음주위험군"]
        print("  [ON] 음주위험군: 0=비음주/1=저위험(월1회이하)/2=고위험(월2회이상)")

    # 5. 걷기 활동량
    if USE_WALK_LEVEL:
        df["걷기활동량"] = pd.cut(
            df["걷기일수"], bins=[-1, 0, 3, 99], labels=[0, 1, 2], right=True
        ).astype(float)
        added += ["걷기활동량"]
        print("  [ON] 걷기활동량: 0=비활동/1=저활동(1~3일)/2=활동(4일이상)")

    # 6. 근력운동 활동량
    if USE_STRENGTH:
        df["근력활동량"] = pd.cut(
            df["근력운동일수"], bins=[-1, 0, 2, 99], labels=[0, 1, 2], right=True
        ).astype(float)
        added += ["근력활동량"]
        print("  [ON] 근력활동량: 0=비활동/1=저활동(1~2일)/2=활동(3일이상)")

    # 7. 가족력 합산
    if USE_FAMILY_SUM:
        df["고혈압가족력_합산"] = (
            df["고혈압가족력_부"].fillna(0)
            + df["고혈압가족력_모"].fillna(0)
            + df["고혈압가족력_형제"].fillna(0)
        ).clip(0, 3)
        df["당뇨가족력_합산"] = (
            df["당뇨가족력_부"].fillna(0)
            + df["당뇨가족력_모"].fillna(0)
            + df["당뇨가족력_형제"].fillna(0)
        ).clip(0, 3)
        df["고지혈증가족력_합산"] = (
            df["고지혈증가족력_부"].fillna(0)
            + df["고지혈증가족력_모"].fillna(0)
            + df["고지혈증가족력_형제"].fillna(0)
        ).clip(0, 3)
        added += ["고혈압가족력_합산", "당뇨가족력_합산", "고지혈증가족력_합산"]
        print("  [ON] 가족력 합산 스코어 (0~3)")

    # 8. BMI × 나이 상호작용
    if USE_BMI_X_AGE:
        df["BMI_X_나이"] = df["BMI"] * df["나이"]
        added += ["BMI_X_나이"]
        print("  [ON] BMI × 나이 상호작용")

    # 9. 비만여부
    if USE_OBESITY_FLAG:
        df["비만여부"] = (df["BMI"] >= 25).astype(float)
        df.loc[df["BMI"].isna(), "비만여부"] = np.nan
        added += ["비만여부"]
        print("  [ON] 비만여부 (BMI >= 25)")

    print(f"\n[FE] 추가된 피처 수: {len(added)}")
    print(f"[FE] 추가 피처: {added}")
    return df


def tune_threshold(
    oof_proba: NDArray[np.float64],
    y_true: NDArray[np.int64],
    recall_min: float = RECALL_MIN,
) -> tuple[float, float, float, float]:
    """OOF 기반 threshold tuning. Recall >= recall_min 만족 시 Precision 최대."""
    best_prec: float = 0.0
    best_thr: float = 0.30

    for thr in THRESHOLD_RANGE:
        pred = (oof_proba >= thr).astype(int)
        r = recall_score(y_true, pred)
        p = precision_score(y_true, pred, zero_division=0)
        if r >= recall_min and p > best_prec:
            best_prec = p
            best_thr = round(float(thr), 2)

    final_pred = (oof_proba >= best_thr).astype(int)
    final_recall = recall_score(y_true, final_pred)
    final_f1 = f1_score(y_true, final_pred, zero_division=0)

    return best_thr, best_prec, final_recall, final_f1


def objective(
    trial: Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    """Optuna objective 함수. Recall >= RECALL_MIN 만족 시 OOF Precision 최대화."""
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba: NDArray[np.float64] = np.zeros(len(y_train))
    fold_recalls: list[float] = []

    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
        params["scale_pos_weight"] = float(cw[1] / cw[0])

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        val_proba: NDArray[np.float64] = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_proba
        fold_recalls.append(float(recall_score(y_val, (val_proba >= 0.5).astype(int))))

    # Fold Recall 편차 너무 크면 trial 무효
    if np.std(fold_recalls) > 0.2:
        raise optuna.exceptions.TrialPruned()

    best_thr, best_prec, oof_recall, oof_f1 = tune_threshold(oof_proba, y_train.values)

    # Recall 조건 미달 시 패널티
    if oof_recall < RECALL_MIN:
        return 0.0

    # 부가 지표 기록
    trial.set_user_attr("oof_auc", round(float(roc_auc_score(y_train, oof_proba)), 4))
    trial.set_user_attr("oof_recall", round(oof_recall, 4))
    trial.set_user_attr("oof_f1", round(oof_f1, 4))
    trial.set_user_attr("best_thr", best_thr)
    trial.set_user_attr("fold_recall_std", round(float(np.std(fold_recalls)), 4))

    return best_prec


def retrain_with_best_params(
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[NDArray[np.float64], list[lgb.LGBMClassifier], pd.DataFrame]:
    """Best 파라미터로 최종 OOF 재학습."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba: NDArray[np.float64] = np.zeros(len(y_train))
    fold_models: list[lgb.LGBMClassifier] = []
    fold_scores: list[dict[str, Any]] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
        spw = float(cw[1] / cw[0])

        params: dict[str, Any] = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "scale_pos_weight": spw,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
            **best_params,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        val_proba: NDArray[np.float64] = model.predict_proba(X_val)[:, 1]
        val_label = (val_proba >= 0.5).astype(int)
        oof_proba[val_idx] = val_proba
        fold_models.append(model)

        fold_scores.append({
            "fold": fold,
            "auc": round(float(roc_auc_score(y_val, val_proba)), 4),
            "recall": round(float(recall_score(y_val, val_label)), 4),
            "precision": round(float(precision_score(y_val, val_label, zero_division=0)), 4),
            "f1": round(float(f1_score(y_val, val_label)), 4),
            "best_iter": model.best_iteration_,
            "scale_pos_weight": round(spw, 4),
        })

        print(
            f"  Fold {fold} | AUC: {fold_scores[-1]['auc']:.4f} | "
            f"Recall: {fold_scores[-1]['recall']:.4f} | "
            f"Prec: {fold_scores[-1]['precision']:.4f} | "
            f"F1: {fold_scores[-1]['f1']:.4f} | "
            f"iter: {model.best_iteration_}"
        )

    return oof_proba, fold_models, pd.DataFrame(fold_scores)


def main() -> None:
    # ── 데이터 로드 ───────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"[0] 데이터 로드 | shape: {df.shape}")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"[0] {TARGET} 결측 제거 후 | shape: {df.shape}")
    print(f"[0] 클래스 분포:\n{df[TARGET].value_counts().to_string()}")

    # ── 피처 엔지니어링 ───────────────────────────────────────
    print("\n[FE] 피처 엔지니어링 시작")
    df = apply_feature_engineering(df)

    # ── X / Y 분리 ────────────────────────────────────────────
    drop_cols = [c for c in ["고혈압유병", "당뇨유병", "이상지질혈증유병", "비만단계"] if c in df.columns]
    X: pd.DataFrame = df.drop(columns=drop_cols)
    y: pd.Series = df[TARGET].astype(int)
    print(f"\n[1] Feature 수: {X.shape[1]}")

    # ── Train / Test split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\n[2] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Train 양성 비율: {y_train.mean():.4f} | Test 양성 비율: {y_test.mean():.4f}")

    # ── Optuna Study ──────────────────────────────────────────
    print(f"\n[3] Optuna 시작 | {N_TRIALS} trials | Recall >= {RECALL_MIN} → Precision 최대")
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

    # ── Best Trial 결과 ───────────────────────────────────────
    print("=" * 60)
    best = study.best_trial
    print(f"\n[4] Best Trial #{best.number}")
    print(f"    OOF Precision : {best.value:.4f}")
    print(f"    OOF Recall    : {best.user_attrs.get('oof_recall', '-')}")
    print(f"    OOF AUC       : {best.user_attrs.get('oof_auc', '-')}")
    print(f"    OOF F1        : {best.user_attrs.get('oof_f1', '-')}")
    print(f"    Best Threshold: {best.user_attrs.get('best_thr', '-')}")
    print(f"    Fold Recall std: {best.user_attrs.get('fold_recall_std', '-')}")
    print("\n    Best Params:")
    for k, v in best.params.items():
        print(f"      {k}: {v}")

    # ── Best 파라미터로 최종 OOF 재학습 ──────────────────────
    print("\n[5] Best 파라미터로 최종 OOF 재학습")
    print("=" * 60)

    best_params: dict[str, Any] = best.params.copy()
    best_thr: float = float(best.user_attrs.get("best_thr", 0.47))

    oof_proba, fold_models, scores_df = retrain_with_best_params(best_params, X_train, y_train)

    # ── OOF 전체 성능 ─────────────────────────────────────────
    print("=" * 60)
    oof_label_05 = (oof_proba >= 0.5).astype(int)
    oof_auc: float = float(roc_auc_score(y_train, oof_proba))

    print("\n[6] OOF 전체 성능 (threshold=0.5)")
    print(f"    AUC    : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
    print(f"    Recall : {recall_score(y_train, oof_label_05):.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})")
    print(f"    F1     : {f1_score(y_train, oof_label_05):.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})")

    # ── OOF Threshold 재확인 ──────────────────────────────────
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
    print(f"    Recall: {final_row['recall']:.4f} | Precision: {final_row['precision']:.4f} | F1: {final_row['f1']:.4f}")

    # ── 최종 Hold-out Test 평가 ───────────────────────────────
    print(f"\n[8] 최종 Hold-out Test 평가 (threshold={best_thr:.2f}, 1회 평가)")
    print("=" * 60)

    test_probas = np.column_stack([m.predict_proba(X_test)[:, 1] for m in fold_models])
    test_proba_ensemble: NDArray[np.float64] = test_probas.mean(axis=1)
    test_label = (test_proba_ensemble >= best_thr).astype(int)

    test_auc: float = float(roc_auc_score(y_test, test_proba_ensemble))
    test_recall: float = float(recall_score(y_test, test_label))
    test_prec: float = float(precision_score(y_test, test_label, zero_division=0))
    test_f1: float = float(f1_score(y_test, test_label))
    cm = confusion_matrix(y_test, test_label)

    print(f"    AUC       : {test_auc:.4f}")
    print(f"    Recall    : {test_recall:.4f}")
    print(f"    Precision : {test_prec:.4f}")
    print(f"    F1        : {test_f1:.4f}")
    print(f"    TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"    FN={cm[1, 0]}  TP={cm[1, 1]}")
    print("\n[8] Classification Report")
    print(classification_report(y_test, test_label, target_names=["정상(0)", "고혈압(1)"]))

    # ── Feature Importance ────────────────────────────────────
    print("[9] Feature Importance Top 20 (gain 평균, 5-fold)")
    fi_matrix = np.column_stack([m.feature_importances_ for m in fold_models])
    fi_mean: NDArray[np.float64] = fi_matrix.mean(axis=1)
    fi_df = pd.DataFrame({"feature": X.columns, "importance": fi_mean}).sort_values(
        "importance", ascending=False
    )
    print(fi_df.head(20).to_string(index=False))

    # ── 저장 ─────────────────────────────────────────────────
    scores_df.to_csv(os.path.join(MODEL_DIR, "fold_scores.csv"), index=False)
    fi_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    tuning_df.to_csv(os.path.join(MODEL_DIR, "threshold_tuning.csv"), index=False)
    np.save(os.path.join(MODEL_DIR, "oof_y_true.npy"), y_train.values)
    np.save(os.path.join(MODEL_DIR, "oof_proba.npy"), oof_proba)
    np.save(os.path.join(MODEL_DIR, "best_threshold.npy"), np.array([best_thr]))

    with open(os.path.join(MODEL_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    study.trials_dataframe().to_csv(os.path.join(MODEL_DIR, "optuna_trials.csv"), index=False)

    print(f"\n[10] 저장 완료 → {MODEL_DIR}")


if __name__ == "__main__":
    main()
