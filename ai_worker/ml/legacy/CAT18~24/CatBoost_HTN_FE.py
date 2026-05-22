"""
고혈압유병 예측 — CatBoost + FE 베이스라인 (hn18~24 통합)
Python 3.13 | catboost>=1.2 | scikit-learn>=1.4

FE 조합: 나이구간 + 음주위험군 + 걷기활동량 + 가족력합산 + BMI_X_나이
Threshold: RECALL_MIN 0.87, F1 최대화
검증: Train/Test 8:2 Hold-out → Train 내부 5-Fold OOF
"""

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from numpy.typing import NDArray
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

# ── 경로 설정 ─────────────────────────────────────────────────
DATA_PATH: str = str(Path(__file__).parent.parent.parent.parent / "ai_worker" / "data" / "hn_all_preprocessed.csv")
MODEL_DIR: str = str(Path(__file__).parent.parent.parent.parent / "ai_worker" / "ml" / "CAT18~24" / "outputs" / "catboost_HTN_FE")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET: str = "고혈압유병"
N_SPLITS: int = 5
SEED: int = 42
RECALL_MIN: float = 0.87
THRESHOLD_RANGE: NDArray[np.float64] = np.arange(0.30, 0.71, 0.01)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HTN 최적 FE 조합
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_AGE_BIN: bool = True
USE_BMI_BIN: bool = False
USE_WEIGHT_BIN: bool = False
USE_ALCOHOL_RISK: bool = True
USE_WALK_LEVEL: bool = True
USE_STRENGTH: bool = False
USE_FAMILY_SUM: bool = True
USE_BMI_X_AGE: bool = True
USE_OBESITY_FLAG: bool = False


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

    if USE_WEIGHT_BIN:
        wt_bins = [0, 50, 70, 90, np.inf]
        wt_labels = ["체중_저체중", "체중_정상", "체중_과체중", "체중_비만"]
        df["_체중구간"] = pd.cut(df["체중"], bins=wt_bins, labels=wt_labels, right=False)
        for label in wt_labels:
            df[label] = (df["_체중구간"] == label).astype(float)
        df = df.drop(columns=["_체중구간"])
        added += wt_labels

    if USE_ALCOHOL_RISK:
        df["음주위험군"] = pd.cut(df["음주빈도"], bins=[-1, 0, 2, 99], labels=[0, 1, 2], right=True).astype(float)
        added += ["음주위험군"]
        print("  [ON] 음주위험군")

    if USE_WALK_LEVEL:
        df["걷기활동량"] = pd.cut(df["걷기일수"], bins=[-1, 0, 3, 99], labels=[0, 1, 2], right=True).astype(float)
        added += ["걷기활동량"]
        print("  [ON] 걷기활동량")

    if USE_STRENGTH:
        df["근력활동량"] = pd.cut(df["근력운동일수"], bins=[-1, 0, 2, 99], labels=[0, 1, 2], right=True).astype(float)
        added += ["근력활동량"]

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

    if USE_OBESITY_FLAG:
        df["비만여부"] = (df["BMI"] >= 25).astype(float)
        df.loc[df["BMI"].isna(), "비만여부"] = np.nan
        added += ["비만여부"]

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


def main() -> None:
    # ── 데이터 로드 ───────────────────────────────────────────
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

    # ── 클래스 가중치 ─────────────────────────────────────────
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    ratio = float(neg / pos)

    # ── CatBoost 파라미터 ─────────────────────────────────────
    params: dict[str, Any] = dict(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights={0: 1.0, 1: ratio},
        early_stopping_rounds=50,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )

    # ── 5-Fold OOF ────────────────────────────────────────────
    print(f"\n[3] 5-Fold OOF | RECALL_MIN={RECALL_MIN} → F1 최대화")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba: NDArray[np.float64] = np.zeros(len(y_train))
    fold_models = []
    fold_scores = []

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
            f"  Fold {fold} | AUC: {fold_scores[-1]['auc']:.4f} | Recall: {fold_scores[-1]['recall']:.4f} | Prec: {fold_scores[-1]['precision']:.4f} | F1: {fold_scores[-1]['f1']:.4f} | iter: {model.best_iteration_}"
        )

    scores_df = pd.DataFrame(fold_scores)

    # ── OOF 전체 성능 ─────────────────────────────────────────
    print("=" * 60)
    oof_label_05 = (oof_proba >= 0.5).astype(int)
    oof_auc = float(roc_auc_score(y_train, oof_proba))
    print("\n[4] OOF 전체 성능 (threshold=0.5)")
    print(f"    AUC    : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
    print(
        f"    Recall : {recall_score(y_train, oof_label_05):.4f}  (fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})"
    )
    print(
        f"    F1     : {f1_score(y_train, oof_label_05):.4f}  (fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})"
    )

    # ── OOF Threshold 탐색 ────────────────────────────────────
    best_thr, best_f1, oof_recall, oof_prec = tune_threshold(oof_proba, y_train.values)
    print(f"\n[5] OOF Threshold 탐색 (RECALL_MIN={RECALL_MIN})")
    print(f"    best_thr: {best_thr:.2f} | Recall: {oof_recall:.4f} | Precision: {oof_prec:.4f} | F1: {best_f1:.4f}")

    # ── Test 평가 ─────────────────────────────────────────────
    print(f"\n[6] 최종 Hold-out Test 평가 (threshold={best_thr:.2f})")
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
    print("\n[6] Classification Report")
    print(classification_report(y_test, test_label, target_names=["정상(0)", "고혈압(1)"]))

    # ── Feature Importance ────────────────────────────────────
    print("[7] Feature Importance Top 20 (마지막 fold)")
    fi_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": fold_models[-1].get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    print(fi_df.head(20).to_string(index=False))

    # ── 저장 ─────────────────────────────────────────────────
    scores_df.to_csv(os.path.join(MODEL_DIR, "fold_scores.csv"), index=False)
    fi_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    np.save(os.path.join(MODEL_DIR, "oof_y_true.npy"), y_train.values)
    np.save(os.path.join(MODEL_DIR, "oof_proba.npy"), oof_proba)
    np.save(os.path.join(MODEL_DIR, "best_threshold.npy"), np.array([best_thr]))

    print(f"\n[8] 저장 완료 → {MODEL_DIR}")


if __name__ == "__main__":
    main()
