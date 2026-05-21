"""
이상지질혈증유병 예측 — Logistic Regression (hn15~24 통합)
Python 3.13 | scikit-learn>=1.4

전처리: StandardScaler + 결측 중앙값/최빈값 대체
피처: 원본 변수 그대로 (나이구간/BMI_X_나이/구간화 없음)
Threshold: RECALL_MIN 0.85, F1 최대화
검증: Train/Test 8:2 Hold-out → Train 내부 5-Fold OOF
"""

import os
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────
DATA_PATH: str = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/data/hn_all_preprocessed.csv"
MODEL_DIR: str = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml/LogisticR/outputs/logistic_DL"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET: str = "이상지질혈증유병"
N_SPLITS: int = 5
SEED: int = 42
RECALL_MIN: float = 0.85
THRESHOLD_RANGE: NDArray[np.float64] = np.arange(0.30, 0.71, 0.01)

# ── 피처 정의 ─────────────────────────────────────────────────
# 연속형 — StandardScaler 적용
CONT_COLS: list[str] = [
    "나이",
    "키",
    "체중",
    "BMI",
    "음주량",
    "걷기일수",
    "근력운동일수",
]

# 범주형 / 이진 — 스케일링 없음, 결측 최빈값 대체
CAT_COLS: list[str] = [
    "성별",
    "현재흡연",
    "음주빈도",
    "직업_관리전문",
    "직업_사무",
    "직업_서비스판매",
    "직업_농림어업",
    "직업_기능노무",
    "직업_주부학생",
    "직업_무직",
    "직업_작업미상",
    "고혈압가족력_부",
    "고혈압가족력_모",
    "고혈압가족력_형제",
    "당뇨가족력_부",
    "당뇨가족력_모",
    "당뇨가족력_형제",
    "고지혈증가족력_부",
    "고지혈증가족력_모",
    "고지혈증가족력_형제",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 가족력 결측 → 0
    fam_cols = [c for c in df.columns if "가족력" in c]
    df[fam_cols] = df[fam_cols].fillna(0)

    # 연속형 결측 → 중앙값
    for c in CONT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # 범주형 결측 → 최빈값
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])

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

    # ── 전처리 ───────────────────────────────────────────────
    print("\n[FE] 전처리 시작 (로지스틱 회귀용)")
    df = preprocess(df)

    feature_cols = CONT_COLS + CAT_COLS
    feature_cols = [c for c in feature_cols if c in df.columns]

    X: pd.DataFrame = df[feature_cols]
    y: pd.Series = df[TARGET].astype(int)
    print(f"  연속형 피처: {len(CONT_COLS)}개 → StandardScaler 적용")
    print(f"  범주형 피처: {len([c for c in CAT_COLS if c in df.columns])}개")
    print(f"\n[1] Feature 수: {X.shape[1]}")
    print(f"    피처 목록: {feature_cols}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"\n[2] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Train 양성 비율: {y_train.mean():.4f} | Test 양성 비율: {y_test.mean():.4f}")

    # ── Pipeline 정의 ─────────────────────────────────────────
    # 연속형만 스케일링, 범주형은 그대로
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    ratio = float(neg / pos)

    from sklearn.compose import ColumnTransformer

    cont_idx = [feature_cols.index(c) for c in CONT_COLS if c in feature_cols]
    cat_idx = [feature_cols.index(c) for c in CAT_COLS if c in feature_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), cont_idx),
            ("passthrough", "passthrough", cat_idx),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight={0: 1.0, 1: ratio},
                    max_iter=1000,
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    # ── 5-Fold OOF ────────────────────────────────────────────
    print(f"\n[3] 5-Fold OOF | RECALL_MIN={RECALL_MIN} → F1 최대화")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba: NDArray[np.float64] = np.zeros(len(y_train))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        val_proba: NDArray[np.float64] = pipeline.predict_proba(X_val)[:, 1]
        val_label = (val_proba >= 0.5).astype(int)
        oof_proba[val_idx] = val_proba

        fold_scores.append(
            {
                "fold": fold,
                "auc": round(float(roc_auc_score(y_val, val_proba)), 4),
                "recall": round(float(recall_score(y_val, val_label)), 4),
                "precision": round(float(precision_score(y_val, val_label, zero_division=0)), 4),
                "f1": round(float(f1_score(y_val, val_label)), 4),
            }
        )
        print(
            f"  Fold {fold} | AUC: {fold_scores[-1]['auc']:.4f} | Recall: {fold_scores[-1]['recall']:.4f} | Prec: {fold_scores[-1]['precision']:.4f} | F1: {fold_scores[-1]['f1']:.4f}"
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

    # ── 전체 Train으로 최종 모델 학습 ────────────────────────
    pipeline.fit(X_train, y_train)

    # ── Test 평가 ─────────────────────────────────────────────
    print(f"\n[6] 최종 Hold-out Test 평가 (threshold={best_thr:.2f})")
    print("=" * 60)

    test_proba: NDArray[np.float64] = pipeline.predict_proba(X_test)[:, 1]
    test_label = (test_proba >= best_thr).astype(int)

    test_auc = float(roc_auc_score(y_test, test_proba))
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
    print(classification_report(y_test, test_label, target_names=["정상(0)", "이상지질혈증(1)"]))

    # ── 계수 확인 ─────────────────────────────────────────────
    print("[7] 계수 Top 15 (절댓값 기준)")
    coef = pipeline.named_steps["model"].coef_[0]

    # ColumnTransformer 후 피처 순서 재구성
    cont_features = [c for c in CONT_COLS if c in feature_cols]
    cat_features = [c for c in CAT_COLS if c in feature_cols]
    ordered_features = cont_features + cat_features

    coef_df = pd.DataFrame(
        {
            "feature": ordered_features,
            "coef": coef,
            "abs_coef": np.abs(coef),
        }
    ).sort_values("abs_coef", ascending=False)
    print(coef_df.head(15).to_string(index=False))

    # ── 저장 ─────────────────────────────────────────────────
    scores_df.to_csv(os.path.join(MODEL_DIR, "fold_scores.csv"), index=False)
    coef_df.to_csv(os.path.join(MODEL_DIR, "coefficients.csv"), index=False)
    np.save(os.path.join(MODEL_DIR, "oof_y_true.npy"), y_train.values)
    np.save(os.path.join(MODEL_DIR, "oof_proba.npy"), oof_proba)
    np.save(os.path.join(MODEL_DIR, "best_threshold.npy"), np.array([best_thr]))

    print(f"\n[8] 저장 완료 → {MODEL_DIR}")


if __name__ == "__main__":
    main()
