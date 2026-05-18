"""
고혈압유병 예측 — LightGBM + 피처 엔지니어링 (hn18~24 통합)
각 FE 블록을 True/False로 켜고 끄면서 ablation 가능
"""

import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────
DATA_PATH = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/data/hn_all_preprocessed.csv"
MODEL_DIR = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml/LGB18~24/outputs/baseline_lgbm_HTN_FE"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
TARGET = "고혈압유병"
N_SPLITS = 5
SEED = 42
THRESHOLD_RANGE = np.arange(0.30, 0.71, 0.01)
RECALL_MIN = 0.85

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 피처 엔지니어링 ON/OFF 스위치 (True=사용 / False=미사용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_AGE_BIN = True       # 나이 구간화 (19~39/40대/50대/60대/70대/80+)
USE_BMI_BIN = False       # BMI 구간화 (한국인 기준: 0~4)
USE_WEIGHT_BIN = False   # 체중 구간화
USE_ALCOHOL_RISK = True  # 음주위험군 (비음주/저위험/고위험)
USE_WALK_LEVEL = True    # 걷기 활동량
USE_STRENGTH = False    # 근력운동 활동량
USE_FAMILY_SUM = True    # 가족력 합산 스코어
USE_BMI_X_AGE = True     # BMI × 나이 상호작용
USE_OBESITY_FLAG = False  # 비만여부 (BMI >= 25)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"[0] 데이터 로드 | shape: {df.shape}")
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
print(f"[0] {TARGET} 결측 제거 후 | shape: {df.shape}")
print(f"[0] 클래스 분포:\n{df[TARGET].value_counts().to_string()}")

# ── 피처 엔지니어링 ───────────────────────────────────────────
print("\n[FE] 피처 엔지니어링 시작")
added = []

# 1. 나이 구간화
if USE_AGE_BIN:
    age_bins = [0, 40, 50, 60, 70, 80, 999]
    age_labels = ["나이_19_39", "나이_40대", "나이_50대", "나이_60대", "나이_70대", "나이_80이상"]
    df["_나이구간"] = pd.cut(df["나이"], bins=age_bins, labels=age_labels, right=False)
    for label in age_labels:
        df[label] = (df["_나이구간"] == label).astype(int)
    df = df.drop(columns=["_나이구간"])
    added += age_labels
    print(f"  [ON] 나이 구간화: {age_labels}")

# 2. BMI 구간화 (fe_bmi_bin.py 기준, 한국인 기준)
if USE_BMI_BIN:
    df['BMI_구간'] = pd.cut(
        df['BMI'], bins=[0, 23, 25, 30, 999],
        labels=[0, 1, 2, 3], right=False
    ).astype(float)
    added += ['BMI_구간']
    print("  ✅ BMI 구간화: 0=정상/1=과체중/2=비만1/3=비만2")

# 3. 체중 구간화
if USE_WEIGHT_BIN:
    wt_bins = [0, 50, 70, 90, 999]
    wt_labels = ["체중_저체중", "체중_정상", "체중_과체중", "체중_비만"]
    df["_체중구간"] = pd.cut(df["체중"], bins=wt_bins, labels=wt_labels, right=False)
    for label in wt_labels:
        df[label] = (df["_체중구간"] == label).astype(float)
    df = df.drop(columns=["_체중구간"])
    added += wt_labels
    print(f"  [ON] 체중 구간화: {wt_labels}")

# 4. 음주위험군 (pd.cut 벡터화 — NaN 자동 유지)
if USE_ALCOHOL_RISK:
    df["음주위험군"] = pd.cut(
        df["음주빈도"],
        bins=[-1, 0, 2, 99],
        labels=[0, 1, 2],
        right=True,
    ).astype(float)
    added += ["음주위험군"]
    print("  [ON] 음주위험군: 0=비음주/1=저위험(월1회이하)/2=고위험(월2회이상)")

# 5. 걷기 활동량 (pd.cut 벡터화 — NaN 자동 유지)
if USE_WALK_LEVEL:
    df["걷기활동량"] = pd.cut(
        df["걷기일수"],
        bins=[-1, 0, 3, 99],
        labels=[0, 1, 2],
        right=True,
    ).astype(float)
    added += ["걷기활동량"]
    print("  [ON] 걷기활동량: 0=비활동/1=저활동(1~3일)/2=활동(4일이상)")

# 6. 근력운동 활동량 (pd.cut 벡터화 — NaN 자동 유지)
if USE_STRENGTH:
    df["근력활동량"] = pd.cut(
        df["근력운동일수"],
        bins=[-1, 0, 2, 99],
        labels=[0, 1, 2],
        right=True,
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
    print("  [ON] BMI x 나이 상호작용")

# 9. 비만여부
if USE_OBESITY_FLAG:
    df["비만여부"] = (df["BMI"] >= 25).astype(float)
    df.loc[df["BMI"].isna(), "비만여부"] = np.nan
    added += ["비만여부"]
    print("  [ON] 비만여부 (BMI >= 25)")

print(f"\n[FE] 추가된 피처 수: {len(added)}")
print(f"[FE] 추가 피처: {added}")

# ── X / Y 분리 ────────────────────────────────────────────────
drop_cols = [
    c for c in ["고혈압유병", "당뇨유병", "이상지질혈증유병", "비만단계"]
    if c in df.columns
]
X = df.drop(columns=drop_cols)
y = df[TARGET].astype(int)
print(f"\n[1] Feature 수: {X.shape[1]}")

# ── Train / Test split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n[2] Train: {X_train.shape} | Test: {X_test.shape}")
print(f"    Train 양성 비율: {y_train.mean():.4f} | Test 양성 비율: {y_test.mean():.4f}")

# ── Stratified 5-Fold OOF CV ──────────────────────────────────
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_proba = np.zeros(len(y_train))
fold_models = []
fold_scores = []

print(f"\n[3] {N_SPLITS}-Fold OOF CV 시작")
print("=" * 60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    spw = cw[1] / cw[0]

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": spw,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    val_label = (val_proba >= 0.5).astype(int)
    oof_proba[val_idx] = val_proba
    fold_models.append(model)

    auc = roc_auc_score(y_val, val_proba)
    recall = recall_score(y_val, val_label)
    prec = precision_score(y_val, val_label, zero_division=0)
    f1 = f1_score(y_val, val_label)

    fold_scores.append({
        "fold": fold,
        "auc": auc,
        "recall": recall,
        "precision": prec,
        "f1": f1,
        "best_iter": model.best_iteration_,
        "scale_pos_weight": round(spw, 4),
    })
    print(
        f"  Fold {fold} | AUC: {auc:.4f} | Recall: {recall:.4f} | "
        f"Prec: {prec:.4f} | F1: {f1:.4f} | iter: {model.best_iteration_}"
    )

# ── OOF 전체 성능 ─────────────────────────────────────────────
print("=" * 60)
oof_label_05 = (oof_proba >= 0.5).astype(int)
oof_auc = roc_auc_score(y_train, oof_proba)
scores_df = pd.DataFrame(fold_scores)

print("\n[4] OOF 전체 성능 (threshold=0.5)")
print(
    f"    AUC    : {oof_auc:.4f}  "
    f"(fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})"
)
print(
    f"    Recall : {recall_score(y_train, oof_label_05):.4f}  "
    f"(fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})"
)
print(
    f"    F1     : {f1_score(y_train, oof_label_05):.4f}  "
    f"(fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})"
)

# ── OOF Threshold Tuning ──────────────────────────────────────
print("\n[5] OOF Threshold Tuning (범위: 0.30~0.70, step=0.01)")
print(f"    기준: Recall >= {RECALL_MIN} 만족 시 Precision 최대")
print("-" * 60)

tuning_rows = []
for thr in THRESHOLD_RANGE:
    pred = (oof_proba >= thr).astype(int)
    r = recall_score(y_train, pred)
    p = precision_score(y_train, pred, zero_division=0)
    f = f1_score(y_train, pred, zero_division=0)
    tuning_rows.append({"threshold": round(thr, 2), "recall": r, "precision": p, "f1": f})

tuning_df = pd.DataFrame(tuning_rows)
candidates = tuning_df[tuning_df["recall"] >= RECALL_MIN]

if len(candidates) > 0:
    best_row = candidates.loc[candidates["precision"].idxmax()]
    best_thr = best_row["threshold"]
    print(f"    선택된 threshold: {best_thr:.2f}")
    print(
        f"       Recall: {best_row['recall']:.4f} | "
        f"Precision: {best_row['precision']:.4f} | "
        f"F1: {best_row['f1']:.4f}"
    )
else:
    best_row = tuning_df.loc[tuning_df["recall"].idxmax()]
    best_thr = best_row["threshold"]
    print(f"    Recall >= {RECALL_MIN} 만족 없음 → Recall 최대 사용: {best_thr:.2f}")
    print(
        f"       Recall: {best_row['recall']:.4f} | "
        f"Precision: {best_row['precision']:.4f} | "
        f"F1: {best_row['f1']:.4f}"
    )

# ── 최종 Hold-out Test 평가 ───────────────────────────────────
print(f"\n[6] 최종 Hold-out Test 평가 (threshold={best_thr:.2f}, 1회 평가)")
print("=" * 60)

test_probas = np.column_stack([m.predict_proba(X_test)[:, 1] for m in fold_models])
test_proba_ensemble = test_probas.mean(axis=1)
test_label = (test_proba_ensemble >= best_thr).astype(int)

test_auc = roc_auc_score(y_test, test_proba_ensemble)
test_recall = recall_score(y_test, test_label)
test_prec = precision_score(y_test, test_label, zero_division=0)
test_f1 = f1_score(y_test, test_label)
cm = confusion_matrix(y_test, test_label)

print(f"    AUC       : {test_auc:.4f}")
print(f"    Recall    : {test_recall:.4f}")
print(f"    Precision : {test_prec:.4f}")
print(f"    F1        : {test_f1:.4f}")
print(f"    TN={cm[0, 0]}  FP={cm[0, 1]}")
print(f"    FN={cm[1, 0]}  TP={cm[1, 1]}")
print("\n[6] Classification Report")
print(classification_report(y_test, test_label, target_names=["정상(0)", "고혈압(1)"]))

# ── Feature Importance ────────────────────────────────────────
print("[7] Feature Importance Top 20 (gain 평균, 5-fold)")
fi_matrix = np.column_stack([m.feature_importances_ for m in fold_models])
fi_mean = fi_matrix.mean(axis=1)
fi_df = pd.DataFrame({"feature": X.columns, "importance": fi_mean}).sort_values(
    "importance", ascending=False
)
print(fi_df.head(20).to_string(index=False))

# ── 저장 ─────────────────────────────────────────────────────
scores_df.to_csv(os.path.join(MODEL_DIR, "fold_scores.csv"), index=False)
fi_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
tuning_df.to_csv(os.path.join(MODEL_DIR, "threshold_tuning.csv"), index=False)
np.save(os.path.join(MODEL_DIR, "oof_y_true.npy"), y_train.values)
np.save(os.path.join(MODEL_DIR, "oof_proba.npy"), oof_proba)
np.save(os.path.join(MODEL_DIR, "best_threshold.npy"), np.array([best_thr]))

print(f"\n[8] 저장 완료 → {MODEL_DIR}")