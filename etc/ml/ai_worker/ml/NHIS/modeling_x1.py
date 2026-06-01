"""
국민건강보험공단 건강검진
X1 모델링 — 통합 실험 관리

실험 추가 방법:
    EXPERIMENTS 리스트에 딕셔너리 추가
    완료된 실험은 주석 처리

실행환경: Python 3.10+
패키지  : pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, optuna
설치    : pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, precision_score, confusion_matrix
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ────────────────────────────────────────────
# 0. 경로 설정
# ────────────────────────────────────────────
BASE_DIR = "/Users/admin/PycharmProjects/AH_03_03/etc/ml/ai_worker"
DATA_24  = os.path.join(BASE_DIR, "data", "국민건강보험공단_건강검진정보_2024.CSV")
DATA_23  = os.path.join(BASE_DIR, "data", "국민건강보험공단_건강검진정보_2023.CSV")
DATA_22  = os.path.join(BASE_DIR, "data", "국민건강보험공단_건강검진정보_20221231_수정.CSV")
OUT_DIR  = os.path.join(BASE_DIR, "ml", "NHIS", "outputs", "Modeling_X1")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE  = 42
TEST_SIZE     = 0.2
RECALL_TARGET = 0.8

# ────────────────────────────────────────────
# 1. df_all 구성 정의
#    2024 전체 + 아래 타겟의 2023/2022 양성 행 추가
#    모든 타겟이 동일한 df_all 기반 → 공정한 비교
# ────────────────────────────────────────────
ADD_23_POS = ["고혈압", "대사증후군", "간기능이상", "신장단백뇨"]
ADD_22_POS = ["고혈압", "대사증후군", "간기능이상", "신장단백뇨"]

# ────────────────────────────────────────────
# 2. 실험 목록
#    완료된 실험은 주석 처리 / 새 실험은 아래에 추가
#
#    optuna: True  → tune_models 에 지정한 모델만 Optuna 튜닝
#            False → 전체 기본 파라미터로 학습
# ────────────────────────────────────────────
EXPERIMENTS = [
    # ── A: 기본 실험 ─────────────────────────────────────────
    # {
    #     "tag":    "A",
    #     "desc":   "2024+2023+2022 통합 df_all 기반 기본 파라미터",
    #     "optuna": False,
    # },

    # ── B: Optuna 튜닝 (LightGBM + CatBoost) ─────────────────
    {
        "tag":         "B",
        "desc":        "Optuna 튜닝 — LightGBM + CatBoost / 전체 6개 타겟 / 50 trials",
        "optuna":      True,
        "trials":      50,
        "tune_models": ["LightGBM", "CatBoost"],
    },

    # ── C: 다음 실험 ──────────────────────────────────────────
    # {
    #     "tag":    "C",
    #     "desc":   "",
    #     "optuna": False,
    # },
]

# ────────────────────────────────────────────
# 3. 타겟 / 피처 정의
# ────────────────────────────────────────────
TARGETS = {
    "당뇨위험":    "target_diabetes",
    "고혈압":      "target_hypertension",
    "이상지질혈증": "target_dyslipidemia",
    "대사증후군":   "target_metabolic",
    "간기능이상":   "target_liver",
    "신장단백뇨":   "target_proteinuria",
}

FEATURE_COLS = [
    "성별코드", "신장(5cm단위)", "체중(5kg단위)", "허리둘레", "음주여부",
    "bmi", "bmi_category", "waist_height_ratio",
    "age_mid", "gender_age_enc", "obesity_combined",
    "smoking_current", "smoking_ever",
]

# ────────────────────────────────────────────
# 4. 전처리 함수
# ────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df["허리둘레"]         = df["허리둘레"].replace(999, np.nan)
    df["감마지티피"]        = df["감마지티피"].replace(9999, np.nan)
    df["식전혈당(공복혈당)"] = df["식전혈당(공복혈당)"].replace(991, np.nan)
    return df

def make_targets(df):
    # 당뇨위험
    df["target_diabetes"] = np.where(df["식전혈당(공복혈당)"] >= 100, 1, 0)
    df.loc[df["식전혈당(공복혈당)"].isna(), "target_diabetes"] = np.nan

    # 고혈압
    df["target_hypertension"] = np.where(
        (df["수축기혈압"] >= 140) | (df["이완기혈압"] >= 90), 1, 0
    )
    df.loc[df["수축기혈압"].isna() & df["이완기혈압"].isna(), "target_hypertension"] = np.nan

    # 이상지질혈증
    hdl_low = (
        ((df["성별코드"] == 1) & (df["HDL콜레스테롤"] < 40)) |
        ((df["성별코드"] == 2) & (df["HDL콜레스테롤"] < 50))
    )
    dyslipidemia = (
        (df["총콜레스테롤"] >= 200) | (df["LDL콜레스테롤"] >= 130) |
        (df["트리글리세라이드"] >= 150) | hdl_low
    )
    chol_all_na = df[["총콜레스테롤","LDL콜레스테롤",
                       "트리글리세라이드","HDL콜레스테롤"]].isna().all(axis=1)
    df["target_dyslipidemia"] = np.where(dyslipidemia, 1, 0)
    df.loc[chol_all_na, "target_dyslipidemia"] = np.nan

    # 대사증후군
    abdom   = (((df["성별코드"]==1) & (df["허리둘레"]>=90)) |
               ((df["성별코드"]==2) & (df["허리둘레"]>=85)))
    tg_hi   = df["트리글리세라이드"] >= 150
    bp_ms   = (df["수축기혈압"] >= 130) | (df["이완기혈압"] >= 85)
    gluc_ms = df["식전혈당(공복혈당)"] >= 100
    ms_score = (abdom.astype(float) + tg_hi.astype(float) +
                hdl_low.astype(float) + bp_ms.astype(float) +
                gluc_ms.astype(float))
    df["target_metabolic"] = np.where(ms_score >= 3, 1, 0)
    ms_na = df[["허리둘레","트리글리세라이드","HDL콜레스테롤",
                "수축기혈압","식전혈당(공복혈당)"]].isna().sum(axis=1)
    df.loc[ms_na >= 3, "target_metabolic"] = np.nan

    # 간기능이상
    liver_ast_alt = (df["혈청지오티(AST)"] > 40) | (df["혈청지피티(ALT)"] > 40)
    ggt_high = (
        ((df["성별코드"] == 1) & (df["감마지티피"] > 63)) |
        ((df["성별코드"] == 2) & (df["감마지티피"] > 35))
    )
    df["target_liver"] = np.where(liver_ast_alt | ggt_high, 1, 0)
    liver_na = df[["혈청지오티(AST)","혈청지피티(ALT)","감마지티피"]].isna().all(axis=1)
    df.loc[liver_na, "target_liver"] = np.nan

    # 신장단백뇨 — 요단백 1+ 이상 (코드 ≥ 3, 대한신장학회 기준)
    df["target_proteinuria"] = np.where(df["요단백"] >= 3, 1, 0)
    df.loc[df["요단백"].isna(), "target_proteinuria"] = np.nan

    return df

def make_features(df):
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    df["bmi"]                = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
    df["bmi_category"]       = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 23.0, 25.0, 30.0, 35.0, 999],
        labels=[0, 1, 2, 3, 4, 5], right=False
    ).astype(float)
    df["waist_height_ratio"] = (df["허리둘레"] / df["신장(5cm단위)"]).round(3)
    age_mid = {5:27,6:32,7:37,8:42,9:47,10:52,
               11:57,12:62,13:67,14:72,15:77,16:82,17:87,18:92}
    df["age_mid"]            = df["연령대코드(5세단위)"].map(age_mid)
    le = LabelEncoder()
    df["gender_age_enc"]     = le.fit_transform(
        df["성별코드"].astype(str) + "_" + df["연령대코드(5세단위)"].astype(str)
    )
    df["obesity_combined"]   = (
        (df["bmi"] >= 25) & (df["waist_height_ratio"] >= 0.5)
    ).astype(float)
    df["smoking_current"]    = (df["흡연상태"] == 3).astype(float)
    df["smoking_ever"]       = (df["흡연상태"] >= 2).astype(float)
    return df

# ────────────────────────────────────────────
# 5. 모델 정의
# ────────────────────────────────────────────
def get_models(scale_pos_weight=1.0, class_weight=None):
    cw = class_weight
    return {
        "LR": Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(
                max_iter=1000, random_state=RANDOM_STATE,
                n_jobs=-1, class_weight=cw
            )),
        ]),
        "RF": Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("model",   RandomForestClassifier(
                n_estimators=300, max_depth=10,
                random_state=RANDOM_STATE, n_jobs=-1,
                class_weight=cw
            )),
        ]),
        "LightGBM": LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=63, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1, class_weight=cw
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric="logloss",
            verbosity=0, scale_pos_weight=scale_pos_weight
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05,
            depth=6, random_state=RANDOM_STATE, verbose=0,
            auto_class_weights="Balanced" if cw == "balanced" else None
        ),
    }

# ────────────────────────────────────────────
# 6. Threshold 자동 조정
# ────────────────────────────────────────────
def find_best_threshold(y_true, y_prob, recall_target=RECALL_TARGET):
    best_thr    = 0.5
    best_f1     = 0.0
    best_recall = 0.0
    for thr in np.arange(0.1, 0.9, 0.01):
        y_pred  = (y_prob >= thr).astype(int)
        recall  = recall_score(y_true, y_pred, zero_division=0)
        f1      = f1_score(y_true, y_pred, zero_division=0)
        if recall >= recall_target and f1 > best_f1:
            best_f1, best_thr, best_recall = f1, thr, recall
    if best_f1 == 0.0:
        for thr in np.arange(0.1, 0.9, 0.01):
            y_pred  = (y_prob >= thr).astype(int)
            recall  = recall_score(y_true, y_pred, zero_division=0)
            if recall > best_recall:
                best_recall = recall
                best_thr    = thr
                best_f1     = f1_score(y_true, y_pred, zero_division=0)
    return round(best_thr, 2)

# ────────────────────────────────────────────
# 7. Optuna objective 함수
# ────────────────────────────────────────────
def optuna_objective_lgbm(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state":      RANDOM_STATE,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    thr    = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    return f1 if recall >= RECALL_TARGET else recall - 1.0

def optuna_objective_catboost(trial, X_train, y_train, X_test, y_test):
    params = {
        "iterations":          trial.suggest_int("iterations", 200, 1000),
        "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth":               trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength":     trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "random_state":        RANDOM_STATE,
        "verbose":             0,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    thr    = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    return f1 if recall >= RECALL_TARGET else recall - 1.0

# ────────────────────────────────────────────
# 8. 데이터 로드
# ────────────────────────────────────────────
print("2024 데이터 로드 중...")
df24 = pd.read_csv(DATA_24, encoding="cp949")
df24 = preprocess(df24); df24 = make_targets(df24); df24 = make_features(df24)
print(f"  완료: {df24.shape[0]:,}행")

print("2023 데이터 로드 중...")
df23 = pd.read_csv(DATA_23, encoding="cp949")
df23 = preprocess(df23); df23 = make_targets(df23); df23 = make_features(df23)
print(f"  완료: {df23.shape[0]:,}행")

print("2022 데이터 로드 중...")
df22 = pd.read_csv(DATA_22, encoding="cp949")
df22 = df22.rename(columns={"성별": "성별코드"})
df22 = preprocess(df22); df22 = make_targets(df22); df22 = make_features(df22)
print(f"  완료: {df22.shape[0]:,}행")

# ── df_all 구성 ──────────────────────────────────────────────
print("\ndf_all 구성 중...")
target_cols = list(TARGETS.values())
all_cols    = FEATURE_COLS + target_cols
df_all      = df24[all_cols].copy()

for target_name, target_col in TARGETS.items():
    if target_name in ADD_23_POS:
        pos23  = df23[all_cols].dropna(subset=[target_col])
        pos23  = pos23[pos23[target_col] == 1]
        df_all = pd.concat([df_all, pos23], ignore_index=True)
        print(f"  +2023 {target_name} 양성: {len(pos23):,}행")
    if target_name in ADD_22_POS:
        pos22  = df22[all_cols].dropna(subset=[target_col])
        pos22  = pos22[pos22[target_col] == 1]
        df_all = pd.concat([df_all, pos22], ignore_index=True)
        print(f"  +2022 {target_name} 양성: {len(pos22):,}행")

df_all = df_all.reset_index(drop=True)
print(f"\ndf_all 최종: {len(df_all):,}행")

print("\n[df_all 타겟별 양성률]")
for t_name, t_col in TARGETS.items():
    valid    = df_all[t_col].dropna()
    pos_rate = valid.mean() * 100
    print(f"  {t_name:8s}: {len(valid):>9,}행 | 양성률 {pos_rate:.1f}%")

# ────────────────────────────────────────────
# 9. 실험 루프
# ────────────────────────────────────────────
all_results     = []
all_params      = []   # Optuna 베스트 파라미터
all_importances = []   # 피처 중요도

for exp in EXPERIMENTS:
    tag         = exp["tag"]
    use_optuna  = exp.get("optuna", False)
    n_trials    = exp.get("trials", 50)
    tune_models = exp.get("tune_models", [])

    print(f"\n{'='*60}")
    print(f"실험: {tag}")
    print(f"  설명: {exp.get('desc', '-')}")
    print(f"{'='*60}")

    for target_name, target_col in TARGETS.items():
        print(f"\n  [타겟: {target_name}]")

        df_use = df_all[FEATURE_COLS + [target_col]].dropna(subset=[target_col])
        X      = df_use[FEATURE_COLS]
        y      = df_use[target_col].astype(int)

        pos_rate = y.mean() * 100
        print(f"  유효 샘플: {len(y):,}행 | 양성률: {pos_rate:.1f}%")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=y
        )
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

        target_results = []

        # ── 일반 모델 ──────────────────────────────────────────
        for model_name, model in get_models().items():
            if use_optuna and model_name in tune_models:
                continue

            model.fit(X_train, y_train)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

            auc      = roc_auc_score(y_test, y_prob)
            best_thr = find_best_threshold(y_test, y_prob)
            y_pred   = (y_prob >= best_thr).astype(int)
            recall   = recall_score(y_test, y_pred, zero_division=0)
            f1       = f1_score(y_test, y_pred, zero_division=0)
            prec     = precision_score(y_test, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            recall_ok = "✅" if recall >= RECALL_TARGET else "❌"
            f1_ok     = "✅" if f1 >= 0.6 else "❌"
            print(f"  {model_name:<12} AUC: {auc:.4f} | Recall: {recall:.4f} {recall_ok} | "
                  f"F1: {f1:.4f} {f1_ok} | Precision: {prec:.4f} | Threshold: {best_thr:.2f}")

            result = {
                "실험": tag, "타겟": target_name, "모델": model_name,
                "AUC": round(auc,4), "Recall": round(recall,4),
                "F1": round(f1,4), "Precision": round(prec,4),
                "Threshold": best_thr,
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                "Recall_달성": recall >= RECALL_TARGET, "F1_달성": f1 >= 0.6,
            }
            target_results.append(result)
            all_results.append(result)

            # 피처 중요도 (LightGBM / CatBoost / RF)
            actual = model.named_steps["model"] if hasattr(model, "named_steps") else model
            if hasattr(actual, "feature_importances_"):
                for fname, fval in zip(FEATURE_COLS, actual.feature_importances_):
                    all_importances.append({
                        "실험": tag, "타겟": target_name, "모델": model_name,
                        "피처": fname, "중요도": round(float(fval), 6),
                    })

        # ── Optuna 튜닝 모델 ────────────────────────────────────
        if use_optuna:
            for model_name in tune_models:
                print(f"  {model_name:<12} Optuna 튜닝 중 ({n_trials} trials)...")

                if model_name == "LightGBM":
                    study = optuna.create_study(direction="maximize")
                    study.optimize(
                        lambda trial: optuna_objective_lgbm(trial, X_train, y_train, X_test, y_test),
                        n_trials=n_trials, show_progress_bar=False
                    )
                    best_params = study.best_params
                    best_model  = LGBMClassifier(
                        **best_params, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
                    )
                elif model_name == "CatBoost":
                    study = optuna.create_study(direction="maximize")
                    study.optimize(
                        lambda trial: optuna_objective_catboost(trial, X_train, y_train, X_test, y_test),
                        n_trials=n_trials, show_progress_bar=False
                    )
                    best_params = study.best_params
                    best_model  = CatBoostClassifier(
                        **best_params, random_state=RANDOM_STATE, verbose=0
                    )

                best_model.fit(X_train, y_train)
                y_prob   = best_model.predict_proba(X_test)[:, 1]
                auc      = roc_auc_score(y_test, y_prob)
                best_thr = find_best_threshold(y_test, y_prob)
                y_pred   = (y_prob >= best_thr).astype(int)
                recall   = recall_score(y_test, y_pred, zero_division=0)
                f1       = f1_score(y_test, y_pred, zero_division=0)
                prec     = precision_score(y_test, y_pred, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                recall_ok = "✅" if recall >= RECALL_TARGET else "❌"
                f1_ok     = "✅" if f1 >= 0.6 else "❌"
                print(f"  {model_name+'(Optuna)':<18} AUC: {auc:.4f} | Recall: {recall:.4f} {recall_ok} | "
                      f"F1: {f1:.4f} {f1_ok} | Precision: {prec:.4f} | Threshold: {best_thr:.2f}")
                print(f"    └ best_params: {best_params}")

                # 베스트 파라미터 저장
                param_record = {
                    "실험": tag, "타겟": target_name,
                    "모델": f"{model_name}(Optuna)",
                    "trials": n_trials,
                    "best_value": round(study.best_value, 4),
                }
                param_record.update({f"param_{k}": v for k, v in best_params.items()})
                all_params.append(param_record)

                result = {
                    "실험": tag, "타겟": target_name, "모델": f"{model_name}(Optuna)",
                    "AUC": round(auc,4), "Recall": round(recall,4),
                    "F1": round(f1,4), "Precision": round(prec,4),
                    "Threshold": best_thr,
                    "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                    "Recall_달성": recall >= RECALL_TARGET, "F1_달성": f1 >= 0.6,
                }
                target_results.append(result)
                all_results.append(result)

                # 피처 중요도
                if hasattr(best_model, "feature_importances_"):
                    for fname, fval in zip(FEATURE_COLS, best_model.feature_importances_):
                        all_importances.append({
                            "실험": tag, "타겟": target_name,
                            "모델": f"{model_name}(Optuna)",
                            "피처": fname, "중요도": round(float(fval), 6),
                        })

        best = max(target_results, key=lambda x: (x["Recall_달성"], x["F1"]))
        print(f"\n  → [{target_name}] 최고: {best['모델']} "
              f"AUC {best['AUC']:.4f} / Recall {best['Recall']:.4f} / F1 {best['F1']:.4f}")

# ────────────────────────────────────────────
# 10. 전체 결과 요약
# ────────────────────────────────────────────
print(f"\n{'='*60}")
print("전체 결과 요약")
print(f"{'='*60}")

results_df = pd.DataFrame(all_results)
best_df = (
    results_df
    .sort_values(["실험","타겟","Recall_달성","F1"], ascending=[True,True,False,False])
    .groupby(["실험","타겟"]).first().reset_index()
)
print("\n[실험별 타겟별 최고 모델]")
print(best_df[["실험","타겟","모델","AUC","Recall","F1","Precision","Threshold"]].to_string(index=False))

print(f"\n[목표 달성 현황 — Recall≥{RECALL_TARGET} AND F1≥0.6]")
achieved = results_df[results_df["Recall_달성"] & results_df["F1_달성"]]
print(f"전체 {len(results_df)}개 중 {len(achieved)}개 달성")
if len(achieved) > 0:
    print(achieved[["실험","타겟","모델","Recall","F1"]].to_string(index=False))

# ────────────────────────────────────────────
# 11. 결과 저장
# ────────────────────────────────────────────
def save_with_accumulate(new_df, filename_tag, dir_path, tag_col="실험"):
    """실험 태그별 개별 저장 + 누적 저장"""
    # 태그별 개별 저장
    for tag_name, group in new_df.groupby(tag_col):
        path = os.path.join(dir_path, f"{filename_tag}_{tag_name}.csv")
        group.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  저장: {path}")
    # 누적 저장
    all_path = os.path.join(dir_path, f"{filename_tag}_all.csv")
    if os.path.exists(all_path):
        existing = pd.read_csv(all_path, encoding="utf-8-sig")
        existing = existing[~existing[tag_col].isin(new_df[tag_col].unique())]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"  누적 저장: {all_path}")

print("\n[성능 결과 저장]")
save_with_accumulate(results_df, "model_comparison", OUT_DIR)

if all_importances:
    print("\n[피처 중요도 저장]")
    fi_df = pd.DataFrame(all_importances)
    fi_df = fi_df.sort_values(["실험","타겟","모델","중요도"], ascending=[True,True,True,False])
    save_with_accumulate(fi_df, "feature_importance", OUT_DIR)

if all_params:
    print("\n[Optuna 베스트 파라미터 저장]")
    params_df = pd.DataFrame(all_params)
    save_with_accumulate(params_df, "best_params", OUT_DIR)

print("\n완료!")
