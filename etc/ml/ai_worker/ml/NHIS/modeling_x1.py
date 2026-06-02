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
import joblib

# imbalanced-learn — 불균형 앙상블
try:
    from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier
    IMBALANCED_AVAILABLE = True
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn", "-q"])
    from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier
    IMBALANCED_AVAILABLE = True

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
ADD_23_POS = ["고혈압","간기능이상"]
ADD_22_POS = ["고혈압","간기능이상"]

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
    # {
    #     "tag":         "B",
    #     "desc":        "Optuna 튜닝 — LightGBM + CatBoost / 전체 6개 타겟 / 50 trials",
    #     "optuna":      True,
    #     "trials":      50,
    #     "tune_models": ["LightGBM", "CatBoost"],
    # },

    # ── C: B 베스트 파라미터 고정 + CLINICAL_BOUNDS 이상치 처리 ──
    # {
    #     "tag":    "C",
    #     "desc":   "B 베스트 파라미터 고정 + CLINICAL_BOUNDS 의학적 이상치 처리",
    #     "optuna": False,
    # },

    # ── D: C + df_all 중복 행 제거 ───────────────────────────
    # {
    #     "tag":    "D",
    #     "desc":   "B 베스트 파라미터 고정 + CLINICAL_BOUNDS + df_all 중복 제거",
    #     "optuna": False,
    #     "dedup":  True,
    # },

    # ── E: C + 타겟별 class_weight balanced ──────────────────
    # {
    #     "tag":          "E",
    #     "desc":         "C + 타겟별 class_weight balanced 불균형 처리",
    #     "optuna":       False,
    #     "class_weight": "balanced",
    # },

    # ── F: E + bmi_category 제거 ─────────────────────────────
    # {
    #     "tag":          "F",
    #     "desc":         "E + bmi_category 전체 제거 (중요도 전 타겟 최하위)",
    #     "optuna":       False,
    #     "class_weight": "balanced",
    #     "save_models":  True,
    # },

    # ── G: F + 신장/체중 5단위 jitter (해상도 개선) ───────────
    # {
    #     "tag":          "G",
    #     "desc":         "F + 신장/체중 5단위 균등분포 jitter → BMI/WHtR 해상도 개선",
    #     "optuna":       False,
    #     "class_weight": "balanced",
    #     "jitter":       True,
    # },

    # ── H: 불균형 앙상블 (고혈압/당뇨 타겟 집중) ────────────────
    # {
    #     "tag":            "H",
    #     "desc":           "불균형 앙상블 — EasyEnsemble + BalancedRF",
    #     "optuna":         False,
    #     "class_weight":   "balanced",
    #     "imb_ensemble":   True,
    #     "skip_targets":   ["대사증후군", "신장단백뇨"],
    # },

    # ── I: OOF 확률 평균 앙상블 ──────────────────────────────
    # {
    #     "tag":          "I",
    #     "desc":         "OOF 확률 평균 앙상블 — LightGBM + CatBoost + XGBoost",
    #     "optuna":       False,
    #     "class_weight": "balanced",
    #     "oof_ensemble": True,
    #     "oof_models":   ["LightGBM", "CatBoost", "XGBoost"],
    #     "skip_targets": ["대사증후군", "신장단백뇨"],
    # },

    # ── J: OOF 스태킹 앙상블 ─────────────────────────────────
    # LightGBM + CatBoost + XGBoost → 5-fold OOF 확률 → LR 메타 모델
    # 대사증후군/신장단백뇨 제외
    {
        "tag":          "J",
        "desc":         "OOF 스태킹 — LightGBM+CatBoost+XGBoost → LR 메타모델",
        "optuna":       False,
        "class_weight": "balanced",
        "stacking":     True,
        "stack_models": ["LightGBM", "CatBoost", "XGBoost"],
        "stack_folds":  5,
        "save_models":  True,   # KNHANES 외부 검증용 스태킹 모델 저장
    },
]

# ────────────────────────────────────────────
# 3. 타겟 / 피처 정의
# ────────────────────────────────────────────
TARGETS = {
    "당뇨위험":    "target_diabetes",
    "고혈압":      "target_hypertension",
    "이상지질혈증": "target_dyslipidemia",
    "간기능이상":   "target_liver",
}

FEATURE_COLS = [
    "성별코드", "신장(5cm단위)", "체중(5kg단위)", "허리둘레", "음주여부",
    "bmi", "waist_height_ratio",
    "age_mid", "gender_age_enc", "obesity_combined",
    "smoking_current", "smoking_ever",
]  # bmi_category 제거 (F 실험: 전 타겟 중요도 최하위, bmi 연속값과 중복)

# ────────────────────────────────────────────
# 4. 전처리 함수
# ────────────────────────────────────────────
CLINICAL_BOUNDS = {
    "신장(5cm단위)":      (100, 250),   # 기네스 최장신 251cm / 100 미만 성인 측정오류
    "체중(5kg단위)":      (20,  350),   # 20 미만 성인 측정오류 / 350 초과 국내 현실적 불가
    "허리둘레":           (40,  200),   # 999 코드값 → 범위 초과 자동 NaN
    "수축기혈압":         (60,  280),   # 60 미만 심인성 쇼크 / 280 초과 고혈압 응급 상한
    "이완기혈압":         (40,  150),   # 임상적 측정 가능 범위
    "식전혈당(공복혈당)": (40,  600),   # 40 미만 중증저혈당 / 600 초과 HHS / 991 코드값 포함
    "총콜레스테롤":       (50,  700),   # 가족성고콜레스테롤혈증 극단값 기준
    "LDL콜레스테롤":      (10,  500),
    "HDL콜레스테롤":      (10,  150),   # 150 초과 측정오류 수준
    "트리글리세라이드":   (20,  5000),  # 급성췌장염 유발 / 드물게 5000대 보고
    "혈청지오티(AST)":    (5,   5000),  # 급성간염/허혈성간염 수천 단위 보고
    "혈청지피티(ALT)":    (5,   5000),
    "감마지티피":         (5,   3000),  # 알코올성 간질환 극단값 / 9999 코드값 포함
}

def preprocess(df):
    """의학적 CLINICAL_BOUNDS 기반 이상치 → NaN 처리 (v2)"""
    df = df.copy()
    for col, (lo, hi) in CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi), other=np.nan)
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
# ── B 실험 베스트 파라미터 (타겟별 고정) ──────────────────────
# Optuna 50 trials 결과 / C 실험에서 고정 파라미터로 사용
BEST_PARAMS_B = {
    "당뇨위험": {
        "LightGBM": dict(n_estimators=522, learning_rate=0.05034719040252834, num_leaves=101,
                         max_depth=8, min_child_samples=81, subsample=0.5420498720757686,
                         colsample_bytree=0.9656316260251595, reg_alpha=0.7639368798428724,
                         reg_lambda=0.00618006575315154),
        "CatBoost": dict(iterations=806, learning_rate=0.04623826630220252, depth=10,
                         l2_leaf_reg=1e-3,  # degenerate 방지: 원본 1.86e-5 → 1e-3 상향
                         bagging_temperature=0.09197660588086623, random_strength=0.0005944485244359853),
    },
    "고혈압": {
        "LightGBM": dict(n_estimators=507, learning_rate=0.22331242412867336, num_leaves=149,
                         max_depth=12, min_child_samples=10, subsample=0.7954172753805078,
                         colsample_bytree=0.7247644266894295, reg_alpha=3.831e-05,
                         reg_lambda=0.0006679438865010269),
        "CatBoost": dict(iterations=742, learning_rate=0.2849015462477188, depth=10,
                         l2_leaf_reg=1e-3,  # degenerate 방지: 원본 1.56e-6 → 1e-3 상향
                         bagging_temperature=0.776653017329181, random_strength=1.580e-05),
    },
    "이상지질혈증": {
        "LightGBM": dict(n_estimators=376, learning_rate=0.09997948419313435, num_leaves=35,
                         max_depth=12, min_child_samples=22, subsample=0.8873235866452984,
                         colsample_bytree=0.7962071632765588, reg_alpha=3.361e-08,
                         reg_lambda=1.159e-06),
        "CatBoost": dict(iterations=814, learning_rate=0.06492060953012901, depth=10,
                         l2_leaf_reg=1e-3,  # degenerate 방지: 원본 3.75e-5 → 1e-3 상향
                         bagging_temperature=0.29206227335309387, random_strength=5.620e-06),
    },
    "대사증후군": {
        "LightGBM": dict(n_estimators=902, learning_rate=0.09685139354186116, num_leaves=139,
                         max_depth=12, min_child_samples=53, subsample=0.649352863649425,
                         colsample_bytree=0.7727081638649321, reg_alpha=1.417e-06,
                         reg_lambda=0.0009399152029515449),
        "CatBoost": dict(iterations=922, learning_rate=0.13844752134393362, depth=10,
                         l2_leaf_reg=0.00723482443317559,
                         bagging_temperature=0.708032370100854, random_strength=2.185e-08),
    },
    "간기능이상": {
        "LightGBM": dict(n_estimators=985, learning_rate=0.09792171704582421, num_leaves=137,
                         max_depth=12, min_child_samples=20, subsample=0.8011011032029123,
                         colsample_bytree=0.9633869918248013, reg_alpha=5.296e-05,
                         reg_lambda=0.6975799435019739),
        "CatBoost": dict(iterations=934, learning_rate=0.09330545961114106, depth=10,
                         l2_leaf_reg=0.018218240559437857,
                         bagging_temperature=0.5525877889462425, random_strength=0.7494583751599895),
    },
    "신장단백뇨": {
        "LightGBM": dict(n_estimators=957, learning_rate=0.23146375268766253, num_leaves=109,
                         max_depth=10, min_child_samples=52, subsample=0.857210459075525,
                         colsample_bytree=0.8043497429670368, reg_alpha=3.892069681051425,
                         reg_lambda=0.05620347256425568),
        "CatBoost": dict(iterations=878, learning_rate=0.01596021282510282, depth=8,
                         l2_leaf_reg=1e-3,  # degenerate 방지: 원본 3.14e-7 → 1e-3 상향
                         bagging_temperature=0.8232527440752893, random_strength=3.514e-05),
    },
}

def get_models(scale_pos_weight=1.0, class_weight=None, target_name=None, imb_ensemble=False):
    cw = class_weight

    # B 베스트 파라미터 고정 적용 (타겟명 있을 때)
    if target_name and target_name in BEST_PARAMS_B:
        bp = BEST_PARAMS_B[target_name]
        lgbm_params = {**bp["LightGBM"], "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1}
        catb_params = {**bp["CatBoost"], "random_state": RANDOM_STATE, "verbose": 0}
    else:
        lgbm_params = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                           random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        catb_params = dict(iterations=500, learning_rate=0.05, depth=6,
                           random_state=RANDOM_STATE, verbose=0)

    # class_weight balanced 적용
    # LightGBM: class_weight 파라미터로 전달
    # XGBoost: scale_pos_weight로 변환 (balanced 지원 안 함)
    # CatBoost: auto_class_weights="Balanced"로 변환
    if cw == "balanced":
        lgbm_params = {**lgbm_params, "class_weight": "balanced"}
        xgb_spw     = scale_pos_weight  # 기본값 유지 (타겟별 계산은 루프에서)
        catb_auto   = "Balanced"
    else:
        xgb_spw   = scale_pos_weight
        catb_auto = None

    models = {
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
        "LightGBM": LGBMClassifier(**lgbm_params),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, random_state=RANDOM_STATE,
            n_jobs=-1, eval_metric="logloss",
            verbosity=0, scale_pos_weight=xgb_spw
        ),
        "CatBoost": CatBoostClassifier(**catb_params,
                                        auto_class_weights=catb_auto),
    }

    # 불균형 앙상블 추가 (imb_ensemble 플래그 있을 때만)
    if imb_ensemble:
        models["EasyEnsemble"] = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("model",   EasyEnsembleClassifier(
                n_estimators=10, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ])
        models["BalancedRF"] = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("model",   BalancedRandomForestClassifier(
                n_estimators=300, max_depth=10,
                random_state=RANDOM_STATE, n_jobs=-1
            )),
        ])

    return models

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

# 연도 컬럼 추가 (중복 제거 시 동일 연도 내 중복만 제거하기 위함)
df24["연도"] = 2024
df23["연도"] = 2023
df22["연도"] = 2022
all_cols_with_year = ["연도"] + all_cols

df_all = df24[all_cols_with_year].copy()

for target_name, target_col in TARGETS.items():
    if target_name in ADD_23_POS:
        pos23  = df23[all_cols_with_year].dropna(subset=[target_col])
        pos23  = pos23[pos23[target_col] == 1]
        df_all = pd.concat([df_all, pos23], ignore_index=True)
        print(f"  +2023 {target_name} 양성: {len(pos23):,}행")
    if target_name in ADD_22_POS:
        pos22  = df22[all_cols_with_year].dropna(subset=[target_col])
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
    tag          = exp["tag"]
    use_optuna   = exp.get("optuna", False)
    n_trials     = exp.get("trials", 50)
    tune_models  = exp.get("tune_models", [])
    use_dedup    = exp.get("dedup", False)
    class_weight = exp.get("class_weight", None)
    save_models  = exp.get("save_models", False)
    use_jitter     = exp.get("jitter", False)
    imb_ensemble   = exp.get("imb_ensemble", False)
    oof_ensemble   = exp.get("oof_ensemble", False)
    oof_models     = exp.get("oof_models", ["LightGBM", "CatBoost", "XGBoost"])
    stacking       = exp.get("stacking", False)
    stack_models   = exp.get("stack_models", ["LightGBM", "CatBoost", "XGBoost"])
    stack_folds    = exp.get("stack_folds", 5)
    skip_targets   = exp.get("skip_targets", [])

    # 모델 저장 디렉토리
    if save_models:
        model_save_dir = os.path.join(OUT_DIR, "saved_models")
        os.makedirs(model_save_dir, exist_ok=True)

    # jitter 적용 — train 데이터에만 적용 (test는 원본값 유지)
    # 실제 서비스에서 사용자는 5단위 원본값 입력 → test는 그 조건 유지
    # train에만 다양성 부여 → 모델 일반화 개선 목적

    # 중복 제거 (dedup 플래그 있을 때만)
    # 연도 포함 기준: 동일 연도 내 피처+타겟 완전 동일 행만 제거
    # 연도가 다른 동일 피처값은 중복으로 보지 않음
    if use_dedup:
        before = len(df_all)
        df_all = df_all.drop_duplicates().reset_index(drop=True)
        after  = len(df_all)
        print(f"\n[중복 제거] {before:,}행 → {after:,}행 (제거: {before - after:,}행)")

    print(f"\n{'='*60}")
    print(f"실험: {tag}")
    print(f"  설명: {exp.get('desc', '-')}")
    print(f"{'='*60}")

    for target_name, target_col in TARGETS.items():
        if target_name in skip_targets:
            print(f"\n  [타겟: {target_name}] 스킵")
            continue
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

        # jitter: train에만 적용 — test는 원본값 유지 (실서비스 조건)
        if use_jitter:
            X_train = X_train.copy()
            rng = np.random.default_rng(RANDOM_STATE)
            X_train["신장(5cm단위)"] = X_train["신장(5cm단위)"] + rng.uniform(-2.5, 2.5, len(X_train))
            X_train["체중(5kg단위)"] = X_train["체중(5kg단위)"] + rng.uniform(-2.5, 2.5, len(X_train))
            X_train["bmi"] = (X_train["체중(5kg단위)"] / ((X_train["신장(5cm단위)"] / 100) ** 2)).round(2)
            X_train["waist_height_ratio"] = (X_train["허리둘레"] / X_train["신장(5cm단위)"]).round(3)
            X_train["obesity_combined"] = (
                (X_train["bmi"] >= 25) & (X_train["waist_height_ratio"] >= 0.5)
            ).astype(float)
            print(f"  [jitter] train에만 ±2.5 균등분포 노이즈 적용")

        target_results = []

        # ── 일반 모델 ──────────────────────────────────────────
        for model_name, model in get_models(target_name=target_name, class_weight=class_weight, imb_ensemble=imb_ensemble).items():
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

        # ── OOF 확률 평균 앙상블 ─────────────────────────────
        if oof_ensemble:
            oof_probs = []
            for oof_model_name in oof_models:
                oof_m = get_models(target_name=target_name, class_weight=class_weight).get(oof_model_name)
                if oof_m is None:
                    continue
                oof_m.fit(X_train, y_train)
                if hasattr(oof_m, "predict_proba"):
                    oof_probs.append(oof_m.predict_proba(X_test)[:, 1])

            if len(oof_probs) >= 2:
                y_prob_ens = np.mean(oof_probs, axis=0)
                auc_ens    = roc_auc_score(y_test, y_prob_ens)
                thr_ens    = find_best_threshold(y_test, y_prob_ens)
                y_pred_ens = (y_prob_ens >= thr_ens).astype(int)
                recall_ens = recall_score(y_test, y_pred_ens, zero_division=0)
                f1_ens     = f1_score(y_test, y_pred_ens, zero_division=0)
                prec_ens   = precision_score(y_test, y_pred_ens, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_ens).ravel()

                model_label = "+".join(oof_models)
                recall_ok = "✅" if recall_ens >= RECALL_TARGET else "❌"
                f1_ok     = "✅" if f1_ens >= 0.6 else "❌"
                print(f"  {'OOF_Ensemble':<18} AUC: {auc_ens:.4f} | Recall: {recall_ens:.4f} {recall_ok} | "
                      f"F1: {f1_ens:.4f} {f1_ok} | Precision: {prec_ens:.4f} | Threshold: {thr_ens:.2f}")
                print(f"    └ ({model_label} 평균)")

                ens_result = {
                    "실험": tag, "타겟": target_name, "모델": "OOF_Ensemble",
                    "AUC": round(auc_ens,4), "Recall": round(recall_ens,4),
                    "F1": round(f1_ens,4), "Precision": round(prec_ens,4),
                    "Threshold": thr_ens,
                    "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                    "Recall_달성": recall_ens >= RECALL_TARGET, "F1_달성": f1_ens >= 0.6,
                }
                target_results.append(ens_result)
                all_results.append(ens_result)

        # ── OOF 스태킹 앙상블 ───────────────────────────────────
        if stacking:
            from sklearn.model_selection import StratifiedKFold
            from sklearn.linear_model import LogisticRegression as MetaLR

            skf = StratifiedKFold(n_splits=stack_folds, shuffle=True, random_state=RANDOM_STATE)
            oof_train_probs = np.zeros((len(X_train), len(stack_models)))
            oof_test_probs  = np.zeros((len(X_test),  len(stack_models)))

            print(f"  [스태킹] {stack_folds}-fold OOF 학습 중...")
            for mi, smodel_name in enumerate(stack_models):
                fold_test_probs = np.zeros((len(X_test), stack_folds))
                for fi, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
                    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

                    smodel = get_models(target_name=target_name, class_weight=class_weight).get(smodel_name)
                    if smodel is None:
                        continue

                    # KNN impute per fold
                    imp = KNNImputer(n_neighbors=5)
                    X_tr_imp = imp.fit_transform(X_tr)
                    X_va_imp = imp.transform(X_va)
                    X_te_imp = imp.transform(X_test)

                    smodel.fit(X_tr_imp, y_tr)
                    oof_train_probs[va_idx, mi] = smodel.predict_proba(X_va_imp)[:, 1]
                    fold_test_probs[:, fi]       = smodel.predict_proba(X_te_imp)[:, 1]

                oof_test_probs[:, mi] = fold_test_probs.mean(axis=1)

            # 메타 모델 학습 (LR)
            meta_model = MetaLR(max_iter=1000, random_state=RANDOM_STATE)

            # train OOF에도 KNN impute 필요 없음 (이미 처리됨)
            meta_model.fit(oof_train_probs, y_train)
            y_prob_stack = meta_model.predict_proba(oof_test_probs)[:, 1]

            auc_st    = roc_auc_score(y_test, y_prob_stack)
            thr_st    = find_best_threshold(y_test, y_prob_stack)
            y_pred_st = (y_prob_stack >= thr_st).astype(int)
            recall_st = recall_score(y_test, y_pred_st, zero_division=0)
            f1_st     = f1_score(y_test, y_pred_st, zero_division=0)
            prec_st   = precision_score(y_test, y_pred_st, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_st).ravel()

            recall_ok = "✅" if recall_st >= RECALL_TARGET else "❌"
            f1_ok     = "✅" if f1_st >= 0.6 else "❌"
            model_label = "+".join(stack_models)
            print(f"  {'Stacking':<18} AUC: {auc_st:.4f} | Recall: {recall_st:.4f} {recall_ok} | "
                  f"F1: {f1_st:.4f} {f1_ok} | Precision: {prec_st:.4f} | Threshold: {thr_st:.2f}")
            print(f"    └ ({model_label} → LR 메타모델)")

            stack_result = {
                "실험": tag, "타겟": target_name, "모델": "Stacking",
                "AUC": round(auc_st,4), "Recall": round(recall_st,4),
                "F1": round(f1_st,4), "Precision": round(prec_st,4),
                "Threshold": thr_st,
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                "Recall_달성": recall_st >= RECALL_TARGET, "F1_달성": f1_st >= 0.6,
            }
            target_results.append(stack_result)
            all_results.append(stack_result)

        best = max(target_results, key=lambda x: (x["Recall_달성"], x["F1"]))
        print(f"\n  → [{target_name}] 최고: {best['모델']} "
              f"AUC {best['AUC']:.4f} / Recall {best['Recall']:.4f} / F1 {best['F1']:.4f}")

        # 최고 모델 저장 (save_models 플래그 있을 때만)
        if save_models:
            best_model_name = best["모델"]

            # 스태킹인 경우 — base 모델들 + 메타 모델 저장
            if best_model_name == "Stacking" and stacking:
                stack_save = {
                    "base_models": {},
                    "meta_model":  meta_model,
                    "stack_models": stack_models,
                    "stack_folds":  stack_folds,
                    "feature_cols": FEATURE_COLS,
                }
                # base 모델 전체 재학습 후 저장
                imp_full = KNNImputer(n_neighbors=5)
                X_train_imp = imp_full.fit_transform(X_train)
                X_test_imp  = imp_full.transform(X_test)
                stack_save["imputer"] = imp_full
                for sn in stack_models:
                    sm = get_models(target_name=target_name, class_weight=class_weight).get(sn)
                    if sm:
                        sm.fit(X_train_imp, y_train)
                        stack_save["base_models"][sn] = sm
                save_path = os.path.join(model_save_dir, f"{tag}_{target_name}_stacking.pkl")
                joblib.dump(stack_save, save_path)
                print(f"    스태킹 모델 저장: {save_path}")

            # 단일 모델인 경우
            else:
                save_model_obj = None
                for model_name, model in get_models(target_name=target_name, class_weight=class_weight).items():
                    if model_name == best_model_name:
                        model.fit(X_train, y_train)
                        save_model_obj = model
                        break
                if save_model_obj is not None:
                    save_path = os.path.join(model_save_dir, f"{tag}_{target_name}_best.pkl")
                    joblib.dump(save_model_obj, save_path)
                    print(f"    모델 저장: {save_path}")

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
