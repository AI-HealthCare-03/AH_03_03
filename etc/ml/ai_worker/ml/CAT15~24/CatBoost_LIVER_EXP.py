"""
간기능이상유병 예측 — CatBoost 실험 관리 (hn15~24 통합)
Python 3.13 | catboost>=1.2 | scikit-learn>=1.4

실험 단계:
  Base : 기본 FE (나이구간+BMI구간+음주위험군+걷기+가족력+BMI_X_나이)
  A    : Base + 총음주량 파생 (음주빈도 × 음주량)
  B    : A + 하이퍼파라미터 튜닝 (iterations↑ depth↑ lr↓)
  C    : Optuna 자동 탐색 (n_trials=50, Recall>=0.80 조건 F1 최대화)

실험별 결과는 outputs/catboost_LIVER_{tag}/ 에 저장
"""

import os
import warnings
from pathlib import Path
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

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ optuna 없음 — pip install optuna")

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────
DATA_PATH: str = str(
    Path(__file__).parent.parent.parent / "data" / "hn_master.parquet"
)
OUTPUT_ROOT: str = str(
    Path(__file__).parent / "outputs"
)

# ── 고정 설정 ──────────────────────────────────────────────────
TARGET: str = "간기능이상유병"
N_SPLITS: int = 5
SEED: int = 42
RECALL_MIN: float = 0.80
THRESHOLD_RANGE: NDArray[np.float64] = np.arange(0.20, 0.71, 0.01)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 목록
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERIMENTS = [
    # ── Base: 기본 FE ─────────────────────────────────────────
    # {
    #     "tag":          "Base",
    #     "desc":         "기본 FE — 나이구간+BMI구간+음주위험군+걷기+가족력+BMI_X_나이",
    #     "total_alcohol": False,
    #     "params": dict(
    #         iterations=500, learning_rate=0.05, depth=6,
    #     ),
    #     "smote": False,
    # },

    # # ── A: Base + 총음주량 파생 ───────────────────────────────
    # {
    #     "tag":           "A",
    #     "desc":          "Base + 총음주량 파생 (음주빈도 × 음주량)",
    #     "total_alcohol": True,
    #     "params": dict(
    #         iterations=500, learning_rate=0.05, depth=6,
    #     ),
    #     "smote": False,
    # },

    # # ── B: A + 하이퍼파라미터 튜닝 ───────────────────────────
    # {
    #     "tag":           "B",
    #     "desc":          "A + 하이퍼파라미터 튜닝 (iterations=1000, depth=8, lr=0.03)",
    #     "total_alcohol": True,
    #     "params": dict(
    #         iterations=1000, learning_rate=0.03, depth=8,
    #     ),
    #     "smote": False,
    # },

    # ── C: Optuna 자동 탐색 ──────────────────────────────────
    # {
    #     "tag":           "C",
    #     "desc":          "Optuna 자동 탐색 (n_trials=50) — Recall>=0.80 조건 F1 최대화",
    #     "total_alcohol": True,
    #     "params":        {},        # Optuna가 결정
    #     "smote":         False,
    #     "optuna":        True,
    #     "optuna_trials": 50,
    # },

    # ── D: C 베스트 파라미터 + 추가 피처 ────────────────────
    # {
    #     "tag":           "D",
    #     "desc":          "C 베스트 파라미터 + 추가 파생 피처 (WHtR/나이×성별/복합위험/활동량)",
    #     "total_alcohol": True,
    #     "extra_fe":      True,
    #     "params": dict(
    #         iterations=415,
    #         learning_rate=0.0984167875692493,
    #         depth=4,
    #         l2_leaf_reg=8.912269345709221,
    #         bagging_temperature=0.073521389236827,
    #     ),
    #     "smote":  False,
    #     "optuna": False,
    # },

    # ── E: D 피처셋 + Optuna 재탐색 ──────────────────────────
    # {
    #     "tag":           "E",
    #     "desc":          "D 피처셋(41개) + Optuna 재탐색 (n_trials=100)",
    #     "total_alcohol": True,
    #     "extra_fe":      True,
    #     "params":        {},
    #     "smote":         False,
    #     "optuna":        True,
    #     "optuna_trials": 100,
    # },

    # ── F: 타겟 재정의 (바이러스성 간염 제외) + Optuna ──────────
    {
        "tag":           "F",
        "desc":          "타겟 재정의 (HBsAg/HCV 제외) + D 피처셋 + Optuna (n_trials=100)",
        "total_alcohol": True,
        "extra_fe":      True,
        "target_redef":  True,   # 바이러스성 간염 제외
        "params":        {},
        "smote":         False,
        "optuna":        True,
        "optuna_trials": 100,
    },

]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 컬럼 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USE_COLS = [
    "survey_year",
    "sex", "age", "occp",
    "HE_ht", "HE_wt", "HE_BMI", "HE_wc", "HE_obe",
    "sm_presnt", "BD1_11", "BD2_1",
    "BE3_31", "BE5_1",
    "HE_HPfh1", "HE_HPfh2", "HE_HPfh3",
    "HE_DMfh1", "HE_DMfh2", "HE_DMfh3",
    "HE_HLfh1", "HE_HLfh2", "HE_HLfh3",
    "DI1_pr", "DL1_dg", "DE1_pr", "DI2_pr",
    "HE_ast", "HE_alt", "HE_ALP",
    "HE_hepaB", "HE_hepaC",
]

COL_RENAME = {
    "survey_year": "연도",
    "sex": "성별", "age": "나이", "occp": "직업",
    "HE_ht": "키", "HE_wt": "체중", "HE_BMI": "BMI",
    "HE_wc": "허리둘레", "HE_obe": "비만단계",
    "sm_presnt": "현재흡연", "BD1_11": "음주빈도", "BD2_1": "음주량",
    "BE3_31": "걷기일수", "BE5_1": "근력운동일수",
    "HE_HPfh1": "고혈압가족력_부", "HE_HPfh2": "고혈압가족력_모", "HE_HPfh3": "고혈압가족력_형제",
    "HE_DMfh1": "당뇨가족력_부",   "HE_DMfh2": "당뇨가족력_모",   "HE_DMfh3": "당뇨가족력_형제",
    "HE_HLfh1": "고지혈증가족력_부","HE_HLfh2": "고지혈증가족력_모","HE_HLfh3": "고지혈증가족력_형제",
    "DI1_pr": "고혈압유병", "DL1_dg": "간질환진단",
    "DE1_pr": "당뇨유병", "DI2_pr": "이상지질혈증유병",
    "HE_ast": "AST", "HE_alt": "ALT", "HE_ALP": "ALP",
    "HE_hepaB": "B형간염항원", "HE_hepaC": "C형간염항체",
}

INT_COLS = [
    "성별", "나이", "직업", "현재흡연", "음주빈도", "음주량", "걷기일수", "근력운동일수",
    "비만단계", "간질환진단", "B형간염항원", "C형간염항체",
    "고혈압가족력_부", "고혈압가족력_모", "고혈압가족력_형제",
    "당뇨가족력_부", "당뇨가족력_모", "당뇨가족력_형제",
    "고지혈증가족력_부", "고지혈증가족력_모", "고지혈증가족력_형제",
    "고혈압유병", "당뇨유병", "이상지질혈증유병",
]

DROP_COLS_MODEL = [
    "간기능이상유병", "고혈압유병", "당뇨유병", "이상지질혈증유병",
    "비만단계", "연도",
    "AST", "ALT", "ALP", "B형간염항원", "C형간염항체", "간질환진단",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 연도별 필터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def apply_year_filter(df: pd.DataFrame) -> pd.DataFrame:
    df_full = df[df["연도"].isin([2022, 2023, 2024])].copy()

    df_21 = df[df["연도"] == 2021].copy()
    if "당뇨유병" in df_21.columns:
        cond_dm_21 = df_21["당뇨유병"] == 1
    else:
        cond_dm_21 = pd.Series(False, index=df_21.index)
    if "이상지질혈증유병" in df_21.columns:
        cond_dysl_21 = df_21["이상지질혈증유병"] == 1
    else:
        cond_dysl_21 = pd.Series(False, index=df_21.index)
    if "간질환진단" in df_21.columns:
        cond_liver_21 = df_21["간질환진단"] == 1
    else:
        cond_liver_21 = pd.Series(False, index=df_21.index)
    df_21 = df_21[cond_dm_21 | cond_dysl_21 | cond_liver_21].reset_index(drop=True)

    df_1520 = df[df["연도"].isin([2015, 2016, 2017, 2018, 2019, 2020])].copy()
    if "당뇨유병" in df_1520.columns:
        cond_dm_1520 = df_1520["당뇨유병"] == 1
    else:
        cond_dm_1520 = pd.Series(False, index=df_1520.index)
    if "간질환진단" in df_1520.columns:
        cond_liver_1520 = df_1520["간질환진단"] == 1
    else:
        cond_liver_1520 = pd.Series(False, index=df_1520.index)
    df_1520 = df_1520[cond_dm_1520 | cond_liver_1520].reset_index(drop=True)

    df_filtered = pd.concat([df_full, df_21, df_1520], ignore_index=True)
    print(f"  hn22~24 전체     : {len(df_full):,}명")
    print(f"  hn21 유병 필터   : {len(df_21):,}명")
    print(f"  hn15~20 유병 필터: {len(df_1520):,}명")
    print(f"  최종             : {len(df_filtered):,}명")
    return df_filtered


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 타겟 파생
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def derive_target(df: pd.DataFrame, target_redef: bool = False) -> pd.DataFrame:
    """
    target_redef=False (기본): AST/ALT/ALP/HBsAg/HCV/간질환진단 복합 OR
    target_redef=True  (F단계): HBsAg/HCV 제외 — 생활습관 예측 가능한 것만
    """
    cond_map = {}
    if "AST"      in df.columns: cond_map["AST>40"]   = df["AST"]  > 40
    if "ALT"      in df.columns: cond_map["ALT>40"]   = df["ALT"]  > 40
    if "ALP"      in df.columns: cond_map["ALP>120"]  = df["ALP"]  > 120
    if not target_redef:
        if "B형간염항원" in df.columns: cond_map["HBsAg양성"] = df["B형간염항원"] == 1
        if "C형간염항체" in df.columns: cond_map["HCV양성"]   = df["C형간염항체"] == 1
    if "간질환진단"  in df.columns: cond_map["간질환진단"]  = df["간질환진단"]  == 1

    if not cond_map:
        raise ValueError("타겟 파생 가능한 변수 없음")

    combined = list(cond_map.values())[0]
    for c in list(cond_map.values())[1:]:
        combined = combined | c
    df[TARGET] = combined.astype(int)

    print("  조건별 해당자:")
    for name, cond in cond_map.items():
        n = int(cond.sum())
        print(f"    {name:<12}: {n:,}명 ({n/len(df)*100:.1f}%)")

    vc = df[TARGET].value_counts().sort_index()
    n0, n1 = vc.get(0, 0), vc.get(1, 0)
    print(f"  [타겟] 정상={n0:,} / 이상={n1:,} | 양성률 {n1/(n0+n1)*100:.1f}% | 불균형 1:{n0/max(n1,1):.1f}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature Engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def apply_feature_engineering(df: pd.DataFrame, total_alcohol: bool = False, extra_fe: bool = False) -> pd.DataFrame:
    added: list[str] = []

    # 나이 구간화
    age_bins = [0, 40, 50, 60, 70, 80, np.inf]
    age_labels = ["나이_19_39", "나이_40대", "나이_50대", "나이_60대", "나이_70대", "나이_80이상"]
    df["_나이구간"] = pd.cut(df["나이"], bins=age_bins, labels=age_labels, right=False)
    for label in age_labels:
        df[label] = (df["_나이구간"] == label).astype(int)
    df = df.drop(columns=["_나이구간"])
    added += age_labels
    print("  [ON] 나이 구간화")

    # BMI 구간화
    df["BMI_구간"] = pd.cut(
        df["BMI"], bins=[0, 23, 25, 30, np.inf], labels=[0, 1, 2, 3], right=False
    ).astype(float)
    added += ["BMI_구간"]
    print("  [ON] BMI 구간화")

    # 음주위험군
    df["음주위험군"] = pd.cut(
        df["음주빈도"], bins=[-np.inf, 0, 2, np.inf], labels=[0, 1, 2], right=True
    ).astype(float)
    added += ["음주위험군"]
    print("  [ON] 음주위험군")

    # 총음주량 (A단계 이후)
    if total_alcohol:
        freq = df["음주빈도"].copy()
        freq = freq.where(freq.isin([1, 2, 3, 4, 5, 6]), other=0)
        df["총음주량"] = freq * df["음주량"].fillna(0)
        added += ["총음주량"]
        print("  [ON] 총음주량 (빈도 × 양)")

    # 걷기활동량
    df["걷기활동량"] = pd.cut(
        df["걷기일수"], bins=[-np.inf, 0, 3, np.inf], labels=[0, 1, 2], right=True
    ).astype(float)
    added += ["걷기활동량"]
    print("  [ON] 걷기활동량")

    # 가족력 합산
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
    print("  [ON] 가족력 합산")

    # BMI × 나이
    df["BMI_X_나이"] = df["BMI"] * df["나이"]
    added += ["BMI_X_나이"]
    print("  [ON] BMI × 나이")

    # ── D단계 추가 파생 피처 ──────────────────────────────────
    if extra_fe:
        # 1) WHtR — 허리둘레/키 비율 (복부비만 지표, 지방간 직접 연관)
        if "허리둘레" in df.columns and "키" in df.columns:
            df["WHtR"] = df["허리둘레"] / (df["키"] + 1e-6)
            added += ["WHtR"]
            print("  [ON] WHtR (허리둘레/키)")

        # 2) 나이 × 성별 (남성 중년 고위험군 포착)
        if "나이" in df.columns and "성별" in df.columns:
            df["나이_X_성별"] = df["나이"] * df["성별"]
            added += ["나이_X_성별"]
            print("  [ON] 나이 × 성별")

        # 3) BMI × 음주빈도 (비만 + 음주 복합 위험)
        if "BMI" in df.columns and "음주빈도" in df.columns:
            df["BMI_X_음주"] = df["BMI"] * df["음주빈도"].fillna(0)
            added += ["BMI_X_음주"]
            print("  [ON] BMI × 음주빈도")

        # 4) 음주빈도 × 현재흡연 (복합 생활습관 위험)
        if "음주빈도" in df.columns and "현재흡연" in df.columns:
            df["음주_X_흡연"] = df["음주빈도"].fillna(0) * df["현재흡연"].fillna(0)
            added += ["음주_X_흡연"]
            print("  [ON] 음주빈도 × 현재흡연")

        # 5) 총 신체활동량 (걷기 + 근력운동 합산)
        if "걷기일수" in df.columns and "근력운동일수" in df.columns:
            df["총활동일수"] = df["걷기일수"].fillna(0) + df["근력운동일수"].fillna(0)
            added += ["총활동일수"]
            print("  [ON] 총활동일수 (걷기+근력)")

        # 6) 비만 × 나이구간 고위험 플래그 (40대이상 + BMI>=25)
        if "BMI" in df.columns and "나이" in df.columns:
            df["비만_중장년"] = (
                (df["BMI"] >= 25) & (df["나이"] >= 40)
            ).astype(float)
            df.loc[df["BMI"].isna() | df["나이"].isna(), "비만_중장년"] = np.nan
            added += ["비만_중장년"]
            print("  [ON] 비만_중장년 (BMI>=25 & 나이>=40)")

    print(f"  → 추가 피처 수: {len(added)} | {added}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Threshold 튜닝
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단일 실험 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_experiment(
    exp: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    tag   = exp["tag"]
    desc  = exp["desc"]
    model_dir = os.path.join(OUTPUT_ROOT, f"catboost_LIVER_{tag}")
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'━'*60}")
    print(f"실험 [{tag}] {desc}")
    print(f"{'━'*60}")

    X_tr = X_train.copy()
    y_tr = y_train.copy()

    # SMOTE
    if exp.get("smote", False):
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=SEED)
            X_tr_arr, y_tr_arr = sm.fit_resample(X_tr, y_tr)
            X_tr = pd.DataFrame(X_tr_arr, columns=X_tr.columns)
            y_tr = pd.Series(y_tr_arr)
            print(f"  SMOTE 적용 후 Train: {X_tr.shape} | 양성: {y_tr.mean():.4f}")
        except ImportError:
            print("  ⚠️ imbalanced-learn 없음 — SMOTE 건너뜀")

    # 클래스 가중치
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    ratio = float(neg / pos)

    # ── Optuna 자동 탐색 ──────────────────────────────────────
    if exp.get("optuna", False) and OPTUNA_AVAILABLE:
        n_trials = exp.get("optuna_trials", 50)
        print(f"  Optuna 탐색 시작 | trials={n_trials} | RECALL_MIN={RECALL_MIN}")

        def objective(trial: "optuna.Trial") -> float:
            trial_params: dict[str, Any] = dict(
                iterations=trial.suggest_int("iterations", 300, 1500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                depth=trial.suggest_int("depth", 4, 10),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                loss_function="Logloss",
                eval_metric="AUC",
                class_weights={0: 1.0, 1: ratio},
                early_stopping_rounds=50,
                random_seed=SEED,
                verbose=False,
                allow_writing_files=False,
            )

            skf_opt = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            fold_f1s: list[float] = []

            for tr_idx_opt, val_idx_opt in skf_opt.split(X_tr, y_tr):
                X_opt_tr  = X_tr.iloc[tr_idx_opt]
                X_opt_val = X_tr.iloc[val_idx_opt]
                y_opt_tr  = y_tr.iloc[tr_idx_opt]
                y_opt_val = y_tr.iloc[val_idx_opt]

                opt_model = CatBoostClassifier(**trial_params)
                opt_model.fit(
                    Pool(X_opt_tr, y_opt_tr),
                    eval_set=Pool(X_opt_val, y_opt_val),
                )

                val_proba_opt = opt_model.predict_proba(X_opt_val)[:, 1]

                best_fold_f1: float = 0.0
                for thr in THRESHOLD_RANGE:
                    pred = (val_proba_opt >= thr).astype(int)
                    r = recall_score(y_opt_val, pred)
                    f = f1_score(y_opt_val, pred, zero_division=0)
                    if r >= RECALL_MIN and f > best_fold_f1:
                        best_fold_f1 = f

                fold_f1s.append(best_fold_f1)

            return float(np.mean(fold_f1s))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        print(f"  Optuna 완료 | best_val_f1={study.best_value:.4f}")
        print(f"  best_params: {best}")

        best_params_df = pd.DataFrame([best])
        best_params_df.to_csv(os.path.join(model_dir, "optuna_best_params.csv"), index=False)

        params: dict[str, Any] = dict(
            iterations=best["iterations"],
            learning_rate=best["learning_rate"],
            depth=best["depth"],
            l2_leaf_reg=best["l2_leaf_reg"],
            bagging_temperature=best["bagging_temperature"],
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights={0: 1.0, 1: ratio},
            early_stopping_rounds=50,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )

    else:
        # 고정 파라미터
        base_params = exp.get("params", {})
        params: dict[str, Any] = dict(
            iterations=base_params.get("iterations", 500),
            learning_rate=base_params.get("learning_rate", 0.05),
            depth=base_params.get("depth", 6),
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights={0: 1.0, 1: ratio},
            early_stopping_rounds=50,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )

    print(f"  params: iterations={params['iterations']} | lr={params['learning_rate']:.4f} | depth={params['depth']}")

    # 5-Fold OOF
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba: NDArray[np.float64] = np.zeros(len(y_tr))
    fold_models = []
    fold_scores = []

    print(f"\n  5-Fold OOF | RECALL_MIN={RECALL_MIN}")
    print(f"  {'─'*55}")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr), 1):
        X_f_tr, X_f_val = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        y_f_tr, y_f_val = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]

        train_pool = Pool(X_f_tr, y_f_tr)
        val_pool   = Pool(X_f_val, y_f_val)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        val_proba: NDArray[np.float64] = model.predict_proba(X_f_val)[:, 1]
        val_label = (val_proba >= 0.5).astype(int)
        oof_proba[val_idx] = val_proba
        fold_models.append(model)

        fold_scores.append({
            "fold":      fold,
            "auc":       round(float(roc_auc_score(y_f_val, val_proba)), 4),
            "recall":    round(float(recall_score(y_f_val, val_label)), 4),
            "precision": round(float(precision_score(y_f_val, val_label, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_f_val, val_label)), 4),
            "best_iter": model.best_iteration_,
        })
        print(
            f"  Fold {fold} | AUC: {fold_scores[-1]['auc']:.4f} | "
            f"Recall: {fold_scores[-1]['recall']:.4f} | "
            f"Prec: {fold_scores[-1]['precision']:.4f} | "
            f"F1: {fold_scores[-1]['f1']:.4f} | "
            f"iter: {model.best_iteration_}"
        )

    scores_df = pd.DataFrame(fold_scores)

    # OOF 전체 성능
    print(f"  {'─'*55}")
    oof_label_05 = (oof_proba >= 0.5).astype(int)
    oof_auc = float(roc_auc_score(y_tr, oof_proba))
    print(f"  OOF (thr=0.5) | AUC: {oof_auc:.4f} | Recall: {recall_score(y_tr, oof_label_05):.4f} | F1: {f1_score(y_tr, oof_label_05):.4f}")
    print(f"  fold avg      | AUC: {scores_df['auc'].mean():.4f}±{scores_df['auc'].std():.4f}")

    # Threshold 탐색
    best_thr, best_f1, oof_recall, oof_prec = tune_threshold(oof_proba, y_tr.values)
    print(f"  Threshold 탐색 | best_thr={best_thr:.2f} | Recall={oof_recall:.4f} | Prec={oof_prec:.4f} | F1={best_f1:.4f}")

    # Test 평가
    test_probas = np.column_stack([m.predict_proba(X_test)[:, 1] for m in fold_models])
    test_proba_ens: NDArray[np.float64] = test_probas.mean(axis=1)
    test_label = (test_proba_ens >= best_thr).astype(int)

    test_auc    = float(roc_auc_score(y_test, test_proba_ens))
    test_recall = float(recall_score(y_test, test_label))
    test_prec   = float(precision_score(y_test, test_label, zero_division=0))
    test_f1     = float(f1_score(y_test, test_label))
    cm = confusion_matrix(y_test, test_label)

    print(f"\n  Test (thr={best_thr:.2f})")
    print(f"  AUC={test_auc:.4f} | Recall={test_recall:.4f} | Prec={test_prec:.4f} | F1={test_f1:.4f}")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    print(classification_report(y_test, test_label, target_names=["정상(0)", "간기능이상(1)"]))

    # Feature Importance
    fi_df = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": fold_models[-1].get_feature_importance(),
    }).sort_values("importance", ascending=False)
    print(f"  Feature Importance Top 15:")
    print(fi_df.head(15).to_string(index=False))

    # 저장
    scores_df.to_csv(os.path.join(model_dir, "fold_scores.csv"), index=False)
    fi_df.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
    np.save(os.path.join(model_dir, "oof_y_true.npy"),     y_tr.values)
    np.save(os.path.join(model_dir, "oof_proba.npy"),      oof_proba)
    np.save(os.path.join(model_dir, "best_threshold.npy"), np.array([best_thr]))

    # fold 모델 저장
    for i, m in enumerate(fold_models, 1):
        m.save_model(os.path.join(model_dir, f"model_fold{i}.cbm"))

    print(f"\n  저장 완료 → {model_dir}")

    return {
        "tag":        tag,
        "oof_auc":    round(oof_auc, 4),
        "oof_recall": round(oof_recall, 4),
        "oof_prec":   round(oof_prec, 4),
        "oof_f1":     round(best_f1, 4),
        "test_auc":   round(test_auc, 4),
        "test_recall":round(test_recall, 4),
        "test_prec":  round(test_prec, 4),
        "test_f1":    round(test_f1, 4),
        "best_thr":   best_thr,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    # ── 마스터 로드 ────────────────────────────────────────────
    print("[0] 마스터 데이터 로드")
    df_raw = pd.read_parquet(DATA_PATH)
    print(f"    shape: {df_raw.shape}")

    # ── 컬럼 선택 + rename ─────────────────────────────────────
    avail = [c for c in USE_COLS if c in df_raw.columns]
    df_raw = df_raw[avail].rename(columns=COL_RENAME)

    for c in INT_COLS:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

    df_raw = df_raw[df_raw["나이"] >= 19].reset_index(drop=True)

    for col in ["고혈압유병", "당뇨유병", "이상지질혈증유병"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].map({1.0: 1, 1: 1, 8.0: 0, 8: 0})

    if "간질환진단" in df_raw.columns:
        df_raw.loc[df_raw["간질환진단"] == 9.0, "간질환진단"] = np.nan

    # ── 연도별 필터 ────────────────────────────────────────────
    print("\n[1] 연도별 필터 적용")
    df_base = apply_year_filter(df_raw)

    # ── 타겟 파생 (실험별 재정의 여부 반영) ───────────────────
    # F단계처럼 target_redef 있는 실험이 있으면 실험 루프 안에서 재파생
    # 기본은 전체 조건으로 파생
    first_redef = any(exp.get("target_redef", False) for exp in EXPERIMENTS)
    all_redef   = all(exp.get("target_redef", False) for exp in EXPERIMENTS)

    if all_redef:
        # 모든 실험이 재정의 타겟 사용
        print("\n[2] 타겟 파생 (재정의: HBsAg/HCV 제외)")
        df_base = derive_target(df_base, target_redef=True)
    else:
        # 기본 타겟으로 파생 (실험별 재파생은 루프 안에서)
        print("\n[2] 타겟 파생 (기본: 전체 조건)")
        df_base = derive_target(df_base, target_redef=False)

    df_base = df_base.dropna(subset=[TARGET]).reset_index(drop=True)

    # ── Train/Test 분리 (실험 전 고정) ─────────────────────────
    # 모든 실험이 동일한 분리 기준 사용
    y_full = df_base[TARGET].astype(int)
    train_idx, test_idx = train_test_split(
        df_base.index, test_size=0.2, random_state=SEED, stratify=y_full
    )
    df_train_base = df_base.loc[train_idx].reset_index(drop=True)
    df_test_base  = df_base.loc[test_idx].reset_index(drop=True)

    print(f"\n[3] Train/Test 고정 분리 | Train={len(df_train_base):,} | Test={len(df_test_base):,}")
    print(f"    Train 양성: {df_train_base[TARGET].mean():.4f} | Test 양성: {df_test_base[TARGET].mean():.4f}")

    # ── 실험 루프 ──────────────────────────────────────────────
    all_results = []

    for exp in EXPERIMENTS:
        tag          = exp["tag"]
        total_alcohol = exp.get("total_alcohol", False)
        target_redef  = exp.get("target_redef", False)

        # target_redef 실험은 타겟 재파생 후 Train/Test 재분리
        if target_redef:
            print(f"\n[타겟 재정의 — {tag}] HBsAg/HCV 제외")
            df_redef = df_base.copy()
            df_redef = df_redef.drop(columns=[TARGET], errors="ignore")
            df_redef = derive_target(df_redef, target_redef=True)
            df_redef = df_redef.dropna(subset=[TARGET]).reset_index(drop=True)

            y_redef = df_redef[TARGET].astype(int)
            train_idx_r, test_idx_r = train_test_split(
                df_redef.index, test_size=0.2, random_state=SEED, stratify=y_redef
            )
            df_tr = df_redef.loc[train_idx_r].reset_index(drop=True)
            df_te = df_redef.loc[test_idx_r].reset_index(drop=True)
            print(f"  재정의 타겟 | Train={len(df_tr):,} | Test={len(df_te):,}")
            vc = y_redef.value_counts().sort_index()
            n0, n1 = vc.get(0,0), vc.get(1,0)
            print(f"  정상={n0:,} / 이상={n1:,} | 양성률 {n1/(n0+n1)*100:.1f}% | 불균형 1:{n0/max(n1,1):.1f}")
        else:
            df_tr = df_train_base.copy()
            df_te = df_test_base.copy()

        extra_fe = exp.get("extra_fe", False)
        print(f"\n[FE — {tag}] 피처 엔지니어링")
        df_tr = apply_feature_engineering(df_tr, total_alcohol=total_alcohol, extra_fe=extra_fe)
        df_te = apply_feature_engineering(df_te, total_alcohol=total_alcohol, extra_fe=extra_fe)

        # X/y 분리
        drop_cols = [c for c in DROP_COLS_MODEL if c in df_tr.columns]
        X_train = df_tr.drop(columns=drop_cols)
        y_train = df_tr[TARGET].astype(int)
        X_test  = df_te.drop(columns=drop_cols)
        y_test  = df_te[TARGET].astype(int)

        print(f"  Feature 수: {X_train.shape[1]} | {list(X_train.columns)}")

        result = run_experiment(exp, X_train, X_test, y_train, y_test)
        all_results.append(result)

    # ── 실험 결과 비교 ─────────────────────────────────────────
    print(f"\n{'━'*60}")
    print("실험 결과 비교")
    print(f"{'━'*60}")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    summary_path = os.path.join(OUTPUT_ROOT, "experiment_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\n결과 요약 저장 → {summary_path}")


if __name__ == "__main__":
    main()
