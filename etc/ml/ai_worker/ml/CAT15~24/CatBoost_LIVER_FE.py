"""
간기능이상유병 예측 — CatBoost + FE 베이스라인 (hn15~24 통합)
Python 3.13 | catboost>=1.2 | scikit-learn>=1.4

데이터: hn_master.parquet (연도별 필터 적용)
  - hn22~24: 전체 인구
  - hn21   : 당뇨 OR 이상지질혈증 OR 간질환 유병
  - hn15~20: 당뇨 OR 간질환 유병

타겟 파생 (복합 OR):
  AST>40 | ALT>40 | ALP>120 | B형간염항원==1 | C형간염항체==1 | 간질환진단==1

FE 조합: 나이구간 + BMI구간 + 음주위험군 + 걷기활동량 + 가족력합산 + BMI_X_나이
피처: 생활습관/인구통계 기반 (혈액수치 제외 — LDL/HDL/콜레스테롤/중성지방/공복혈당)
Threshold: RECALL_MIN 0.80, F1 최대화
검증: Train/Test 8:2 Hold-out → Train 내부 5-Fold OOF
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

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────
DATA_PATH: str = str(
    Path(__file__).parent.parent.parent / "data" / "hn_master.parquet"
)
MODEL_DIR: str = str(
    Path(__file__).parent / "outputs" / "catboost_LIVER_FE"
)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 설정 ───────────────────────────────────────────────────────
TARGET: str = "간기능이상유병"
N_SPLITS: int = 5
SEED: int = 42
RECALL_MIN: float = 0.80        # 서비스 스크리닝 목적 — Recall 우선
THRESHOLD_RANGE: NDArray[np.float64] = np.arange(0.20, 0.71, 0.01)

# ── FE 플래그 ──────────────────────────────────────────────────
USE_AGE_BIN: bool = True
USE_BMI_BIN: bool = True
USE_WEIGHT_BIN: bool = False
USE_ALCOHOL_RISK: bool = True
USE_WALK_LEVEL: bool = True
USE_STRENGTH: bool = False
USE_FAMILY_SUM: bool = True
USE_BMI_X_AGE: bool = True
USE_OBESITY_FLAG: bool = False
USE_AST_ALT_RATIO: bool = False   # 간기능 특화: AST/ALT 비 (알코올성 vs 지방간 패턴)
USE_LIVER_ENZYME_BIN: bool = False # 간기능 특화: AST/ALT 정상/경계/이상 구간화


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 연도별 필터 적용
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def apply_year_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    hn22~24: 전체
    hn21   : 당뇨 OR 이상지질혈증 OR 간질환 유병
    hn15~20: 당뇨 OR 간질환 유병
    """
    # hn22~24 전체
    df_full = df[df["연도"].isin([2022, 2023, 2024])].copy()

    # hn21
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

    # hn15~20
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
# 컬럼 선택 및 rename
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
    # 혈액수치 제외 — 생활습관/인구통계 기반 스크리닝 목적
    # "HE_glu", "HE_chol", "HE_TG", "HE_HDL_st2", "HE_LDL_drct",
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
    # 혈액수치 제외
    # "HE_glu": "공복혈당", "HE_chol": "총콜레스테롤",
    # "HE_TG": "중성지방", "HE_HDL_st2": "HDL", "HE_LDL_drct": "LDL",
}

INT_COLS = [
    "성별", "나이", "직업", "현재흡연", "음주빈도", "음주량", "걷기일수", "근력운동일수",
    "비만단계", "간질환진단", "B형간염항원", "C형간염항체",
    "고혈압가족력_부", "고혈압가족력_모", "고혈압가족력_형제",
    "당뇨가족력_부", "당뇨가족력_모", "당뇨가족력_형제",
    "고지혈증가족력_부", "고지혈증가족력_모", "고지혈증가족력_형제",
    "고혈압유병", "당뇨유병", "이상지질혈증유병",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 타겟 파생
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def derive_target(df: pd.DataFrame) -> pd.DataFrame:
    """복합 OR 조건으로 간기능이상유병 파생"""
    cond_map = {}
    if "AST"      in df.columns: cond_map["AST>40"]    = df["AST"]      > 40
    if "ALT"      in df.columns: cond_map["ALT>40"]    = df["ALT"]      > 40
    if "ALP"      in df.columns: cond_map["ALP>120"]   = df["ALP"]      > 120
    if "B형간염항원" in df.columns: cond_map["HBsAg양성"] = df["B형간염항원"] == 1
    if "C형간염항체" in df.columns: cond_map["HCV양성"]   = df["C형간염항체"] == 1
    if "간질환진단"  in df.columns: cond_map["간질환진단"]  = df["간질환진단"]  == 1

    if not cond_map:
        raise ValueError("타겟 파생 가능한 변수 없음")

    combined = list(cond_map.values())[0]
    for c in list(cond_map.values())[1:]:
        combined = combined | c
    df[TARGET] = combined.astype(int)

    print("조건별 해당자 수:")
    for name, cond in cond_map.items():
        n = int(cond.sum())
        print(f"  {name:<12}: {n:,}명 ({n/len(df)*100:.1f}%)")

    vc = df[TARGET].value_counts().sort_index()
    n0, n1 = vc.get(0, 0), vc.get(1, 0)
    print(f"\n  [타겟] 정상={n0:,} / 이상={n1:,} | 양성률 {n1/(n0+n1)*100:.1f}% | 불균형 1:{n0/max(n1,1):.1f}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature Engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        df["BMI_구간"] = pd.cut(
            df["BMI"], bins=[0, 23, 25, 30, np.inf], labels=[0, 1, 2, 3], right=False
        ).astype(float)
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
        df["음주위험군"] = pd.cut(
            df["음주빈도"], bins=[-np.inf, 0, 2, np.inf], labels=[0, 1, 2], right=True
        ).astype(float)
        added += ["음주위험군"]
        print("  [ON] 음주위험군")

    if USE_WALK_LEVEL:
        df["걷기활동량"] = pd.cut(
            df["걷기일수"], bins=[-np.inf, 0, 3, np.inf], labels=[0, 1, 2], right=True
        ).astype(float)
        added += ["걷기활동량"]
        print("  [ON] 걷기활동량")

    if USE_STRENGTH:
        df["근력활동량"] = pd.cut(
            df["근력운동일수"], bins=[-np.inf, 0, 2, np.inf], labels=[0, 1, 2], right=True
        ).astype(float)
        added += ["근력활동량"]

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
        print("  [ON] 가족력 합산")

    if USE_BMI_X_AGE:
        df["BMI_X_나이"] = df["BMI"] * df["나이"]
        added += ["BMI_X_나이"]
        print("  [ON] BMI × 나이")

    if USE_OBESITY_FLAG:
        df["비만여부"] = (df["BMI"] >= 25).astype(float)
        df.loc[df["BMI"].isna(), "비만여부"] = np.nan
        added += ["비만여부"]

    # ── 간기능 특화 FE ──────────────────────────────────────────
    if USE_AST_ALT_RATIO:
        # AST/ALT 비: >2 → 알코올성, <1 → 지방간 패턴
        df["AST_ALT_비"] = df["AST"] / (df["ALT"] + 1e-6)
        added += ["AST_ALT_비"]
        print("  [ON] AST/ALT 비 (간질환 패턴)")

    if USE_LIVER_ENZYME_BIN:
        # AST 구간: 정상(≤40) / 경계(41~80) / 이상(>80)
        if "AST" in df.columns:
            df["AST_구간"] = pd.cut(
                df["AST"], bins=[-np.inf, 40, 80, np.inf], labels=[0, 1, 2], right=True
            ).astype(float)
            added += ["AST_구간"]

        # ALT 구간: 정상(≤40) / 경계(41~80) / 이상(>80)
        if "ALT" in df.columns:
            df["ALT_구간"] = pd.cut(
                df["ALT"], bins=[-np.inf, 40, 80, np.inf], labels=[0, 1, 2], right=True
            ).astype(float)
            added += ["ALT_구간"]

        print("  [ON] AST/ALT 구간화")

    print(f"\n[FE] 추가 피처 수: {len(added)} | 목록: {added}")
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

    # 성인 필터
    df_raw = df_raw[df_raw["나이"] >= 19].reset_index(drop=True)

    # 질환 타겟 이진화 (필터 조건용)
    for col in ["고혈압유병", "당뇨유병", "이상지질혈증유병"]:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].map({1.0: 1, 1: 1, 8.0: 0, 8: 0})

    # 간질환진단 9.0 → NaN
    if "간질환진단" in df_raw.columns:
        df_raw.loc[df_raw["간질환진단"] == 9.0, "간질환진단"] = np.nan

    # ── 연도별 필터 ────────────────────────────────────────────
    print("\n[1] 연도별 필터 적용")
    df = apply_year_filter(df_raw)

    # ── 타겟 파생 ──────────────────────────────────────────────
    print("\n[2] 타겟 파생")
    df = derive_target(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # ── FE ─────────────────────────────────────────────────────
    print("\n[3] 피처 엔지니어링")
    df = apply_feature_engineering(df)

    # ── X / y 분리 ─────────────────────────────────────────────
    drop_cols = [
        c for c in [
            "간기능이상유병", "고혈압유병", "당뇨유병", "이상지질혈증유병",
            "비만단계", "연도",
            # 타겟 파생에 직접 쓰인 원본 수치 — 누수 방지
            # (간수치 파생 FE도 OFF — 생활습관/인구통계만으로 예측)
            "AST", "ALT", "ALP", "B형간염항원", "C형간염항체", "간질환진단",
        ] if c in df.columns
    ]
    X: pd.DataFrame = df.drop(columns=drop_cols)
    y: pd.Series = df[TARGET].astype(int)
    print(f"\n[4] Feature 수: {X.shape[1]}")
    print(f"    Feature 목록: {list(X.columns)}")

    # ── Train / Test 분리 ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"\n[5] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"    Train 양성 비율: {y_train.mean():.4f} | Test 양성 비율: {y_test.mean():.4f}")

    # ── 클래스 가중치 ──────────────────────────────────────────
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    ratio = float(neg / pos)

    # ── CatBoost 파라미터 ──────────────────────────────────────
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

    # ── 5-Fold OOF ─────────────────────────────────────────────
    print(f"\n[6] 5-Fold OOF | RECALL_MIN={RECALL_MIN} → F1 최대화")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_proba: NDArray[np.float64] = np.zeros(len(y_train))
    fold_models = []
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        train_pool = Pool(X_tr, y_tr)
        val_pool   = Pool(X_val, y_val)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        val_proba: NDArray[np.float64] = model.predict_proba(X_val)[:, 1]
        val_label = (val_proba >= 0.5).astype(int)
        oof_proba[val_idx] = val_proba
        fold_models.append(model)

        fold_scores.append({
            "fold":      fold,
            "auc":       round(float(roc_auc_score(y_val, val_proba)), 4),
            "recall":    round(float(recall_score(y_val, val_label)), 4),
            "precision": round(float(precision_score(y_val, val_label, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_val, val_label)), 4),
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

    # ── OOF 전체 성능 ──────────────────────────────────────────
    print("=" * 60)
    oof_label_05 = (oof_proba >= 0.5).astype(int)
    oof_auc = float(roc_auc_score(y_train, oof_proba))
    print("\n[7] OOF 전체 성능 (threshold=0.5)")
    print(f"    AUC    : {oof_auc:.4f}  (fold avg: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f})")
    print(
        f"    Recall : {recall_score(y_train, oof_label_05):.4f}  "
        f"(fold avg: {scores_df['recall'].mean():.4f} ± {scores_df['recall'].std():.4f})"
    )
    print(
        f"    F1     : {f1_score(y_train, oof_label_05):.4f}  "
        f"(fold avg: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f})"
    )

    # ── OOF Threshold 탐색 ─────────────────────────────────────
    best_thr, best_f1, oof_recall, oof_prec = tune_threshold(oof_proba, y_train.values)
    print(f"\n[8] OOF Threshold 탐색 (RECALL_MIN={RECALL_MIN})")
    print(f"    best_thr: {best_thr:.2f} | Recall: {oof_recall:.4f} | Precision: {oof_prec:.4f} | F1: {best_f1:.4f}")

    # ── Test 평가 ──────────────────────────────────────────────
    print(f"\n[9] 최종 Hold-out Test 평가 (threshold={best_thr:.2f})")
    print("=" * 60)

    test_probas = np.column_stack([m.predict_proba(X_test)[:, 1] for m in fold_models])
    test_proba_ens: NDArray[np.float64] = test_probas.mean(axis=1)
    test_label = (test_proba_ens >= best_thr).astype(int)

    test_auc    = float(roc_auc_score(y_test, test_proba_ens))
    test_recall = float(recall_score(y_test, test_label))
    test_prec   = float(precision_score(y_test, test_label, zero_division=0))
    test_f1     = float(f1_score(y_test, test_label))
    cm = confusion_matrix(y_test, test_label)

    print(f"    AUC       : {test_auc:.4f}")
    print(f"    Recall    : {test_recall:.4f}")
    print(f"    Precision : {test_prec:.4f}")
    print(f"    F1        : {test_f1:.4f}")
    print(f"    TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"    FN={cm[1, 0]}  TP={cm[1, 1]}")
    print("\n[9] Classification Report")
    print(classification_report(y_test, test_label, target_names=["정상(0)", "간기능이상(1)"]))

    # ── Feature Importance ─────────────────────────────────────
    print("[10] Feature Importance Top 20 (마지막 fold)")
    fi_df = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": fold_models[-1].get_feature_importance(),
    }).sort_values("importance", ascending=False)
    print(fi_df.head(20).to_string(index=False))

    # ── 저장 ───────────────────────────────────────────────────
    scores_df.to_csv(os.path.join(MODEL_DIR, "fold_scores.csv"), index=False)
    fi_df.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    np.save(os.path.join(MODEL_DIR, "oof_y_true.npy"),      y_train.values)
    np.save(os.path.join(MODEL_DIR, "oof_proba.npy"),       oof_proba)
    np.save(os.path.join(MODEL_DIR, "best_threshold.npy"),  np.array([best_thr]))

    print(f"\n[11] 저장 완료 → {MODEL_DIR}")


if __name__ == "__main__":
    main()
