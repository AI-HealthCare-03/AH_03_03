"""
공통 피처 엔지니어링 모듈
타겟별 FE 플래그는 experiments/configs/*.json 에서 관리
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FE 블록 함수 (단위 함수 — 플래그로 개별 호출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def add_age_bin(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """나이 6구간 원핫 인코딩 (19~39 / 40대 / 50대 / 60대 / 70대 / 80이상)"""
    age_bins = [0, 40, 50, 60, 70, 80, np.inf]
    age_labels = ["나이_19_39", "나이_40대", "나이_50대", "나이_60대", "나이_70대", "나이_80이상"]
    df["_나이구간"] = pd.cut(df["나이"], bins=age_bins, labels=age_labels, right=False)
    for label in age_labels:
        df[label] = (df["_나이구간"] == label).astype(int)
    df = df.drop(columns=["_나이구간"])
    return df, age_labels


def add_bmi_bin(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """BMI 4구간화 (저체중/정상/과체중/비만)"""
    df["BMI_구간"] = pd.cut(df["BMI"], bins=[0, 23, 25, 30, np.inf], labels=[0, 1, 2, 3], right=False).astype(float)
    return df, ["BMI_구간"]


def add_family_sum(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """고혈압/당뇨/고지혈증 가족력 합산 (부+모+형제, 0~3 clip)"""
    df["고혈압가족력_합산"] = (
        df["고혈압가족력_부"].fillna(0) + df["고혈압가족력_모"].fillna(0) + df["고혈압가족력_형제"].fillna(0)
    ).clip(0, 3)
    df["당뇨가족력_합산"] = (
        df["당뇨가족력_부"].fillna(0) + df["당뇨가족력_모"].fillna(0) + df["당뇨가족력_형제"].fillna(0)
    ).clip(0, 3)
    df["고지혈증가족력_합산"] = (
        df["고지혈증가족력_부"].fillna(0) + df["고지혈증가족력_모"].fillna(0) + df["고지혈증가족력_형제"].fillna(0)
    ).clip(0, 3)
    return df, ["고혈압가족력_합산", "당뇨가족력_합산", "고지혈증가족력_합산"]


def add_bmi_x_age(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """BMI × 나이 상호작용 (전 타겟 importance 1~2위)"""
    df["BMI_X_나이"] = df["BMI"] * df["나이"]
    return df, ["BMI_X_나이"]


def add_obesity_flag(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """비만 여부 플래그 (BMI >= 25)"""
    df["비만여부"] = (df["BMI"] >= 25).astype(float)
    df.loc[df["BMI"].isna(), "비만여부"] = np.nan
    return df, ["비만여부"]


def add_alcohol_risk(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """음주 위험군 구간화 (0=비음주 / 1=저위험 / 2=고위험)"""
    df["음주위험군"] = pd.cut(df["음주빈도"], bins=[-1, 0, 2, 99], labels=[0, 1, 2], right=True).astype(float)
    return df, ["음주위험군"]


def add_walk_level(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """걷기 활동량 구간화"""
    df["걷기활동량"] = pd.cut(df["걷기일수"], bins=[-1, 0, 3, 99], labels=[0, 1, 2], right=True).astype(float)
    return df, ["걷기활동량"]


def add_strength_level(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """근력 운동 활동량 구간화"""
    df["근력활동량"] = pd.cut(df["근력운동일수"], bins=[-1, 0, 2, 99], labels=[0, 1, 2], right=True).astype(float)
    return df, ["근력활동량"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 타겟별 확정 FE 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# HTN: 나이구간 + 음주위험군 + 걷기활동량 + 가족력합산 + BMI_X_나이 (39개)
# DM:  나이구간 + BMI구간 + 가족력합산 + BMI_X_나이 (38개)
# DL:  나이구간 + BMI구간 + 가족력합산 + BMI_X_나이 (38개)

DISEASE_FE_MAP: dict[str, list[str]] = {
    "HTN": ["age_bin", "alcohol_risk", "walk_level", "family_sum", "bmi_x_age"],
    "DM": ["age_bin", "bmi_bin", "family_sum", "bmi_x_age"],
    "DL": ["age_bin", "bmi_bin", "family_sum", "bmi_x_age"],
}

_FE_REGISTRY = {
    "age_bin": add_age_bin,
    "bmi_bin": add_bmi_bin,
    "family_sum": add_family_sum,
    "bmi_x_age": add_bmi_x_age,
    "obesity_flag": add_obesity_flag,
    "alcohol_risk": add_alcohol_risk,
    "walk_level": add_walk_level,
    "strength": add_strength_level,
}


def apply_feature_engineering(
    df: pd.DataFrame,
    disease: str,
    extra_fe: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    disease 기준으로 확정 FE 적용.
    extra_fe: ablation 실험용 추가 FE 키 리스트 (예: ["bmi_x_age"])
    verbose: FE 로그 출력 여부

    Parameters
    ----------
    df      : 원본 DataFrame
    disease : "HTN" | "DM" | "DL"
    extra_fe: 추가 실험용 FE 키 (DISEASE_FE_MAP 외 추가)
    verbose : 로그 출력 여부

    Returns
    -------
    FE 적용된 DataFrame
    """
    if disease not in DISEASE_FE_MAP:
        raise ValueError(f"disease는 {list(DISEASE_FE_MAP.keys())} 중 하나여야 합니다. 입력값: {disease}")

    fe_keys = DISEASE_FE_MAP[disease][:]
    if extra_fe:
        fe_keys += [k for k in extra_fe if k not in fe_keys]

    if verbose:
        print("[FE] 피처 엔지니어링 시작")

    added: list[str] = []
    for key in fe_keys:
        if key not in _FE_REGISTRY:
            raise ValueError(f"알 수 없는 FE 키: {key}. 등록된 키: {list(_FE_REGISTRY.keys())}")
        df, cols = _FE_REGISTRY[key](df)
        added += cols
        if verbose:
            print(f"  [ON] {key} → {cols}")

    if verbose:
        print(f"\n[FE] 추가 피처 수: {len(added)} | 목록: {added}")

    return df
