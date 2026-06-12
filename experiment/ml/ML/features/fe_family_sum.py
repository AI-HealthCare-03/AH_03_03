"""
fe_family_sum.py
────────────────
가족력 합산 피처 추가

적용 대상: 고혈압 · 당뇨 · 이상지질혈증 (세 질환 공통)

근거:
    - 고혈압/당뇨/고지혈증 가족력 3종(부·모·형제)이 각 질환 SHAP 상위권
    - '몇 명이 해당 질환인가'라는 정보가 더 직접적
    - 3개 독립 변수 대신 합산 1개로 트리 분기 단순화
    - 개별 변수의 노이즈 감소 효과 기대

추가 컬럼:
    고혈압가족력_합계  (int): 0~3 (부+모+형제)
    당뇨가족력_합계    (int): 0~3
    고지혈증가족력_합계 (int): 0~3
"""

import pandas as pd


def add_family_sum(
    df: pd.DataFrame,
    hypertension: bool = True,
    diabetes: bool = True,
    dyslipidemia: bool = True,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    가족력 합산 피처를 추가한 DataFrame 반환.

    Parameters
    ----------
    df             : 전처리 완료 데이터프레임
    hypertension   : 고혈압 가족력 합산 추가 여부
    diabetes       : 당뇨 가족력 합산 추가 여부
    dyslipidemia   : 고지혈증 가족력 합산 추가 여부
    drop_original  : True이면 합산에 사용한 원본 3종 컬럼 제거
                     (합산 피처만 남기는 실험용)

    Returns
    -------
    pd.DataFrame
        가족력 합산 컬럼이 추가된 데이터프레임 (원본 변경 없음)
    """
    df = df.copy()

    if hypertension:
        cols = ["고혈압가족력_부", "고혈압가족력_모", "고혈압가족력_형제"]
        df["고혈압가족력_합계"] = df[cols].sum(axis=1).astype(int)
        print(
            f"[fe_family_sum] '고혈압가족력_합계' 추가 | 분포: {df['고혈압가족력_합계'].value_counts().sort_index().to_dict()}"
        )
        if drop_original:
            df = df.drop(columns=cols)
            print(f"[fe_family_sum] 원본 제거: {cols}")

    if diabetes:
        cols = ["당뇨가족력_부", "당뇨가족력_모", "당뇨가족력_형제"]
        df["당뇨가족력_합계"] = df[cols].sum(axis=1).astype(int)
        print(
            f"[fe_family_sum] '당뇨가족력_합계' 추가   | 분포: {df['당뇨가족력_합계'].value_counts().sort_index().to_dict()}"
        )
        if drop_original:
            df = df.drop(columns=cols)
            print(f"[fe_family_sum] 원본 제거: {cols}")

    if dyslipidemia:
        cols = ["고지혈증가족력_부", "고지혈증가족력_모", "고지혈증가족력_형제"]
        df["고지혈증가족력_합계"] = df[cols].sum(axis=1).astype(int)
        print(
            f"[fe_family_sum] '고지혈증가족력_합계' 추가 | 분포: {df['고지혈증가족력_합계'].value_counts().sort_index().to_dict()}"
        )
        if drop_original:
            df = df.drop(columns=cols)
            print(f"[fe_family_sum] 원본 제거: {cols}")

    return df
