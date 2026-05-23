"""
fe_exercise.py
──────────────
신체활동 합산 피처 추가

적용 대상: 당뇨 우선 · 이상지질혈증

근거:
    - 당뇨 SHAP: 근력운동일수 10위 (세 질환 CatBoost 중 가장 높음)
    - 이상지질혈증 SHAP: 걷기일수 11위
    - 두 변수 모두 '신체활동 수준'을 표현 → 합산으로 총 신체활동 표현
    - 인슐린 저항성 개선에 유산소(걷기) + 근력운동이 모두 기여

추가 컬럼:
    총운동일수 (int): 걷기일수 + 근력운동일수 (0~14)
"""

import pandas as pd


def add_exercise_total(df: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """
    신체활동 합산 피처를 추가한 DataFrame 반환.

    Parameters
    ----------
    df             : 전처리 완료 데이터프레임
                     '걷기일수', '근력운동일수' 컬럼 필요
    drop_original  : True면 걷기일수 · 근력운동일수 원본 컬럼 제거

    Returns
    -------
    pd.DataFrame
        '총운동일수' 컬럼이 추가된 데이터프레임 (원본 변경 없음)
    """
    df = df.copy()

    df["총운동일수"] = df["걷기일수"] + df["근력운동일수"]

    print("[fe_exercise] '총운동일수' 추가 완료")
    print(f"  범위: {df['총운동일수'].min()}~{df['총운동일수'].max()}일")
    print(f"  평균: {df['총운동일수'].mean():.2f}일")

    if drop_original:
        df = df.drop(columns=["걷기일수", "근력운동일수"])
        print("[fe_exercise] 원본 제거: ['걷기일수', '근력운동일수']")

    return df
