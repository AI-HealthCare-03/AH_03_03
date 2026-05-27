"""
fe_alcohol.py
─────────────
음주 복합 피처 추가

적용 대상: 고혈압 · 당뇨 (이상지질혈증 선택적)

근거:
    - 고혈압 SHAP: 음주량(9위) > 음주빈도(11위)
    - 당뇨   SHAP: 음주빈도(7위) > 음주량(11위)
    - 두 변수의 독립적 기여 외에 상호작용(빈도×양) 정보가 추가로 존재
    - 음주_총부하 = 빈도 × 양 → '주당 총 음주량' proxy

추가 컬럼:
    음주_총부하 (float): 음주빈도_enc × 음주량_enc
                        음주량_enc가 NaN이면 NaN 유지
                        (CatBoost·XGBoost 모두 NaN 자체 처리 가능)
"""

import numpy as np
import pandas as pd


def add_alcohol_load(df: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """
    음주 복합 피처(총부하)를 추가한 DataFrame 반환.

    Parameters
    ----------
    df             : 전처리 완료 데이터프레임
                     '음주빈도_enc', '음주량_enc' 컬럼 필요
    drop_original  : True면 음주빈도_enc · 음주량_enc 원본 컬럼 제거

    Returns
    -------
    pd.DataFrame
        '음주_총부하' 컬럼이 추가된 데이터프레임 (원본 변경 없음)
    """
    df = df.copy()

    # 음주량_enc -1 → NaN 처리 (OrdinalEncoder encoded_missing_value=-1 방어 처리)
    음주량 = df["음주량_enc"].replace(-1, np.nan)

    # 음주량_enc NaN 유지 (비음주자 0 처리하지 않음 — 정보 왜곡 방지)
    df["음주_총부하"] = df["음주빈도_enc"] * 음주량

    non_null = df["음주_총부하"].notna().sum()
    null_cnt = df["음주_총부하"].isna().sum()
    print("[fe_alcohol] '음주_총부하' 추가 완료")
    print(f"  유효값: {non_null}건 | NaN(비음주/결측): {null_cnt}건")
    print(
        f"  분포: min={df['음주_총부하'].min():.1f} "
        f"mean={df['음주_총부하'].mean():.2f} "
        f"max={df['음주_총부하'].max():.1f}"
    )

    if drop_original:
        df = df.drop(columns=["음주빈도_enc", "음주량_enc"])
        print("[fe_alcohol] 원본 제거: ['음주빈도_enc', '음주량_enc']")

    return df
