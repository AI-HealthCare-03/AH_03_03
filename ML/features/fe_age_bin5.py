"""
fe_age_bin5.py
──────────────
나이 구간화 피처 추가 (5구간)

적용 대상: 고혈압 · 당뇨 · 이상지질혈증 (세 질환 공통)

근거:
    - 4구간(fe_age_bin.py)에서 60세 이상(3구간)에 샘플이 집중(9,260명)
    - 60~69세 vs 70세 이상은 만성질환 위험 수준이 다름
      (70세 이상에서 위험 추가 급증)
    - 5구간으로 분리 시 60~69(4,535명) / 70세 이상(4,725명)으로 균등 분포

추가 컬럼:
    나이_구간5 (int):
        0 = 19~39세
        1 = 40~49세
        2 = 50~59세
        3 = 60~69세
        4 = 70세 이상
"""

import pandas as pd


def add_age_bin5(df: pd.DataFrame, drop_original: bool = False) -> pd.DataFrame:
    """
    나이 5구간화 피처를 추가한 DataFrame 반환.

    Parameters
    ----------
    df             : '나이' 컬럼이 포함된 전처리 완료 데이터프레임
    drop_original  : True면 원본 '나이' 컬럼 제거

    Returns
    -------
    pd.DataFrame
        '나이_구간5' 컬럼이 추가된 데이터프레임 (원본 변경 없음)
    """
    df = df.copy()

    bins = [18, 39, 49, 59, 69, 999]
    labels = [0, 1, 2, 3, 4]

    df["나이_구간5"] = pd.cut(df["나이"], bins=bins, labels=labels, right=True).astype(int)

    print("[fe_age_bin5] '나이_구간5' 추가 완료")
    print(f"  분포:\n{df['나이_구간5'].value_counts().sort_index().to_string()}")
    print("  (0: 19~39세 / 1: 40~49세 / 2: 50~59세 / 3: 60~69세 / 4: 70세 이상)")

    if drop_original:
        df = df.drop(columns=["나이"])
        print("[fe_age_bin5] 원본 '나이' 컬럼 제거")

    return df
