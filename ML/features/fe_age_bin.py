"""
fe_age_bin.py
─────────────
나이 구간화 피처 추가

적용 대상: 고혈압 · 당뇨 · 이상지질혈증 (세 질환 공통)

근거:
    - 나이가 세 질환 모두 SHAP 1위
    - 만성질환 위험은 특정 연령대에서 계단식으로 급증하는 비선형 패턴
    - 연속형 그대로 쓰면 트리가 분기점을 반복 탐색해야 하지만
      구간화로 명시하면 학습 효율 향상 및 다른 변수 기여도 포착에 유리

추가 컬럼:
    나이_구간 (int): 0=19~39세 / 1=40~49세 / 2=50~59세 / 3=60세 이상
"""

import pandas as pd


def add_age_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    나이 구간화 피처를 추가한 DataFrame 반환.

    Parameters
    ----------
    df : pd.DataFrame
        '나이' 컬럼이 포함된 전처리 완료 데이터프레임

    Returns
    -------
    pd.DataFrame
        '나이_구간' 컬럼이 추가된 데이터프레임 (원본 변경 없음)
    """
    df = df.copy()

    bins = [18, 39, 49, 59, 999]
    labels = [0, 1, 2, 3]

    df["나이_구간"] = pd.cut(df["나이"], bins=bins, labels=labels, right=True).astype(int)

    print("[fe_age_bin] '나이_구간' 추가 완료")
    print(f"  분포:\n{df['나이_구간'].value_counts().sort_index().to_string()}")
    print("  (0: 19~39세 / 1: 40~49세 / 2: 50~59세 / 3: 60세 이상)")

    return df
