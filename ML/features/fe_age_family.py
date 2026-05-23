"""
fe_age_family.py
────────────────
나이 × 가족력 상호작용 피처 추가

적용 대상: 고혈압 · 당뇨

근거:
    - 나이가 많고 가족력도 있으면 위험이 복합 증폭되는 의학적 사실
    - 두 변수가 독립적으로 들어가면 트리가 교차 효과를 찾기 어려움
    - 상호작용 피처로 복합 위험을 단 한 번의 분기로 포착 가능
    - 고혈압(나이 의존도 최고) + 당뇨(가족력 중요도 최고)에서 효과 기대

추가 컬럼:
    나이_고혈압가족력 (int): 나이_구간 × 고혈압가족력_합계
    나이_당뇨가족력   (int): 나이_구간 × 당뇨가족력_합계
    → add_age_bin() + add_family_sum() 먼저 호출 필요
"""

import pandas as pd


def add_age_family_interaction(df: pd.DataFrame, hypertension: bool = True, diabetes: bool = True) -> pd.DataFrame:
    """
    나이 × 가족력 상호작용 피처를 추가한 DataFrame 반환.

    Parameters
    ----------
    df           : 전처리 완료 데이터프레임
                   '나이_구간', '고혈압가족력_합계', '당뇨가족력_합계' 필요
                   → add_age_bin() + add_family_sum() 선행 호출 필요
    hypertension : 나이 × 고혈압가족력 추가 여부
    diabetes     : 나이 × 당뇨가족력 추가 여부

    Returns
    -------
    pd.DataFrame
        상호작용 피처가 추가된 데이터프레임 (원본 변경 없음)
    """
    df = df.copy()

    required = ["나이_구간"]
    if hypertension:
        required.append("고혈압가족력_합계")
    if diabetes:
        required.append("당뇨가족력_합계")

    for col in required:
        if col not in df.columns:
            print(f"[fe_age_family] 경고: '{col}' 없음 → 선행 FE 함수를 먼저 호출하세요.")
            return df

    if hypertension:
        df["나이_고혈압가족력"] = (df["나이_구간"] * df["고혈압가족력_합계"]).astype(int)
        print(
            f"[fe_age_family] '나이_고혈압가족력' 추가 | "
            f"분포: {df['나이_고혈압가족력'].value_counts().sort_index().to_dict()}"
        )

    if diabetes:
        df["나이_당뇨가족력"] = (df["나이_구간"] * df["당뇨가족력_합계"]).astype(int)
        print(
            f"[fe_age_family] '나이_당뇨가족력' 추가   | "
            f"분포: {df['나이_당뇨가족력'].value_counts().sort_index().to_dict()}"
        )

    return df
