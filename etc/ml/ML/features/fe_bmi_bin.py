"""
fe_bmi_bin.py
─────────────
BMI 구간화 피처 추가

적용 대상: 이상지질혈증 우선 · 고혈압 · 당뇨

근거:
    - BMI가 세 질환 모두 SHAP 2위
    - WHO 비만 기준이 의학적으로 명확하게 정의돼 있음
    - 특히 이상지질혈증에서 BMI·키·체중이 2~4위 독식 → 구간화로 압축 표현
    - 임상적으로 의미 있는 경계값(23, 25, 30)을 명시적으로 반영

추가 컬럼:
    BMI_구간 (float, NaN 허용):
        [한국인 기준]
        0 = 저체중 (BMI < 18.5)
        1 = 정상   (18.5 ≤ BMI < 23)
        2 = 과체중 (23   ≤ BMI < 25)
        3 = 비만1  (25   ≤ BMI < 30)
        4 = 비만2  (BMI ≥ 30)

        [WHO 기준]
        0 = 저체중 (BMI < 18.5)
        1 = 정상   (18.5 ≤ BMI < 25)
        2 = 과체중 (25   ≤ BMI < 30)
        3 = 비만   (BMI ≥ 30)

        BMI NaN → BMI_구간 NaN 유지 (CatBoost/XGBoost 자체 처리)
"""

import pandas as pd


def add_bmi_bin(df: pd.DataFrame, korean_standard: bool = True, drop_original: bool = False) -> pd.DataFrame:
    """
    BMI 구간화 피처를 추가한 DataFrame 반환.

    Parameters
    ----------
    df               : 전처리 완료 데이터프레임
    korean_standard  : True면 한국인 기준(23/25), False면 WHO 기준(25/30)
    drop_original    : True면 원본 BMI 컬럼 제거 (구간 피처만 남김)

    Returns
    -------
    pd.DataFrame
        'BMI_구간' 컬럼이 추가된 데이터프레임 (원본 변경 없음)
    """
    df = df.copy()

    if korean_standard:
        bins = [0, 18.5, 23, 25, 30, 999]
        labels = [0, 1, 2, 3, 4]
        standard_note = "한국인 기준 (18.5/23/25/30)"
    else:
        bins = [0, 18.5, 25, 30, 999]
        labels = [0, 1, 2, 3]
        standard_note = "WHO 기준 (18.5/25/30)"

    # NaN 유지: pd.cut 결과를 float으로 변환 (NaN 허용)
    # CatBoost/XGBoost 모두 NaN 자체 처리 가능
    df["BMI_구간"] = pd.cut(df["BMI"], bins=bins, labels=labels, right=False).astype(float)  # int 대신 float → NaN 유지

    nan_cnt = df["BMI_구간"].isna().sum()
    print(f"[fe_bmi_bin] 'BMI_구간' 추가 완료 ({standard_note})")
    print(f"  분포:\n{df['BMI_구간'].value_counts().sort_index().to_string()}")
    if nan_cnt > 0:
        print(f"  NaN: {nan_cnt}건 (BMI 결측 → 모델 자체 처리)")

    if drop_original:
        df = df.drop(columns=["BMI"])
        print("[fe_bmi_bin] 원본 'BMI' 컬럼 제거")

    return df
