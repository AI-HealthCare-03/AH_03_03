"""
fe_body.py
──────────
체형 복합 피처 추가

적용 대상: 이상지질혈증 특화 · 당뇨

근거:
    - 이상지질혈증 SHAP에서 BMI(2위)·키(3위)·체중(7위) 모두 상위권
    - 당뇨 SHAP에서도 BMI(3위)·체중(6위)·키(11위) 모두 Top 15
    - BMI 단독으로 표현되지 않는 체형 정보가 존재함을 시사
    - 체중/키 비율, BMI×나이 상호작용으로 이를 명시적 표현

추가 컬럼:
    체중_키_비율      (float): 체중 / 키 (키 대비 체중)
    BMI_나이_상호작용 (float): BMI × 나이_구간 또는 나이_구간5
                              → add_age_bin() 또는 add_age_bin5() 먼저 호출 필요
"""

import pandas as pd


def add_body_features(
    df: pd.DataFrame, weight_height_ratio: bool = True, bmi_age_interaction: bool = True, drop_original: bool = False
) -> pd.DataFrame:
    """
    체형 복합 피처를 추가한 DataFrame 반환.

    Parameters
    ----------
    df                   : 전처리 완료 데이터프레임
                           '체중', '키', 'BMI' 컬럼 필요
                           bmi_age_interaction=True 시 '나이_구간' 또는
                           '나이_구간5' 컬럼도 필요 (add_age_bin/add_age_bin5 선행 호출)
    weight_height_ratio  : True면 체중_키_비율 추가
    bmi_age_interaction  : True면 BMI_나이_상호작용 추가
    drop_original        : True면 체중·키·BMI 원본 컬럼 제거

    Returns
    -------
    pd.DataFrame
        체형 복합 피처가 추가된 데이터프레임 (원본 변경 없음)
    """
    df = df.copy()

    if weight_height_ratio:
        df["체중_키_비율"] = (df["체중"] / df["키"]).round(4)
        print(f"[fe_body] '체중_키_비율' 추가 완료 | mean={df['체중_키_비율'].mean():.4f}")

    if bmi_age_interaction:
        # 나이_구간5 우선, 없으면 나이_구간 사용
        if "나이_구간5" in df.columns:
            age_col = "나이_구간5"
        elif "나이_구간" in df.columns:
            age_col = "나이_구간"
        else:
            print(
                "[fe_body] 경고: '나이_구간' / '나이_구간5' 없음 → add_age_bin() 또는 add_age_bin5() 먼저 호출하세요."
            )
            age_col = None

        if age_col:
            df["BMI_나이_상호작용"] = (df["BMI"] * df[age_col]).round(4)
            print(
                f"[fe_body] 'BMI_나이_상호작용' 추가 완료 ({age_col} 기준) | mean={df['BMI_나이_상호작용'].mean():.4f}"
            )

    if drop_original:
        drop_cols = [c for c in ["체중", "키", "BMI"] if c in df.columns]
        df = df.drop(columns=drop_cols)
        print(f"[fe_body] 원본 제거: {drop_cols}")

    return df
