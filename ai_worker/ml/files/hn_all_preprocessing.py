"""
hn22 + hn23 + hn24 통합 전처리 파이프라인 (성인 한정)
hn22 / hn23 / hn24 : SAS7BDAT

Python 3.13 | pandas>=2.2 | numpy>=2.0 | scikit-learn>=1.4 | pyreadstat>=1.2

[타겟 코드북]
DI1_pr / DE1_pr / DI2_pr:
  1 = 유병 (현재 치료중)
  0 = 유병 (치료 안함)  → NaN 처리
  8 = 해당없음 (정상)   → 0 으로 이진화
  9 = 무응답            → NaN 처리
"""

import os

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight

# ── 경로 설정 (여기만 수정) ───────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_HN22    = os.path.join(BASE_DIR, 'data', 'hn22_all.sas7bdat')
PATH_HN23    = os.path.join(BASE_DIR, 'data', 'hn23_all.sas7bdat')
PATH_HN24    = os.path.join(BASE_DIR, 'data', 'hn24_all.sas7bdat')
OUTPUT_PATH  = os.path.join(BASE_DIR, 'data', 'hn_all_preprocessed.csv')


# ══════════════════════════════════════════════
# 공통 설정
# ══════════════════════════════════════════════
USE_COLS_SAS = [
    'ID','sex','age','occp','HE_ht','HE_wt','HE_BMI','sm_presnt',
    'BD1_11','BD2_1',
    'HE_HPfh1','HE_HPfh2','HE_HPfh3',
    'HE_DMfh1','HE_DMfh2','HE_DMfh3',
    'HE_HLfh1','HE_HLfh2','HE_HLfh3',
    'BE3_31','BE5_1',
    'DI1_pr','DE1_pr','DI2_pr','HE_obe',
]

COL_RENAME = {
    'sex':'성별','age':'나이','occp':'직업','HE_ht':'키','HE_wt':'체중','HE_BMI':'BMI',
    'sm_presnt':'현재흡연','BD1_11':'음주빈도','BD2_1':'음주량',
    'HE_HPfh1':'고혈압가족력_부','HE_HPfh2':'고혈압가족력_모','HE_HPfh3':'고혈압가족력_형제',
    'HE_DMfh1':'당뇨가족력_부','HE_DMfh2':'당뇨가족력_모','HE_DMfh3':'당뇨가족력_형제',
    'HE_HLfh1':'고지혈증가족력_부','HE_HLfh2':'고지혈증가족력_모','HE_HLfh3':'고지혈증가족력_형제',
    'BE3_31':'걷기일수','BE5_1':'근력운동일수',
    'DI1_pr':'고혈압유병','DE1_pr':'당뇨유병','DI2_pr':'이상지질혈증유병','HE_obe':'비만단계',
}

INT_COLS = [
    '성별','나이','직업','현재흡연','음주빈도','음주량','걷기일수','근력운동일수',
    '고혈압유병','당뇨유병','이상지질혈증유병','비만단계',
    '고혈압가족력_부','고혈압가족력_모','고혈압가족력_형제',
    '당뇨가족력_부','당뇨가족력_모','당뇨가족력_형제',
    '고지혈증가족력_부','고지혈증가족력_모','고지혈증가족력_형제',
]

FH_COLS = [
    '고혈압가족력_부','고혈압가족력_모','고혈압가족력_형제',
    '당뇨가족력_부','당뇨가족력_모','당뇨가족력_형제',
    '고지혈증가족력_부','고지혈증가족력_모','고지혈증가족력_형제',
]

JOB_LABEL = {
    1:'관리전문',2:'사무',3:'서비스판매',4:'농림어업',
    5:'기능노무',6:'주부학생',7:'무직',8:'작업미상',
}

Y_COLS = ['고혈압유병','당뇨유병','이상지질혈증유병','비만단계']


# ══════════════════════════════════════════════
# 전처리 함수
# ══════════════════════════════════════════════
def preprocess(df: pd.DataFrame, year: int) -> pd.DataFrame:

    # 0. 소아 제거 & ID 삭제
    df = df[df['나이'] >= 19].reset_index(drop=True)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # 1. 가족력 결측 → 0
    df[FH_COLS] = df[FH_COLS].fillna(0)

    # 2. 직업 결측 → 8
    df['직업'] = df['직업'].fillna(8)

    # 3. 키·체중 중앙값 대체
    df['키']   = df['키'].fillna(df['키'].median())
    df['체중'] = df['체중'].fillna(df['체중'].median())

    # 4. BMI 재계산
    bmi_mask = df['BMI'].isnull()
    df.loc[bmi_mask, 'BMI'] = (
        df.loc[bmi_mask,'체중'] / (df.loc[bmi_mask,'키']/100)**2
    ).round(6)

    # 5. 음주빈도 특수값 처리
    #    hn22·hn23: 8·9 추가 / hn24: 없음 → replace로 통일 처리
    df['과거음주_현재금주'] = (df['음주빈도'] == 6).astype(int)
    df['음주빈도'] = df['음주빈도'].replace({6:0, 8:0, 9:np.nan})
    if df['음주빈도'].isnull().sum() > 0:
        df['음주빈도'] = df['음주빈도'].fillna(df['음주빈도'].median())

    # 6. 음주량 특수값 처리
    df['음주량'] = df['음주량'].replace({8:0, 9:np.nan})

    # 7. 걷기·근력운동 결측 → 중앙값
    for col in ['걷기일수','근력운동일수']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 8. OrdinalEncoding
    oe_freq = OrdinalEncoder(categories=[[0,1,2,3,4,5]])
    df['음주빈도_enc'] = oe_freq.fit_transform(df[['음주빈도']]).astype(int)
    oe_amt = OrdinalEncoder(categories=[[0,1,2,3,4,5]], handle_unknown='use_encoded_value',
                            unknown_value=-1, encoded_missing_value=np.nan)
    df['음주량_enc'] = oe_amt.fit_transform(df[['음주량']])
    df['걷기일수']     = df['걷기일수'].astype(int)
    df['근력운동일수'] = df['근력운동일수'].astype(int)
    df = df.drop(columns=['음주빈도','음주량'])

    # 9. 직업 OHE
    df['직업_str'] = df['직업'].map(JOB_LABEL)
    job_dummies = pd.get_dummies(df['직업_str'], prefix='직업').astype(int)
    df = pd.concat([df, job_dummies], axis=1)
    df = df.drop(columns=['직업','직업_str'])

    # 10. 현재흡연 결측 → 0
    if df['현재흡연'].isnull().sum() > 0:
        df['현재흡연'] = df['현재흡연'].fillna(0)

    # 11. 타겟 이진화
    #     1=유병 / 8(해당없음)→0=정상 / 0·9→NaN(제외)
    for col in ['고혈압유병','당뇨유병','이상지질혈증유병']:
        df[col] = df[col].map({1.0:1, 1:1, 8.0:0, 8:0})

    # 12. 연도 컬럼 추가
    df['연도'] = year

    return df


# ══════════════════════════════════════════════
# 데이터 로드 & 전처리
# ══════════════════════════════════════════════
print("=" * 50)
dfs = []
for year, path in [(2022, PATH_HN22), (2023, PATH_HN23), (2024, PATH_HN24)]:
    print(f"[hn{year}] 로드 중...")
    df_tmp, _ = pyreadstat.read_sas7bdat(path, usecols=USE_COLS_SAS)
    df_tmp = df_tmp.rename(columns=COL_RENAME)
    for c in INT_COLS:
        if c in df_tmp.columns:
            df_tmp[c] = pd.to_numeric(df_tmp[c], errors='coerce')
    df_tmp = preprocess(df_tmp, year=year)
    print(f"[hn{year}] 완료 | shape: {df_tmp.shape}")
    dfs.append(df_tmp)


# ══════════════════════════════════════════════
# 통합 & 직업 컬럼 정렬
# ══════════════════════════════════════════════
JOB_COLS_ALL = [f'직업_{v}' for v in JOB_LABEL.values()]
for df_ in dfs:
    for col in JOB_COLS_ALL:
        if col not in df_.columns:
            df_[col] = 0

df_all = pd.concat(dfs, ignore_index=True)

print("=" * 50)
print(f"\n[통합] shape: {df_all.shape}")
print(f"  연도별 샘플: { df_all['연도'].value_counts().sort_index().to_dict() }")

# 타겟 이진화 결과 확인
print("\n[타겟 분포 (0=정상 / 1=유병 / NaN=제외)]")
for col in ['고혈압유병','당뇨유병','이상지질혈증유병']:
    vc = df_all[col].value_counts(dropna=False).sort_index()
    print(f"  {col}: {vc.to_dict()}")


# ══════════════════════════════════════════════
# 잔여 결측 확인
# ══════════════════════════════════════════════
remaining = df_all.isnull().sum()
remaining = remaining[remaining > 0]
print("\n[잔여 결측치]")
print("  → 없음 ✓" if remaining.empty else remaining)
print(f"\n[최종 컬럼 목록]\n  {list(df_all.columns)}")


# ══════════════════════════════════════════════
# class_weight (통합 기준)
# ══════════════════════════════════════════════
class_weights: dict = {}
print("\n[class_weight (balanced) — 통합 기준]")
for y in Y_COLS:
    y_clean = df_all[y].dropna()
    classes = np.sort(y_clean.unique())
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_clean)
    cw_dict = dict(zip(classes.astype(int), cw.round(4)))
    class_weights[y] = cw_dict
    print(f"  {y}: {cw_dict}")


# ══════════════════════════════════════════════
# 저장
# ══════════════════════════════════════════════
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_all.to_csv(OUTPUT_PATH, index=False)
print(f"\n[저장 완료] → {OUTPUT_PATH}")
print(f"  최종 shape: {df_all.shape}")
