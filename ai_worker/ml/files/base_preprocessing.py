"""
hn24 file1 전처리 파이프라인 (성인 한정)
Python 3.13 | pandas>=2.2 | numpy>=2.0 | scikit-learn>=1.4

[타겟 코드북]
DI1_pr / DE1_pr / DI2_pr:
  1 = 유병 (현재 치료중)
  0 = 유병 (치료 안함)  → NaN 처리
  8 = 해당없음 (정상)   → 0 으로 이진화
  9 = 무응답            → NaN 처리
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight

# ── 경로 설정 (여기만 수정) ───────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, 'data', 'hn24_file1_survey.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'hn24_file1_preprocessed.csv')


# ──────────────────────────────────────────────
# 0. 데이터 로드 & 소아 제거 (19세 미만 제외)
# ──────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)

col_rename = {
    'sex':      '성별',   'age':    '나이',   'occp':      '직업',
    'HE_ht':    '키',     'HE_wt':  '체중',   'HE_BMI':    'BMI',
    'sm_presnt':'현재흡연','BD1_11': '음주빈도','BD2_1':     '음주량',
    'HE_HPfh1': '고혈압가족력_부',  'HE_HPfh2': '고혈압가족력_모',  'HE_HPfh3': '고혈압가족력_형제',
    'HE_DMfh1': '당뇨가족력_부',    'HE_DMfh2': '당뇨가족력_모',    'HE_DMfh3': '당뇨가족력_형제',
    'HE_HLfh1': '고지혈증가족력_부','HE_HLfh2': '고지혈증가족력_모','HE_HLfh3': '고지혈증가족력_형제',
    'BE3_31':   '걷기일수','BE5_1':  '근력운동일수',
    'DI1_pr':   '고혈압유병','DE1_pr': '당뇨유병','DI2_pr': '이상지질혈증유병','HE_obe': '비만단계',
}
df = df.rename(columns=col_rename)
df = df[df['나이'] >= 19].reset_index(drop=True)
df = df.drop(columns=['ID'])

print(f"[0] 로드 완료 | shape: {df.shape}")


# ──────────────────────────────────────────────
# 1. 가족력 결측 → 0
# ──────────────────────────────────────────────
fh_cols = [
    '고혈압가족력_부',   '고혈압가족력_모',   '고혈압가족력_형제',
    '당뇨가족력_부',     '당뇨가족력_모',     '당뇨가족력_형제',
    '고지혈증가족력_부', '고지혈증가족력_모', '고지혈증가족력_형제',
]
df[fh_cols] = df[fh_cols].fillna(0)
print(f"[1] 가족력 결측 → 0 | 잔여 결측: {df[fh_cols].isnull().sum().sum()}")


# ──────────────────────────────────────────────
# 2. 직업 결측 → 8 (작업미상)
# ──────────────────────────────────────────────
df['직업'] = df['직업'].fillna(8)
print(f"[2] 직업 결측 → 8(작업미상) | 잔여 결측: {df['직업'].isnull().sum()}")


# ──────────────────────────────────────────────
# 3. 키·체중 결측 → 중앙값 대체
# ──────────────────────────────────────────────
ht_median = df['키'].median()
wt_median = df['체중'].median()
df['키']   = df['키'].fillna(ht_median)
df['체중'] = df['체중'].fillna(wt_median)
print(f"[3] 키 중앙값={ht_median} / 체중 중앙값={wt_median} 대체 완료")


# ──────────────────────────────────────────────
# 4. BMI 결측 → 재계산
# ──────────────────────────────────────────────
bmi_mask = df['BMI'].isnull()
df.loc[bmi_mask, 'BMI'] = (
    df.loc[bmi_mask, '체중'] / (df.loc[bmi_mask, '키'] / 100) ** 2
).round(6)
print(f"[4] BMI {bmi_mask.sum()}건 재계산 완료 | 잔여 결측: {df['BMI'].isnull().sum()}")


# ──────────────────────────────────────────────
# 5. 음주빈도 특수값 처리
# ──────────────────────────────────────────────
df['과거음주_현재금주'] = (df['음주빈도'] == 6).astype(int)
df['음주빈도'] = df['음주빈도'].replace(6, 0)
print(f"[5] 음주빈도 6번 분리 | 과거음주_현재금주=1: {df['과거음주_현재금주'].sum()}명")


# ──────────────────────────────────────────────
# 6. 음주량 특수값 처리
# ──────────────────────────────────────────────
n_8 = (df['음주량'] == 8).sum()
n_9 = (df['음주량'] == 9).sum()
df['음주량'] = df['음주량'].replace({8: 0, 9: np.nan})
print(f"[6] 음주량 특수값 | 8(모름)→0: {n_8}건 / 9(무응답)→NaN: {n_9}건")


# ──────────────────────────────────────────────
# 7. Ordinal Encoding
# ──────────────────────────────────────────────
check_cols = ['음주빈도', '음주량', '걷기일수', '근력운동일수']
print(f"\n[7-사전확인] 결측 현황: { {c: int(df[c].isnull().sum()) for c in check_cols} }")

for col in ['음주빈도', '걷기일수', '근력운동일수']:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"  → {col} 결측 {n_miss}건 중앙값({med:.1f}) 대체")

oe_freq = OrdinalEncoder(categories=[[0, 1, 2, 3, 4, 5]])
df['음주빈도_enc'] = oe_freq.fit_transform(df[['음주빈도']]).astype(int)

oe_amt = OrdinalEncoder(
    categories=[[0, 1, 2, 3, 4, 5]],
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=np.nan,
)
df['음주량_enc'] = oe_amt.fit_transform(df[['음주량']])

df['걷기일수']     = df['걷기일수'].astype(int)
df['근력운동일수'] = df['근력운동일수'].astype(int)
df = df.drop(columns=['음주빈도', '음주량'])

print(f"[7] Ordinal Encoding 완료")


# ──────────────────────────────────────────────
# 8. 직업 One-Hot Encoding
# ──────────────────────────────────────────────
job_label = {
    1: '관리전문', 2: '사무',    3: '서비스판매', 4: '농림어업',
    5: '기능노무', 6: '주부학생', 7: '무직',       8: '작업미상',
}
df['직업_str'] = df['직업'].map(job_label)
job_dummies   = pd.get_dummies(df['직업_str'], prefix='직업').astype(int)
df = pd.concat([df, job_dummies], axis=1)
df = df.drop(columns=['직업', '직업_str'])
print(f"[8] 직업 OHE 완료 | 생성 컬럼: {[c for c in df.columns if c.startswith('직업_')]}")


# ──────────────────────────────────────────────
# 9. 현재흡연 결측 → 0
# ──────────────────────────────────────────────
n_sm = df['현재흡연'].isnull().sum()
if n_sm > 0:
    df['현재흡연'] = df['현재흡연'].fillna(0)
    print(f"[9] 현재흡연 결측 {n_sm}건 → 0(비흡연) 대체")
else:
    print(f"[9] 현재흡연 결측 없음")


# ──────────────────────────────────────────────
# 10. 타겟 이진화
#     1=유병 / 8(해당없음)→0=정상 / 0·9→NaN(제외)
# ──────────────────────────────────────────────
print("\n[10] 타겟 이진화 전 분포:")
for col in ['고혈압유병', '당뇨유병', '이상지질혈증유병']:
    print(f"  {col}: {df[col].value_counts().sort_index().to_dict()}")

for col in ['고혈압유병', '당뇨유병', '이상지질혈증유병']:
    df[col] = df[col].map({1.0: 1, 1: 1, 8.0: 0, 8: 0})  # 0·9 → NaN 자동

print("\n[10] 타겟 이진화 후 분포 (0=정상 / 1=유병 / NaN=제외):")
for col in ['고혈압유병', '당뇨유병', '이상지질혈증유병']:
    vc = df[col].value_counts(dropna=False).sort_index()
    print(f"  {col}: {vc.to_dict()}")


# ──────────────────────────────────────────────
# 11. 잔여 결측 확인
# ──────────────────────────────────────────────
remaining = df.isnull().sum()
remaining = remaining[remaining > 0]
print(f"\n[11] 잔여 결측치:")
if remaining.empty:
    print("  → 없음 ✓")
else:
    print(remaining)

print(f"\n[12] 최종 shape: {df.shape}")
print(f"     컬럼 목록: {list(df.columns)}")


# ──────────────────────────────────────────────
# 13. class_weight 계산
# ──────────────────────────────────────────────
y_cols = ['고혈압유병', '당뇨유병', '이상지질혈증유병', '비만단계']
class_weights: dict = {}

print("\n[13] class_weight (balanced):")
for y in y_cols:
    y_clean = df[y].dropna()
    classes = np.sort(y_clean.unique())
    cw      = compute_class_weight(class_weight='balanced', classes=classes, y=y_clean)
    cw_dict = dict(zip(classes.astype(int), cw.round(4)))
    class_weights[y] = cw_dict
    print(f"  {y}: {cw_dict}")


# ──────────────────────────────────────────────
# 14. 저장
# ──────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[14] 저장 완료 → {OUTPUT_PATH}")
