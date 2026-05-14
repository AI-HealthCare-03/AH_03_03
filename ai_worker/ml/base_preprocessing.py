"""
hn24 file1 전처리 파이프라인 (성인 한정)
Python 3.13 | pandas>=2.2 | numpy>=2.0 | scikit-learn>=1.4
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight

# ── 경로 설정 (여기만 수정) ───────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # models/ 한 단계 위 → chronic-health/
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
#    정보 없음 = 가족력 없음으로 처리
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
#    코드북: 1~7 (7=무직) / 8=작업미상 신규 추가
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
# 4. BMI 결측 → 키/체중 공식 재계산
#    BMI = 체중(kg) / (키(m))^2  — 키·체중 채운 후 실행
# ──────────────────────────────────────────────
bmi_mask = df['BMI'].isnull()
df.loc[bmi_mask, 'BMI'] = (
    df.loc[bmi_mask, '체중'] / (df.loc[bmi_mask, '키'] / 100) ** 2
).round(6)
print(f"[4] BMI {bmi_mask.sum()}건 재계산 완료 | 잔여 결측: {df['BMI'].isnull().sum()}")


# ──────────────────────────────────────────────
# 5. 음주빈도 6번 분리 → 플래그 컬럼 후 0으로 대체
#    6 = "최근 1년 안 마심" = 과거음주 + 현재금주
#    건강 악화로 끊은 케이스 포함 가능성 → 정보 보존
# ──────────────────────────────────────────────
df['과거음주_현재금주'] = (df['음주빈도'] == 6).astype(int)
df['음주빈도'] = df['음주빈도'].replace(6, 0)
print(f"[5] 음주빈도 6번 분리 | 과거음주_현재금주=1: {df['과거음주_현재금주'].sum()}명")


# ──────────────────────────────────────────────
# 6. 음주량 특수값 처리
#    코드북: 0=비음주 / 1=1-2잔 / 2=3-4잔 / 3=5-6잔 / 4=7-9잔 / 5=10잔이상
#            8=모름   → 0 (비음주로 보수적 처리)
#            9=무응답 → NaN (트리 계열 모델이므로 그대로 유지)
# ──────────────────────────────────────────────
n_8 = (df['음주량'] == 8).sum()
n_9 = (df['음주량'] == 9).sum()
df['음주량'] = df['음주량'].replace({8: 0, 9: np.nan})
print(f"[6] 음주량 특수값 | 8(모름)→0: {n_8}건 / 9(무응답)→NaN: {n_9}건")


# ──────────────────────────────────────────────
# 7. Ordinal Encoding
#    음주빈도(0~5), 음주량(0~5, NaN 유지), 걷기일수, 근력운동일수
# ──────────────────────────────────────────────
check_cols = ['음주빈도', '음주량', '걷기일수', '근력운동일수']
print(f"\n[7-사전확인] 결측 현황: { {c: int(df[c].isnull().sum()) for c in check_cols} }")

# 음주빈도/걷기일수/근력운동일수 결측 → 중앙값
for col in ['음주빈도', '걷기일수', '근력운동일수']:
    n_miss = df[col].isnull().sum()
    if n_miss > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"  → {col} 결측 {n_miss}건 중앙값({med:.1f}) 대체")

# 음주빈도 OrdinalEncoding: 0~5
oe_freq = OrdinalEncoder(categories=[[0, 1, 2, 3, 4, 5]])
df['음주빈도_enc'] = oe_freq.fit_transform(df[['음주빈도']]).astype(int)

# 음주량 OrdinalEncoding: 0~5, NaN 유지 (sklearn>=1.4)
oe_amt = OrdinalEncoder(
    categories=[[0, 1, 2, 3, 4, 5]],
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=np.nan,
)
df['음주량_enc'] = oe_amt.fit_transform(df[['음주량']])

# 걷기일수/근력운동일수: 값 자체가 순서 (0~7) → 정수 변환만
df['걷기일수']     = df['걷기일수'].astype(int)
df['근력운동일수'] = df['근력운동일수'].astype(int)

df = df.drop(columns=['음주빈도', '음주량'])

print(f"[7] Ordinal Encoding 완료")
print(f"    음주빈도_enc: {df['음주빈도_enc'].value_counts().sort_index().to_dict()}")
print(f"    음주량_enc:  {df['음주량_enc'].value_counts(dropna=False).sort_index().to_dict()}")


# ──────────────────────────────────────────────
# 8. 직업 One-Hot Encoding (작업미상=8 포함)
#    1=관리전문 / 2=사무 / 3=서비스판매 / 4=농림어업
#    5=기능노무 / 6=주부학생 / 7=무직 / 8=작업미상
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
# 9. 현재흡연 결측 → 0 (비흡연으로 보수적 처리)
# ──────────────────────────────────────────────
n_sm = df['현재흡연'].isnull().sum()
if n_sm > 0:
    df['현재흡연'] = df['현재흡연'].fillna(0)
    print(f"[9] 현재흡연 결측 {n_sm}건 → 0(비흡연) 대체")
else:
    print(f"[9] 현재흡연 결측 없음")


# ──────────────────────────────────────────────
# 10. 잔여 결측 확인
# ──────────────────────────────────────────────
remaining = df.isnull().sum()
remaining = remaining[remaining > 0]
print(f"\n[10] 잔여 결측치:")
if remaining.empty:
    print("  → 없음 ✓  (음주량_enc NaN 은 의도적 유지)")
else:
    print(remaining)


# ──────────────────────────────────────────────
# 11. 최종 shape & 컬럼 확인
# ──────────────────────────────────────────────
print(f"\n[11] 최종 shape: {df.shape}")
print(f"     컬럼 목록: {list(df.columns)}")


# ──────────────────────────────────────────────
# 12. class_weight 계산 (XGBoost 적용 시 재논의)
# ──────────────────────────────────────────────
y_cols = ['고혈압유병', '당뇨유병', '이상지질혈증유병', '비만단계']
class_weights: dict = {}

print("\n[12] class_weight (balanced):")
for y in y_cols:
    y_clean = df[y].dropna()
    classes = np.sort(y_clean.unique())
    cw      = compute_class_weight(class_weight='balanced', classes=classes, y=y_clean)
    cw_dict = dict(zip(classes.astype(int), cw.round(4)))
    class_weights[y] = cw_dict
    print(f"  {y}: {cw_dict}")


# ──────────────────────────────────────────────
# 13. 저장
# ──────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[13] 저장 완료 → {OUTPUT_PATH}")
