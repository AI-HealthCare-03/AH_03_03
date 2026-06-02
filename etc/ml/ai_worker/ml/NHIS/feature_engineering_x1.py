"""
국민건강보험공단 건강검진 2024
X1 피처 엔지니어링 — 기본 정보만으로 예측 (검진 전 입력 가능한 변수)

X1 피처: 성별, 연령대, 신장, 체중, 허리둘레, 흡연상태, 음주여부, BMI(파생)
ML 타겟 5개: 당뇨위험, 고혈압, 이상지질혈증, 대사증후군, 간기능이상
룰엔진: 비만 (BMI 구간으로 즉시 판정 — ML 불필요)

실행환경: Python 3.10+
패키지  : pandas, numpy, scikit-learn
설치    : pip install pandas numpy scikit-learn
"""

import os
import pandas as pd
import numpy as np

# ────────────────────────────────────────────
# 0. 설정
# ────────────────────────────────────────────
BASE_DIR  = "/Users/admin/PycharmProjects/AH_03_03/etc/ml/ai_worker"
DATA_PATH = os.path.join(BASE_DIR, "data", "국민건강보험공단_건강검진정보_2024.CSV")
OUT_DIR   = os.path.join(BASE_DIR, "ml", "NHIS", "outputs", "Feature_Engineering")

os.makedirs(OUT_DIR, exist_ok=True)
print(f"출력 경로: {OUT_DIR}")

# ────────────────────────────────────────────
# 1. 데이터 로드
# ────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding="cp949")
print(f"로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")

# ────────────────────────────────────────────
# 2. 이상치 → NaN 처리 (의학적 CLINICAL_BOUNDS 기준)
#    범위 벗어난 값 → NaN (클램핑 X)
#    기존 코드값(999, 9999, 991)은 범위 초과로 자동 처리됨
# ────────────────────────────────────────────
CLINICAL_BOUNDS = {
    "신장(5cm단위)":      (100, 250),   # 기네스 최장신 251cm / 100 미만은 성인 측정오류
    "체중(5kg단위)":      (20,  350),   # 20 미만 성인 측정오류 / 350 초과 국내 현실적 불가
    "허리둘레":           (40,  200),   # 999 코드값 → 범위 초과로 자동 NaN
    "수축기혈압":         (60,  280),   # 60 미만 심인성 쇼크 / 280 초과 고혈압 응급 상한
    "이완기혈압":         (40,  150),   # 임상적 측정 가능 범위
    "식전혈당(공복혈당)": (40,  600),   # 40 미만 중증저혈당 / 600 초과 HHS 수준 / 991 코드값 포함
    "총콜레스테롤":       (50,  700),   # 가족성고콜레스테롤혈증 극단값 기준
    "LDL콜레스테롤":      (10,  500),
    "HDL콜레스테롤":      (10,  150),   # 150 초과 측정오류 수준
    "트리글리세라이드":   (20,  5000),  # 급성췌장염 유발 수준 / 드물게 5000대 보고
    "혈청지오티(AST)":    (5,   5000),  # 급성간염/허혈성간염 시 수천 단위 실제 보고
    "혈청지피티(ALT)":    (5,   5000),
    "감마지티피":         (5,   3000),  # 알코올성 간질환 극단값 / 9999 코드값 포함
}

print("\n=== 이상치 → NaN 처리 (CLINICAL_BOUNDS) ===")
total_replaced = 0
for col, (lo, hi) in CLINICAL_BOUNDS.items():
    if col not in df.columns:
        continue
    before = df[col].isna().sum()
    df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi), other=np.nan)
    replaced = df[col].isna().sum() - before
    if replaced > 0:
        total_replaced += replaced
        print(f"  [{col}] 범위({lo}~{hi}) 벗어난 값 → NaN: {replaced:,}건")
print(f"  총 NaN 전환: {total_replaced:,}건")

# ────────────────────────────────────────────
# 3. 타겟 변수 6개 생성
# ────────────────────────────────────────────
# [1] 당뇨 위험
df["target_diabetes"] = np.where(df["식전혈당(공복혈당)"] >= 100, 1, 0)
df.loc[df["식전혈당(공복혈당)"].isna(), "target_diabetes"] = np.nan

# [2] 고혈압
df["target_hypertension"] = np.where(
    (df["수축기혈압"] >= 140) | (df["이완기혈압"] >= 90), 1, 0
)
df.loc[df["수축기혈압"].isna() & df["이완기혈압"].isna(), "target_hypertension"] = np.nan

# [3] 비만 → 룰엔진으로 처리 (ML 타겟 제외)
#     obesity_rule_engine(bmi) 함수 사용 — 피처 엔지니어링 섹션 참고
df["bmi"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)

# [4] 고지혈증
hdl_low = (
    ((df["성별코드"] == 1) & (df["HDL콜레스테롤"] < 40)) |
    ((df["성별코드"] == 2) & (df["HDL콜레스테롤"] < 50))
)
dyslipidemia = (
    (df["총콜레스테롤"] >= 200) | (df["LDL콜레스테롤"] >= 130) |
    (df["트리글리세라이드"] >= 150) | hdl_low
)
chol_all_na = df[["총콜레스테롤","LDL콜레스테롤","트리글리세라이드","HDL콜레스테롤"]].isna().all(axis=1)
df["target_dyslipidemia"] = np.where(dyslipidemia, 1, 0)
df.loc[chol_all_na, "target_dyslipidemia"] = np.nan

# [5] 대사증후군
abdom   = (((df["성별코드"]==1)&(df["허리둘레"]>=90)) | ((df["성별코드"]==2)&(df["허리둘레"]>=85)))
tg_hi   = df["트리글리세라이드"] >= 150
bp_ms   = (df["수축기혈압"] >= 130) | (df["이완기혈압"] >= 85)
gluc_ms = df["식전혈당(공복혈당)"] >= 100
ms_score = (abdom.astype(float) + tg_hi.astype(float) +
            hdl_low.astype(float) + bp_ms.astype(float) + gluc_ms.astype(float))
df["target_metabolic"] = np.where(ms_score >= 3, 1, 0)
ms_na = df[["허리둘레","트리글리세라이드","HDL콜레스테롤","수축기혈압","식전혈당(공복혈당)"]].isna().sum(axis=1)
df.loc[ms_na >= 3, "target_metabolic"] = np.nan

# [6] 간기능 이상
liver_ast_alt = (df["혈청지오티(AST)"] > 40) | (df["혈청지피티(ALT)"] > 40)
ggt_high = (
    ((df["성별코드"] == 1) & (df["감마지티피"] > 63)) |
    ((df["성별코드"] == 2) & (df["감마지티피"] > 35))
)
df["target_liver"] = np.where(liver_ast_alt | ggt_high, 1, 0)
liver_na = df[["혈청지오티(AST)","혈청지피티(ALT)","감마지티피"]].isna().all(axis=1)
df.loc[liver_na, "target_liver"] = np.nan

targets = ["target_diabetes","target_hypertension",
           "target_dyslipidemia","target_metabolic","target_liver"]

# ────────────────────────────────────────────
# 4. X1 피처 선택
# ────────────────────────────────────────────
x1_raw = ["성별코드","연령대코드(5세단위)","신장(5cm단위)","체중(5kg단위)","허리둘레","흡연상태","음주여부"]
df_x1  = df[x1_raw + targets].copy()

# ────────────────────────────────────────────
# 5. 파생 피처 생성
# ────────────────────────────────────────────

# [5-1] BMI
df_x1["bmi"] = df_x1["체중(5kg단위)"] / ((df_x1["신장(5cm단위)"] / 100) ** 2)
df_x1["bmi"] = df_x1["bmi"].round(2)

# [5-2] BMI 구간 — 대한비만학회 2022 한국인 기준 6단계
#   0=저체중(<18.5) / 1=정상(18.5~23) / 2=과체중(23~25)
#   3=1단계비만(25~30) / 4=2단계비만(30~35) / 5=3단계비만(35+)
bmi_bins   = [0, 18.5, 23.0, 25.0, 30.0, 35.0, 999]
bmi_labels = [0, 1, 2, 3, 4, 5]
df_x1["bmi_category"] = pd.cut(
    df_x1["bmi"], bins=bmi_bins, labels=bmi_labels, right=False
).astype(float)

# ────────────────────────────────────────────
# [룰 엔진] 비만 판정 함수 — ML 타겟 제외, 서비스에서 직접 사용
# ────────────────────────────────────────────
def obesity_rule_engine(bmi: float) -> dict:
    """
    BMI 입력 → 비만 단계 즉시 판정 (대한비만학회 2022)
    Returns:
        stage      : 0~5 단계
        label      : 단계명
        is_obese   : 비만 여부 (단계 3 이상)
        description: 설명
    """
    if bmi < 18.5:
        return {"stage": 0, "label": "저체중",        "is_obese": False, "description": "BMI 18.5 미만"}
    elif bmi < 23.0:
        return {"stage": 1, "label": "정상",          "is_obese": False, "description": "BMI 18.5~22.9"}
    elif bmi < 25.0:
        return {"stage": 2, "label": "과체중(비만전단계)", "is_obese": False, "description": "BMI 23.0~24.9"}
    elif bmi < 30.0:
        return {"stage": 3, "label": "1단계 비만",    "is_obese": True,  "description": "BMI 25.0~29.9"}
    elif bmi < 35.0:
        return {"stage": 4, "label": "2단계 비만",    "is_obese": True,  "description": "BMI 30.0~34.9"}
    else:
        return {"stage": 5, "label": "3단계 비만(고도비만)", "is_obese": True, "description": "BMI 35.0 이상"}

# 룰엔진 적용 예시
print("\n=== 룰엔진 비만 판정 예시 ===")
for test_bmi in [17.0, 21.0, 24.0, 27.0, 32.0, 37.0]:
    result = obesity_rule_engine(test_bmi)
    print(f"  BMI {test_bmi:5.1f} → {result['label']:18s} ({result['description']})")

# [5-3] 허리/신장 비율 (WHtR — Waist-to-Height Ratio)
#   복부비만 지표: 0.5 이상이면 위험 (BMI보다 대사질환 예측력 높음)
df_x1["waist_height_ratio"] = (df_x1["허리둘레"] / df_x1["신장(5cm단위)"]).round(3)

# [5-4] 연령대 실제 나이 중간값으로 변환
#   연령대 코드 5 = 25~29세 → 중간값 27
age_mid = {5:27, 6:32, 7:37, 8:42, 9:47, 10:52,
           11:57, 12:62, 13:67, 14:72, 15:77,
           16:82, 17:87, 18:92}
df_x1["age_mid"] = df_x1["연령대코드(5세단위)"].map(age_mid)

# [5-5] 성별 × 연령대 상호작용
#   고혈압은 65세 이후 여성이 남성 추월 → 상호작용 피처가 유효
df_x1["gender_age"] = df_x1["성별코드"].astype(str) + "_" + df_x1["연령대코드(5세단위)"].astype(str)
# 레이블 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_x1["gender_age_enc"] = le.fit_transform(df_x1["gender_age"])
df_x1.drop(columns=["gender_age"], inplace=True)

# [5-6] 비만 복합 지표 (BMI ≥ 25 AND WHtR ≥ 0.5)
df_x1["obesity_combined"] = (
    (df_x1["bmi"] >= 25) & (df_x1["waist_height_ratio"] >= 0.5)
).astype(float)

# [5-7] 흡연 이진화 (현재흡연 여부)
#   1=비흡연, 2=과거흡연, 3=현재흡연 → 현재흡연=1, 나머지=0
df_x1["smoking_current"] = (df_x1["흡연상태"] == 3).astype(float)

# [5-8] 흡연 경험 여부 (과거+현재)
df_x1["smoking_ever"] = (df_x1["흡연상태"] >= 2).astype(float)

# ────────────────────────────────────────────
# 6. 결측 현황 확인 (처리 X — 모델링 코드에서 KNNImputer로 처리)
# ────────────────────────────────────────────
# X1 피처 결측: 허리둘레 499건, 흡연상태 118건, 음주여부 45건 → 총 651행 (0.065%)
# → train/test split 이후 KNNImputer(n_neighbors=5) 적용 예정
# → 여기서 처리하면 데이터 누수 발생하므로 결측 그대로 저장

missing_x1 = df_x1[x1_raw].isnull().sum()
missing_x1 = missing_x1[missing_x1 > 0]
print(f"\n=== X1 피처 결측 현황 (모델링에서 KNN으로 처리 예정) ===")
print(missing_x1)
print(f"결측 있는 행: {df_x1[x1_raw].isnull().any(axis=1).sum():,}건 "
      f"({df_x1[x1_raw].isnull().any(axis=1).sum()/len(df_x1)*100:.3f}%)")

# ────────────────────────────────────────────
# 7. 최종 피처 정리
# ────────────────────────────────────────────
# 원본 코드 컬럼 (모델 입력에서 제외 — 파생 피처로 대체)
# 신장/체중은 BMI에 이미 반영되어 있지만 트리 모델에선 그대로 두는 게 나음
# 연령대 코드는 age_mid로 대체

feature_cols = [
    # 원본
    "성별코드",
    "신장(5cm단위)",
    "체중(5kg단위)",
    "허리둘레",
    "음주여부",
    # 파생
    "bmi",
    "bmi_category",
    "waist_height_ratio",
    "age_mid",
    "gender_age_enc",
    "obesity_combined",
    "smoking_current",
    "smoking_ever",
]

print(f"\n=== 최종 X1 피처 목록 ({len(feature_cols)}개) ===")
for i, f in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {f}")

print(f"\n=== ML 타겟 5개 유효 샘플 수 ===")
target_labels = ["당뇨위험","고혈압","이상지질혈증","대사증후군","간기능이상"]
for t, l in zip(targets, target_labels):
    valid = df_x1[t].dropna()
    pos   = int(valid.sum())
    print(f"  [{l:6s}] 전체: {len(valid):>8,} | 양성: {pos:>7,} ({pos/len(valid)*100:.1f}%)")

# ────────────────────────────────────────────
# 8. 저장
# ────────────────────────────────────────────
save_cols = feature_cols + targets
df_save   = df_x1[save_cols].copy()

out_path  = os.path.join(OUT_DIR, "x1_features.csv")
df_save.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {out_path}")
print(f"최종 shape: {df_save.shape}")

# 피처 기술통계
print("\n=== 피처 기술통계 ===")
print(df_save[feature_cols].describe().round(2).to_string())
