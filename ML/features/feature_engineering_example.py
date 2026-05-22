"""
feature_engineering_example.py
────────────────────────────────
모델 파일에서 피처 엔지니어링 파일을 불러다 쓰는 예시

폴더 구조:
    Desktop/
    └── final_project/
        └── ML/
            ├── data/
            │   └── x1_preprocessed.csv
            ├── features/
            │   ├── fe_age_bin.py
            │   ├── fe_family_sum.py
            │   ├── fe_bmi_bin.py
            │   ├── fe_alcohol.py
            │   ├── fe_exercise.py
            │   ├── fe_body.py
            │   └── fe_age_family.py
            └── models/
                └── 이 파일 위치
"""

import os
import sys

import pandas as pd

# ── 경로 설정 (상대 경로) ──────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "x1_preprocessed.csv")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
NPY_DIR = os.path.join(BASE_DIR, "outputs", "oof")

# features 폴더를 import 경로에 추가
sys.path.insert(0, FEATURES_DIR)

# ── 피처 엔지니어링 함수 import ────────────────────────────────
from fe_age_bin import add_age_bin  # 나이 구간화
from fe_alcohol import add_alcohol_load  # 음주 복합
from fe_bmi_bin import add_bmi_bin  # BMI 구간화
from fe_exercise import add_exercise_total  # 신체활동 합산
from fe_family_sum import add_family_sum  # 가족력 합산

# from fe_body        import add_body_features   # 체형 복합 (이상지질혈증 특화)
# from fe_age_family  import add_age_family_interaction  # 나이×가족력 상호작용

# ── 데이터 로드 ────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"로드 완료 | shape: {df.shape}")

# ── 피처 엔지니어링 적용 ───────────────────────────────────────
# 순서 중요: 나이_구간 → 가족력 합산 → 상호작용 순으로 적용

df = add_age_bin(df)  # 나이_구간 추가
df = add_family_sum(df)  # 고혈압/당뇨/고지혈증 가족력_합계 추가
df = add_bmi_bin(df)  # BMI_구간 추가
df = add_alcohol_load(df)  # 음주_총부하 추가
df = add_exercise_total(df)  # 총운동일수 추가
# df = add_body_features(df)     # 체형 복합 (이상지질혈증 모델에서만 주석 해제)
# df = add_age_family_interaction(df)  # 나이×가족력 (고혈압/당뇨 모델에서만 주석 해제)

print(f"\n피처 엔지니어링 완료 | shape: {df.shape}")
print(f"추가된 피처: {[c for c in df.columns if c not in pd.read_csv(DATA_PATH).columns]}")

# ── 피처 / 타겟 분리 ───────────────────────────────────────────
TARGET = "당뇨유병"  # 질환에 맞게 변경
DROP_COLS = ["고혈압유병", "당뇨유병", "이상지질혈증유병"]

data = df.dropna(subset=[TARGET]).copy()
X = data.drop(columns=DROP_COLS)
y = data[TARGET].astype(int)

print(f"\nX shape: {X.shape}")
print(
    f"추가된 피처 확인: {[c for c in X.columns if c in ['나이_구간', 'BMI_구간', '당뇨가족력_합계', '음주_총부하', '총운동일수']]}"
)
