"""
국민건강보험공단 건강검진 2024
타겟 변수 생성 + 중복도 EDA

실행환경: Python 3.10+
패키지  : pandas, numpy, matplotlib, seaborn
설치    : pip install pandas numpy matplotlib seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ────────────────────────────────────────────
# 0. 설정
# ────────────────────────────────────────────
BASE_DIR  = "/Users/admin/PycharmProjects/AH_03_03/etc/ml/ai_worker"
DATA_PATH = os.path.join(BASE_DIR, "data", "국민건강보험공단_건강검진정보_2024.CSV")
OUT_DIR   = os.path.join(BASE_DIR, "ml", "NHIS", "outputs", "Target_EDA")

os.makedirs(OUT_DIR, exist_ok=True)
print(f"출력 경로: {OUT_DIR}")

# 한글 폰트 설정 (없으면 영문으로 대체)
try:
    plt.rcParams["font.family"] = "AppleGothic"   # macOS
except:
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"  # Windows
    except:
        pass
plt.rcParams["axes.unicode_minus"] = False

# ────────────────────────────────────────────
# 1. 데이터 로드
# ────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding="cp949")
print(f"로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")

# ────────────────────────────────────────────
# 2. 결측코드 → NaN 처리 (이상치 제거)
# ────────────────────────────────────────────
df["허리둘레"]           = df["허리둘레"].replace(999, np.nan)
df["감마지티피"]          = df["감마지티피"].replace(9999, np.nan)
df["식전혈당(공복혈당)"]   = df["식전혈당(공복혈당)"].replace(991, np.nan)

# ────────────────────────────────────────────
# 3. 타겟 변수 6개 생성
# ────────────────────────────────────────────

# [1] 당뇨 위험 — 공복혈당 ≥ 100 mg/dL (전당뇨+당뇨, 대한당뇨병학회 2023)
df["target_diabetes"] = np.where(df["식전혈당(공복혈당)"] >= 100, 1, 0)
df.loc[df["식전혈당(공복혈당)"].isna(), "target_diabetes"] = np.nan

# [2] 고혈압 — 수축기 ≥ 140 OR 이완기 ≥ 90 mmHg (대한고혈압학회 2022)
df["target_hypertension"] = np.where(
    (df["수축기혈압"] >= 140) | (df["이완기혈압"] >= 90), 1, 0
)
df.loc[df["수축기혈압"].isna() & df["이완기혈압"].isna(), "target_hypertension"] = np.nan

# [3] 비만 — BMI ≥ 25 (대한비만학회 2022 한국인 기준)
#     주의: 신장/체중이 5단위 코드이므로 근사값 (오차 ±2~3 BMI 가능)
df["bmi_approx"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
df["target_obesity"] = np.where(df["bmi_approx"] >= 25, 1, 0)

# [4] 고지혈증 — 한국지질동맥경화학회 2022
#     TC≥200 OR LDL≥130 OR TG≥150 OR HDL 低(남<40, 여<50)
hdl_low = (
    ((df["성별코드"] == 1) & (df["HDL콜레스테롤"] < 40)) |
    ((df["성별코드"] == 2) & (df["HDL콜레스테롤"] < 50))
)
dyslipidemia = (
    (df["총콜레스테롤"] >= 200) |
    (df["LDL콜레스테롤"] >= 130) |
    (df["트리글리세라이드"] >= 150) |
    hdl_low
)
chol_all_na = df[["총콜레스테롤", "LDL콜레스테롤", "트리글리세라이드", "HDL콜레스테롤"]].isna().all(axis=1)
df["target_dyslipidemia"] = np.where(dyslipidemia, 1, 0)
df.loc[chol_all_na, "target_dyslipidemia"] = np.nan

# [5] 복합 대사증후군 — NCEP-ATP III (5항목 중 3개 이상)
#     한국인 복부비만 기준: 남≥90cm, 여≥85cm (대한비만학회)
abdom  = (((df["성별코드"]==1)&(df["허리둘레"]>=90)) | ((df["성별코드"]==2)&(df["허리둘레"]>=85)))
tg_hi  = df["트리글리세라이드"] >= 150
bp_ms  = (df["수축기혈압"] >= 130) | (df["이완기혈압"] >= 85)
gluc_ms = df["식전혈당(공복혈당)"] >= 100

ms_score = (abdom.astype(float) + tg_hi.astype(float) +
            hdl_low.astype(float) + bp_ms.astype(float) +
            gluc_ms.astype(float))
df["target_metabolic"] = np.where(ms_score >= 3, 1, 0)
ms_na = df[["허리둘레","트리글리세라이드","HDL콜레스테롤","수축기혈압","식전혈당(공복혈당)"]].isna().sum(axis=1)
df.loc[ms_na >= 3, "target_metabolic"] = np.nan

# [6] 간기능 이상 — 대한간학회 2023
#     AST or ALT > 40 IU/L OR GGT > 남63/여35 IU/L
liver_ast_alt = (df["혈청지오티(AST)"] > 40) | (df["혈청지피티(ALT)"] > 40)
ggt_high = (
    ((df["성별코드"] == 1) & (df["감마지티피"] > 63)) |
    ((df["성별코드"] == 2) & (df["감마지티피"] > 35))
)
df["target_liver"] = np.where(liver_ast_alt | ggt_high, 1, 0)
liver_na = df[["혈청지오티(AST)","혈청지피티(ALT)","감마지티피"]].isna().all(axis=1)
df.loc[liver_na, "target_liver"] = np.nan

targets = ["target_diabetes","target_hypertension","target_obesity",
           "target_dyslipidemia","target_metabolic","target_liver"]
labels  = ["당뇨위험","고혈압","비만","고지혈증","대사증후군","간기능이상"]

# ────────────────────────────────────────────
# 4. 타겟 분포 출력
# ────────────────────────────────────────────
print("\n=== 타겟 변수 분포 ===")
for t, l in zip(targets, labels):
    valid = df[t].dropna()
    pos   = int(valid.sum())
    neg   = int((valid == 0).sum())
    na    = int(df[t].isna().sum())
    print(f"[{l:8s}] 양성: {pos:>8,} ({pos/len(valid)*100:.1f}%)  "
          f"음성: {neg:>8,}  결측: {na:,}")

# ────────────────────────────────────────────
# 5. 중복도 EDA
# ────────────────────────────────────────────
valid_df = df[targets].dropna().astype(int)
valid_df.columns = labels
print(f"\n공통 유효 샘플 (6개 타겟 모두 비결측): {len(valid_df):,}행")

# 5-1. Phi 상관계수 행렬
corr = valid_df.corr().round(3)
print("\n=== 타겟 간 Phi 상관계수 ===")
print(corr.to_string())

# 5-2. A양성 중 B도 양성인 비율
print("\n=== A양성 중 B도 양성 비율 (%) ===")
header = f"{'':8s}" + "".join(f"{l:>8s}" for l in labels)
print(header)
for l1 in labels:
    row = f"{l1:8s}"
    mask = valid_df[l1] == 1
    for l2 in labels:
        if l1 == l2:
            row += f"{'  -':>8s}"
        else:
            pct = valid_df.loc[mask, l2].mean() * 100
            row += f"{pct:>7.1f}%"
    print(row)

# 5-3. 동시 양성 타겟 수 분포
valid_df["n_pos"] = valid_df[labels].sum(axis=1)
print("\n=== 동시 양성 타겟 수 분포 ===")
print(valid_df["n_pos"].value_counts().sort_index())

# ────────────────────────────────────────────
# 6. 시각화 (3개 플롯)
# ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("타겟 변수 중복도 EDA — 국민건강보험공단 건강검진 2024", fontsize=14)

# 플롯 1: 상관계수 히트맵
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, vmin=0, vmax=0.5,
    linewidths=0.5, ax=axes[0], cbar_kws={"shrink": 0.8}
)
axes[0].set_title("Phi 상관계수 히트맵", fontsize=11)
axes[0].tick_params(axis="x", rotation=30, labelsize=9)
axes[0].tick_params(axis="y", rotation=0, labelsize=9)

# 플롯 2: A양성 중 B양성 비율 히트맵
overlap_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
for l1 in labels:
    mask = valid_df[l1] == 1
    for l2 in labels:
        if l1 == l2:
            overlap_matrix.loc[l1, l2] = np.nan
        else:
            overlap_matrix.loc[l1, l2] = valid_df.loc[mask, l2].mean() * 100

sns.heatmap(
    overlap_matrix.astype(float), annot=True, fmt=".0f",
    cmap="YlOrRd", vmin=0, vmax=100,
    linewidths=0.5, ax=axes[1], cbar_kws={"shrink": 0.8}
)
axes[1].set_title("A양성 중 B도 양성 비율 (%)\n(행=A, 열=B)", fontsize=11)
axes[1].tick_params(axis="x", rotation=30, labelsize=9)
axes[1].tick_params(axis="y", rotation=0, labelsize=9)

# 플롯 3: 동시 양성 개수 분포
cnt = valid_df["n_pos"].value_counts().sort_index()
colors = ["#B4B2A9","#97C459","#EF9F27","#E24B4A","#D4537E","#534AB7","#2C2C2A"]
axes[2].bar(cnt.index, cnt.values, color=colors[:len(cnt)], edgecolor="white")
axes[2].set_xlabel("동시 양성 타겟 수")
axes[2].set_ylabel("인원 수")
axes[2].set_title("동시 양성 타겟 수 분포", fontsize=11)
axes[2].set_xticks(range(7))
for i, (x, y) in enumerate(zip(cnt.index, cnt.values)):
    axes[2].text(x, y + 500, f"{y:,}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "target_overlap_eda.png"), dpi=150, bbox_inches="tight")
plt.show()
print("\n시각화 저장 완료: target_overlap_eda.png")

# ────────────────────────────────────────────
# 7. 타겟 포함 데이터 저장 (선택)
# ────────────────────────────────────────────
# df.to_csv(os.path.join(OUT_DIR, "health_with_targets.csv"), index=False, encoding="utf-8-sig")
# print("저장 완료: health_with_targets.csv")
