"""
국민건강보험공단 건강검진 2023
기본 EDA (Exploratory Data Analysis)

실행환경: Python 3.10+
패키지  : pandas, numpy, matplotlib, seaborn
설치    : pip install pandas numpy matplotlib seaborn
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ────────────────────────────────────────────
# 0. 설정
# ────────────────────────────────────────────
BASE_DIR  = "/Users/admin/PycharmProjects/AH_03_03/etc/ml/ai_worker"
DATA_PATH = os.path.join(BASE_DIR, "data", "국민건강보험공단_건강검진정보_2023.CSV")
OUT_DIR   = os.path.join(BASE_DIR, "ml", "NHIS", "outputs", "Basic_EDA_2023")

os.makedirs(OUT_DIR, exist_ok=True)  # 폴더 없으면 자동 생성
print(f"출력 경로: {OUT_DIR}")

try:
    plt.rcParams["font.family"] = "AppleGothic"    # macOS
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
# 2. 기본 정보
# ────────────────────────────────────────────
print("\n=== 컬럼 목록 & 데이터 타입 ===")
print(df.dtypes.to_string())

print("\n=== 기술통계 (수치형) ===")
print(df.describe().round(2).to_string())

# ────────────────────────────────────────────
# 3. 결측값 현황
# ────────────────────────────────────────────
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
miss_df     = pd.DataFrame({"결측수": missing, "결측률(%)": missing_pct})
miss_df     = miss_df[miss_df["결측수"] > 0].sort_values("결측률(%)", ascending=False)

print("\n=== 결측값 현황 (결측 있는 컬럼만) ===")
print(miss_df.to_string())

# ────────────────────────────────────────────
# 4. 결측코드 → NaN 처리
# ────────────────────────────────────────────
df["허리둘레"]          = df["허리둘레"].replace(999, np.nan)
df["감마지티피"]         = df["감마지티피"].replace(9999, np.nan)
df["식전혈당(공복혈당)"]  = df["식전혈당(공복혈당)"].replace(991, np.nan)

# ────────────────────────────────────────────
# 5. 범주형 컬럼 분포
# ────────────────────────────────────────────
cat_cols = {
    "성별코드":          {1: "남성", 2: "여성"},
    "흡연상태":          {1: "비흡연", 2: "과거흡연", 3: "현재흡연"},
    "음주여부":          {0: "안함", 1: "음주"},
    "구강검진수검여부":   {0: "미수검", 1: "수검"},
    "요단백":            {1:"음성", 2:"+/-", 3:"1+", 4:"2+", 5:"3+", 6:"4+"},
}

print("\n=== 범주형 컬럼 분포 ===")
for col, mapping in cat_cols.items():
    vc = df[col].value_counts(dropna=False)
    print(f"\n[{col}]")
    for k, v in vc.items():
        label = mapping.get(k, str(k)) if not pd.isna(k) else "결측"
        print(f"  {label:10s}: {v:>8,} ({v/len(df)*100:.1f}%)")

# 연령대 코드 → 실제 구간 매핑
age_map = {5:"25-29",6:"30-34",7:"35-39",8:"40-44",9:"45-49",
           10:"50-54",11:"55-59",12:"60-64",13:"65-69",14:"70-74",
           15:"75-79",16:"80-84",17:"85-89",18:"90+"}
print("\n[연령대코드(5세단위)]")
for k, v in df["연령대코드(5세단위)"].value_counts().sort_index().items():
    label = age_map.get(k, str(k))
    print(f"  {label:8s}: {v:>8,} ({v/len(df)*100:.1f}%)")

# ────────────────────────────────────────────
# 6. 이상치 탐지 (주요 수치형 컬럼)
# ────────────────────────────────────────────
outlier_cols = {
    "식전혈당(공복혈당)":   (40, 500),
    "수축기혈압":            (60, 250),
    "이완기혈압":            (30, 150),
    "허리둘레":              (40, 200),
    "총콜레스테롤":          (50, 1000),
    "트리글리세라이드":       (10, 3000),
    "혈청지오티(AST)":        (5, 2000),
    "혈청지피티(ALT)":        (1, 2000),
    "감마지티피":             (1, 2000),
    "혈색소":                (3, 25),
}

print("\n=== 이상치 탐지 (정상 범위 초과) ===")
for col, (lo, hi) in outlier_cols.items():
    if col not in df.columns:
        continue
    s    = df[col].dropna()
    out  = ((s < lo) | (s > hi)).sum()
    print(f"[{col:20s}] 범위 {lo}~{hi} | 이상치: {out:,}건 ({out/len(s)*100:.2f}%)")

# ────────────────────────────────────────────
# 7. 시각화
# ────────────────────────────────────────────

# ── 7-1. 결측률 막대 ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = miss_df["결측률(%)"].apply(
    lambda x: "#E24B4A" if x >= 50 else ("#EF9F27" if x >= 1 else "#1D9E75")
)
ax.barh(miss_df.index, miss_df["결측률(%)"], color=colors)
ax.set_xlabel("결측률 (%)")
ax.set_title("컬럼별 결측률")
ax.axvline(50, color="#E24B4A", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axvline(1,  color="#EF9F27", linewidth=0.8, linestyle="--", alpha=0.5)
for i, (v, c) in enumerate(zip(miss_df["결측률(%)"], colors)):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_01_missing_2023.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n저장: eda_01_missing_2023.png")

# ── 7-2. 성별 / 연령대 분포 ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 성별 파이
gender_cnt = df["성별코드"].value_counts().sort_index()
axes[0].pie(
    gender_cnt.values,
    labels=["남성", "여성"],
    autopct="%1.1f%%",
    colors=["#378ADD", "#D4537E"],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
axes[0].set_title("성별 비율")

# 연령대 막대
age_cnt = df["연령대코드(5세단위)"].value_counts().sort_index()
age_labels = [age_map.get(k, str(k)) for k in age_cnt.index]
axes[1].bar(age_labels, age_cnt.values, color="#378ADD", edgecolor="white")
axes[1].set_xlabel("연령대")
axes[1].set_ylabel("수검자 수")
axes[1].set_title("연령대 분포")
axes[1].tick_params(axis="x", rotation=45)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_02_demographic_2023.png"), dpi=150, bbox_inches="tight")
plt.close()
print("저장: eda_02_demographic_2023.png")

# ── 7-3. 주요 임상수치 분포 (히스토그램) ────────
clin_cols = [
    "수축기혈압", "이완기혈압", "식전혈당(공복혈당)",
    "총콜레스테롤", "트리글리세라이드", "HDL콜레스테롤",
    "혈색소", "혈청지오티(AST)", "감마지티피"
]
# 이상치 클리핑 후 시각화
clip_ranges = {
    "수축기혈압":       (60, 220),
    "이완기혈압":       (30, 140),
    "식전혈당(공복혈당)": (50, 300),
    "총콜레스테롤":     (50, 500),
    "트리글리세라이드":  (10, 800),
    "HDL콜레스테롤":    (10, 200),
    "혈색소":           (4, 22),
    "혈청지오티(AST)":   (5, 200),
    "감마지티피":        (1, 300),
}

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
fig.suptitle("주요 임상수치 분포", fontsize=13)
for ax, col in zip(axes.flatten(), clin_cols):
    s = df[col].dropna()
    if col in clip_ranges:
        lo, hi = clip_ranges[col]
        s = s.clip(lo, hi)
    ax.hist(s, bins=60, color="#378ADD", edgecolor="white", linewidth=0.3)
    ax.set_title(col, fontsize=10)
    ax.set_ylabel("빈도")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_03_clinical_dist_2023.png"), dpi=150, bbox_inches="tight")
plt.close()
print("저장: eda_03_clinical_dist_2023.png")

# ── 7-4. 생활습관 분포 ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("생활습관 분포", fontsize=12)

# 흡연
smoke_cnt = df["흡연상태"].value_counts().sort_index().dropna()
smoke_labels = {1: "비흡연", 2: "과거흡연", 3: "현재흡연"}
axes[0].bar(
    [smoke_labels.get(k, k) for k in smoke_cnt.index],
    smoke_cnt.values,
    color=["#1D9E75", "#888780", "#E24B4A"],
    edgecolor="white"
)
axes[0].set_title("흡연상태")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# 음주
drink_cnt = df["음주여부"].value_counts().sort_index().dropna()
axes[1].pie(
    drink_cnt.values,
    labels=["비음주", "음주"],
    autopct="%1.1f%%",
    colors=["#888780", "#639922"],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
axes[1].set_title("음주여부")

# 요단백
prot_cnt = df["요단백"].value_counts().sort_index().dropna()
prot_labels = {1:"음성", 2:"+/-", 3:"1+", 4:"2+", 5:"3+", 6:"4+"}
axes[2].bar(
    [prot_labels.get(k, k) for k in prot_cnt.index],
    prot_cnt.values,
    color="#534AB7",
    edgecolor="white"
)
axes[2].set_title("요단백")
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_04_lifestyle_2023.png"), dpi=150, bbox_inches="tight")
plt.close()
print("저장: eda_04_lifestyle_2023.png")

# ── 7-5. 수치형 상관계수 히트맵 ─────────────
num_cols = [
    "신장(5cm단위)", "체중(5kg단위)", "허리둘레",
    "수축기혈압", "이완기혈압", "식전혈당(공복혈당)",
    "총콜레스테롤", "트리글리세라이드", "HDL콜레스테롤", "LDL콜레스테롤",
    "혈색소", "혈청크레아티닌",
    "혈청지오티(AST)", "혈청지피티(ALT)", "감마지티피"
]
corr_df = df[num_cols].corr().round(2)

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.eye(len(corr_df), dtype=bool)
sns.heatmap(
    corr_df, annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, vmin=-1, vmax=1,
    mask=mask, linewidths=0.3,
    ax=ax, annot_kws={"size": 7},
    cbar_kws={"shrink": 0.8}
)
ax.set_title("수치형 피처 상관계수 히트맵", fontsize=12)
ax.tick_params(axis="x", rotation=40, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_05_feature_corr_2023.png"), dpi=150, bbox_inches="tight")
plt.close()
print("저장: eda_05_feature_corr_2023.png")

# ── 7-6. 성별 × 연령대 별 주요 수치 박스플롯 ─
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("성별 × 연령대별 주요 임상수치", fontsize=13)

plot_pairs = [
    ("수축기혈압",       "수축기혈압 (mmHg)"),
    ("식전혈당(공복혈당)", "공복혈당 (mg/dL)"),
    ("체중(5kg단위)",    "체중 (kg)"),
    ("혈청지피티(ALT)",  "ALT (IU/L)"),
]

df["성별"] = df["성별코드"].map({1: "남성", 2: "여성"})
df["연령대"] = df["연령대코드(5세단위)"].map(age_map)

for ax, (col, ylabel) in zip(axes.flatten(), plot_pairs):
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    sub = df[(df[col] >= lo) & (df[col] <= hi)].copy()
    # 연령대 순서 정렬
    age_order = [v for k, v in sorted(age_map.items()) if v in sub["연령대"].unique()]
    sns.boxplot(
        data=sub, x="연령대", y=col, hue="성별",
        order=age_order,
        palette={"남성": "#378ADD", "여성": "#D4537E"},
        linewidth=0.8, fliersize=1, ax=ax
    )
    ax.set_title(ylabel, fontsize=10)
    ax.set_xlabel("연령대")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=8, title_fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "eda_06_age_gender_box_2023.png"), dpi=150, bbox_inches="tight")
plt.close()
print("저장: eda_06_age_gender_box_2023.png")

print("\n=== 모든 EDA 완료 ===")
print("생성 파일:")
for i, name in enumerate([
    "eda_01_missing_2023.png         — 결측률 현황",
    "eda_02_demographic_2023.png     — 성별·연령대 분포",
    "eda_03_clinical_dist_2023.png   — 임상수치 히스토그램",
    "eda_04_lifestyle_2023.png       — 생활습관 분포",
    "eda_05_feature_corr_2023.png    — 수치형 피처 상관계수",
    "eda_06_age_gender_box_2023.png  — 성별×연령대 박스플롯",
], 1):
    print(f"  {i}. {name}")
