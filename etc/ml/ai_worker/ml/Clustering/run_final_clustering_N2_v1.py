"""
run_final_clustering.py

N2 설정 기준 전체 331,372명 최종 군집화
- K=6, StandardScaler, 이완기혈압 제거(14개), 크레아티닌 ≤5
- 후처리: 당뇨 플래그 (식전혈당 ≥ 126) 추가
- 람다 함수 사용 금지
"""

from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent  # ai_worker/
DATA_PATH = ROOT / "data/lipid_only.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "final"

# ────────────────────────────────────────────────
# N2 확정 설정
# ────────────────────────────────────────────────
K = 6
BEST_GAMMA = 0.3        # N2 실험 최적 gamma
N_INIT = 20
CLUSTER_SEED = 42
SAMPLE_SEED = 42

# ────────────────────────────────────────────────
# 변수 구성 (N2 기준 — 이완기혈압 제거, 14개)
# ────────────────────────────────────────────────
CONT_COLS = [
    "신장(5cm단위)",
    "체중(5kg단위)",
    "허리둘레",
    "수축기혈압",
    # 이완기혈압 제거 (수축기혈압과 r=0.68, 중복 신호)
    "식전혈당(공복혈당)",
    "총콜레스테롤",
    "HDL콜레스테롤",
    "LDL콜레스테롤",
    "혈색소",
    "혈청크레아티닌",
    "혈청지오티(AST)",
    "혈청지피티(ALT)",
    "감마지티피",
    "BMI",
]
CAT_COLS = ["성별코드", "연령대코드(5세단위)", "흡연상태", "음주여부", "요단백"]

# ────────────────────────────────────────────────
# 임상 기준 (크레아티닌 ≤5 적용)
# ────────────────────────────────────────────────
CLINICAL_BOUNDS = {
    "수축기혈압":         (60,  250),
    "이완기혈압":         (30,  150),
    "식전혈당(공복혈당)": (50,  500),
    "총콜레스테롤":       (50,  500),
    "HDL콜레스테롤":      (10,  150),
    "LDL콜레스테롤":      (10,  400),
    "혈색소":             (5,    22),
    "혈청크레아티닌":     (0.3,   5),  # 15 → 5 변경 (말기 신부전 수준 제거)
    "혈청지오티(AST)":    (5,   200),
    "혈청지피티(ALT)":    (5,   200),
    "감마지티피":         (1,   300),
    "허리둘레":           (40,  160),
}

DROP_COLS = [
    "치아우식증유무", "결손치 유무", "치아마모증유무",
    "제3대구치(사랑니) 이상", "치석", "기준년도",
    "가입자일련번호", "시도코드", "구강검진수검여부",
    "시력(좌)", "시력(우)", "청력(좌)", "청력(우)",
]

# ────────────────────────────────────────────────
# 전처리 함수
# ────────────────────────────────────────────────
def remove_outliers_clinical(df):
    df = df.copy()
    for col, (lower, upper) in CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
    return df


def add_bmi(df):
    df = df.copy()
    df["BMI"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
    return df


def add_clinical_labels(df):
    df = df.copy()
    df["고혈압_기준"]       = ((df["수축기혈압"] >= 140) | (df["이완기혈압"] >= 90)).astype(int)
    df["당뇨_기준"]         = (df["식전혈당(공복혈당)"] >= 126).astype(int)
    df["이상지질혈증_기준"] = (df["총콜레스테롤"] >= 240).astype(int)
    df["비만_기준"]         = (df["BMI"] >= 25).astype(int)
    return df


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def main():
    print("[0] 데이터 로드 및 전처리")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = remove_outliers_clinical(df)
    df = add_bmi(df)
    df = add_clinical_labels(df)
    df = df.dropna(subset=CONT_COLS + CAT_COLS)
    print(f"  전처리 후 전체: {len(df):,}명")

    # # 테스트용 샘플링 — 전체 실행 시 아래 두 줄 주석 처리
    # df = df.sample(n=50000, random_state=SAMPLE_SEED).reset_index(drop=True)
    # print(f"  샘플링 후: {len(df):,}명 (seed={SAMPLE_SEED})")

    print(f"  연속형: {len(CONT_COLS)}개 | 범주형: {len(CAT_COLS)}개")

    # 원본 스케일 보존
    df_orig = df.copy()

    # 스케일링
    df_use = df[CONT_COLS + CAT_COLS].copy()
    cat_indices = [df_use.columns.get_loc(c) for c in CAT_COLS]
    scaler = StandardScaler()
    df_use[CONT_COLS] = scaler.fit_transform(df_use[CONT_COLS])
    X = df_use.values
    X_cont = df_use[CONT_COLS].values

    # 최종 군집화
    print(f"\n[1] 최종 군집화 | K={K} | gamma={BEST_GAMMA} | n_init={N_INIT}")
    kp = KPrototypes(
        n_clusters=K,
        init="Huang",
        n_init=N_INIT,
        gamma=BEST_GAMMA,
        random_state=CLUSTER_SEED,
        verbose=0,
    )
    labels = kp.fit_predict(X, categorical=cat_indices)
    counts = np.bincount(labels)

    sil = silhouette_score(X_cont, labels, sample_size=10000, random_state=CLUSTER_SEED)
    min_ratio = counts.min() / len(labels)

    print(f"  Silhouette:     {sil:.4f}")
    print(f"  최소군집비율:   {min_ratio:.3f}")
    print(f"  군집 크기:      {sorted(counts, reverse=True)}")

    # 결과 저장
    df_orig["cluster"] = labels
    # ── 후처리: 당뇨 플래그
    df_orig["당뇨_고위험"] = (df_orig["식전혈당(공복혈당)"] >= 126).astype(int)
    df_orig["당뇨_전단계"] = (
        (df_orig["식전혈당(공복혈당)"] >= 100) &
        (df_orig["식전혈당(공복혈당)"] < 126)
    ).astype(int)

    # 군집별 분석
    clinical_cols = ["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]
    summary = df_orig.groupby("cluster")[clinical_cols].mean().round(3) * 100
    cluster_count = df_orig["cluster"].value_counts().sort_index()
    summary.insert(0, "샘플수", cluster_count)
    summary.insert(1, "비율(%)", (cluster_count / len(df_orig) * 100).round(1))
    # 당뇨 플래그 군집별 집계
    diabetes_summary = df_orig.groupby("cluster")[["당뇨_고위험", "당뇨_전단계"]].mean().round(3) * 100

    cont_mean = df_orig.groupby("cluster")[CONT_COLS].mean().round(2)
    gender_dist = df_orig.groupby("cluster")["성별코드"].value_counts(normalize=True).unstack().fillna(0).round(3)
    gender_dist.columns = [f"성별_{int(c)}" for c in gender_dist.columns]
    age_mean = df_orig.groupby("cluster")["연령대코드(5세단위)"].mean().round(1).rename("평균연령대코드")

    print(f"\n[2] 군집별 임상 유병률")
    print(summary.to_string())
    print(f"\n[3] 당뇨 후처리 플래그")
    print(diabetes_summary.to_string())
    print(f"\n[4] 군집별 연속형 변수 평균 (원본 스케일)")
    print(cont_mean.to_string())
    print(f"\n[5] 성별 분포")
    print(gender_dist.to_string())
    print(f"\n[6] 연령대 분포")
    print(age_mean.to_string())

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_orig.to_csv(OUTPUT_DIR / "final_clustered_data.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "final_cluster_summary.csv")
    diabetes_summary.to_csv(OUTPUT_DIR / "final_diabetes_flag.csv")
    cont_mean.to_csv(OUTPUT_DIR / "final_cluster_cont_mean.csv")
    gender_dist.join(age_mean).to_csv(OUTPUT_DIR / "final_cluster_demographic.csv")

    print(f"\n[저장 완료] → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
