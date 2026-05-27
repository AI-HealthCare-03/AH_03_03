from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent  # ai_worker/
DATA_PATH = ROOT / "data/lipid_only.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "selected_vars"

SAMPLE_N = 50000
SEEDS = [42, 123, 777]  # 샘플링 seed만 바꿈
CLUSTER_SEED = 42        # 군집화 seed는 항상 고정
K = 4

CONT_COLS = [
    "신장(5cm단위)",
    "체중(5kg단위)",
    "허리둘레",
    "수축기혈압",
    "이완기혈압",
    "식전혈당(공복혈당)",
    "총콜레스테롤",
    "HDL콜레스테롤",
    "LDL콜레스테롤",
    "혈색소",
    "혈청크레아티닌",
    "혈청지오티(AST)",
    "혈청지피티(ALT)",
    "감마지티피",
]
CAT_COLS = ["성별코드", "연령대코드(5세단위)", "흡연상태", "음주여부", "요단백"]

GAMMA_COARSE = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]

CLINICAL_BOUNDS = {
    "수축기혈압":         (60,  250),
    "이완기혈압":         (30,  150),
    "식전혈당(공복혈당)": (50,  500),
    "총콜레스테롤":       (50,  500),
    "트리글리세라이드":   (10, 1000),
    "HDL콜레스테롤":      (10,  150),
    "LDL콜레스테롤":      (10,  400),
    "혈색소":             (5,    22),
    "혈청크레아티닌":     (0.3,  15),
    "혈청지오티(AST)":    (5,   200),
    "혈청지피티(ALT)":    (5,   200),
    "감마지티피":         (1,   300),
    "허리둘레":           (40,  160),
}

DROP_COLS = [
    "치아우식증유무",
    "결손치 유무",
    "치아마모증유무",
    "제3대구치(사랑니) 이상",
    "치석",
    "기준년도",
    "가입자일련번호",
    "시도코드",
    "구강검진수검여부",
    "시력(좌)",
    "시력(우)",
    "청력(좌)",
    "청력(우)",
]


def remove_outliers_clinical(df):
    df = df.copy()
    for col, (lower, upper) in CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
    return df


def add_clinical_labels(df):
    df["고혈압_기준"] = ((df["수축기혈압"] >= 140) | (df["이완기혈압"] >= 90)).astype(int)
    df["당뇨_기준"] = (df["식전혈당(공복혈당)"] >= 126).astype(int)
    df["이상지질혈증_기준"] = (df["총콜레스테롤"] >= 240).astype(int)
    df["BMI"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
    df["비만_기준"] = (df["BMI"] >= 25).astype(int)
    return df


def run_clustering(X, X_cont, cat_indices, gamma, n_init=3):
    # 군집화 seed는 CLUSTER_SEED로 항상 고정 — 샘플링만 다를 때 군집화 조건 동일하게 유지
    kp = KPrototypes(
        n_clusters=K,
        init="Huang",
        n_init=n_init,
        gamma=gamma,
        random_state=CLUSTER_SEED,
        verbose=0,
    )
    labels = kp.fit_predict(X, categorical=cat_indices)
    sil = silhouette_score(X_cont, labels, sample_size=10000, random_state=CLUSTER_SEED)
    counts = np.bincount(labels)
    min_ratio = counts.min() / counts.max()
    return labels, round(sil, 4), round(min_ratio, 3), round(kp.cost_, 0)


def detailed_cluster_analysis(df, labels, cont_cols):
    df = df.copy()
    df["cluster"] = labels
    clinical_cols = ["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]

    summary = df.groupby("cluster")[clinical_cols].mean().round(3) * 100
    summary.insert(0, "샘플수", df["cluster"].value_counts().sort_index())
    summary.insert(1, "비율(%)", (summary["샘플수"] / len(df) * 100).round(1))

    cont_mean = df.groupby("cluster")[cont_cols].mean().round(2)

    gender = df.groupby("cluster")["성별코드"].value_counts(normalize=True).unstack().round(3)
    gender.columns = ["남성비율", "여성비율"]
    age = df.groupby("cluster")["연령대코드(5세단위)"].mean().round(1).rename("평균연령대코드")

    return summary, cont_mean, gender, age


def run_one_seed(df_base, sample_seed):
    """샘플링 seed만 바꿔서 돌림 — 군집화 조건은 동일"""
    print(f"\n{'=' * 70}")
    print(f"[샘플링 SEED={sample_seed}]")
    print(f"{'=' * 70}")

    df = df_base.sample(n=SAMPLE_N, random_state=sample_seed).reset_index(drop=True)
    CONT_COLS_FINAL = CONT_COLS + ["BMI"]
    df = add_clinical_labels(df)

    df_use = df[CONT_COLS_FINAL + CAT_COLS].copy()
    cat_indices = [df_use.columns.get_loc(c) for c in CAT_COLS]
    scaler = StandardScaler()
    df_use[CONT_COLS_FINAL] = scaler.fit_transform(df_use[CONT_COLS_FINAL])
    X = df_use.values
    X_cont = df_use[CONT_COLS_FINAL].values

    # gamma 탐색 — n_init=3으로 빠르게
    print(f"\n[gamma 탐색] K={K} | gamma 범위: {GAMMA_COARSE}")
    results = []
    for gamma in GAMMA_COARSE:
        labels, sil, min_ratio, cost = run_clustering(X, X_cont, cat_indices, gamma, n_init=3)
        counts = np.bincount(labels)
        results.append({"gamma": gamma, "silhouette": sil, "min_cluster_ratio": min_ratio, "cost": cost})
        print(f"  gamma={gamma:.3f} | Silhouette: {sil:.4f} | 최소군집비율: {min_ratio:.3f} | Cost: {cost:,.0f} | 군집크기: {sorted(counts, reverse=True)}")

    results_df = pd.DataFrame(results)

    balanced = results_df[results_df["min_cluster_ratio"] > 0.1]
    best = balanced.loc[balanced["silhouette"].idxmax()] if len(balanced) > 0 else results_df.loc[results_df["silhouette"].idxmax()]
    best_gamma = best["gamma"]
    print(f"\n→ 최적 gamma: {best_gamma} (Silhouette: {best['silhouette']:.4f} | 최소군집비율: {best['min_cluster_ratio']:.3f})")

    # 최종 군집화 — n_init=20으로 정밀하게
    print(f"\n[최종 군집화] gamma={best_gamma}")
    labels, sil, min_ratio, cost = run_clustering(X, X_cont, cat_indices, best_gamma, n_init=20)
    print(f"  Silhouette: {sil:.4f}")
    print(f"  최소군집비율: {min_ratio:.3f}")
    print(f"  군집 크기: {sorted(np.bincount(labels), reverse=True)}")

    summary, cont_mean, gender, age = detailed_cluster_analysis(df, labels, CONT_COLS_FINAL)

    print(f"\n[군집별 임상 유병률]")
    print(summary.to_string())
    print(f"\n[군집별 연속형 변수 평균]")
    print(cont_mean.to_string())
    print(f"\n[성별 분포]")
    print(gender.to_string())
    print(f"\n[연령대 분포]")
    print(age.to_string())

    # 저장
    seed_dir = OUTPUT_DIR / f"seed_{sample_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(seed_dir / "gamma_search_results.csv", index=False)
    summary.to_csv(seed_dir / "cluster_summary.csv")
    cont_mean.to_csv(seed_dir / "cluster_cont_mean.csv")
    gender.join(age).to_csv(seed_dir / "cluster_demographic.csv")
    df["cluster"] = labels
    df.to_csv(seed_dir / "clustered_data.csv", index=False)
    print(f"\n[저장 완료] → {seed_dir}")

    return {
        "sample_seed": sample_seed,
        "best_gamma": best_gamma,
        "silhouette": sil,
        "min_ratio": min_ratio,
        "cluster_sizes": sorted(np.bincount(labels), reverse=True),
    }


def main():
    print("[0] 데이터 로드 및 전처리 (공통)")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = remove_outliers_clinical(df)
    df["BMI"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
    CONT_COLS_FINAL = CONT_COLS + ["BMI"]
    df = df.dropna(subset=CONT_COLS_FINAL + CAT_COLS)
    print(f"  outlier 제거 후 전체: {len(df):,}명")
    print(f"  연속형: {len(CONT_COLS_FINAL)}개 | 범주형: {len(CAT_COLS)}개")
    print(f"  군집화 고정 seed: {CLUSTER_SEED} | 샘플링 seeds: {SEEDS}")

    summary_rows = []
    for sample_seed in SEEDS:
        row = run_one_seed(df, sample_seed)
        summary_rows.append(row)

    print(f"\n{'=' * 70}")
    print("[전체 샘플링 seed 요약]")
    print(f"{'=' * 70}")
    for row in summary_rows:
        print(f"  sample_seed={row['sample_seed']} | best_gamma={row['best_gamma']} | "
              f"Silhouette={row['silhouette']:.4f} | 최소군집비율={row['min_ratio']:.3f} | "
              f"군집크기={row['cluster_sizes']}")


if __name__ == "__main__":
    main()
