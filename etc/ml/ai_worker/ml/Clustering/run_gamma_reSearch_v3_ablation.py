from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(__file__).parent.parent.parent  # ai_worker/
DATA_PATH = ROOT / "data/lipid_only.csv"
OUTPUT_BASE = Path(__file__).parent / "outputs"

SAMPLE_N = 50000
CLUSTER_SEED = 42
SAMPLE_SEED = 42

# ── 실험 설계 ────────────────────────────────────────────────────────────────
# A: 키+몸무게+BMI / 로그X / K=4 / StandardScaler  (기존 베이스라인)
# B: BMI만         / 로그X / K=4 / StandardScaler
# C: BMI만         / 로그O / K=4 / StandardScaler
# D: 키+몸무게+BMI / 로그O / K=4 / StandardScaler
# E: 키+몸무게+BMI / 로그X / K=5 / StandardScaler
# F: 키+몸무게+BMI / 로그X / K=4 / RobustScaler
# G: 키+몸무게+BMI / 로그X / K=5 / RobustScaler
# H: 키+몸무게+BMI / 로그X / K=6 / StandardScaler  ← 현재 실험
# I: 키+몸무게+BMI / 로그X / K=6 / RobustScaler    ← 현재 실험
# ─────────────────────────────────────────────────────────────────────────────

CONT_COLS_BMI_ONLY = [
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
    "BMI",
]

CONT_COLS_WITH_HW = [
    "신장(5cm단위)",
    "체중(5kg단위)",
] + CONT_COLS_BMI_ONLY

LOG_COLS = [
    "혈청지오티(AST)",
    "혈청지피티(ALT)",
    "감마지티피",
    "혈청크레아티닌",
]

EXPERIMENTS = [
    {"tag": "H_k6_standard", "cont_cols": CONT_COLS_WITH_HW, "use_log": False, "k": 6, "scaler": "standard"},
    {"tag": "I_k6_robust",   "cont_cols": CONT_COLS_WITH_HW, "use_log": False, "k": 6, "scaler": "robust"},
]

CAT_COLS = ["성별코드", "연령대코드(5세단위)", "흡연상태", "음주여부", "요단백"]

GAMMA_COARSE = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]

CLINICAL_BOUNDS = {
    "수축기혈압":         (60,  250),
    "이완기혈압":         (30,  150),
    "식전혈당(공복혈당)": (50,  500),
    "총콜레스테롤":       (50,  500),
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
    "치아우식증유무", "결손치 유무", "치아마모증유무",
    "제3대구치(사랑니) 이상", "치석", "기준년도",
    "가입자일련번호", "시도코드", "구강검진수검여부",
    "시력(좌)", "시력(우)", "청력(좌)", "청력(우)",
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


def run_clustering(X, X_cont, cat_indices, gamma, k, n_init=3):
    kp = KPrototypes(
        n_clusters=k,
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

    # cont_mean은 원본 df 기준 (스케일 변환 전 값) — 해석 용이
    cont_mean = df.groupby("cluster")[cont_cols].mean().round(2)

    gender = df.groupby("cluster")["성별코드"].value_counts(normalize=True).unstack().round(3)
    gender.columns = ["남성비율", "여성비율"]
    age = df.groupby("cluster")["연령대코드(5세단위)"].mean().round(1).rename("평균연령대코드")

    return summary, cont_mean, gender, age


def run_experiment(df_sampled, tag, cont_cols, use_log, k, scaler_type, output_dir):
    print(f"\n{'=' * 70}")
    print(f"[실험 {tag}] 변수 {len(cont_cols)}개 | K={k} | log1p={'ON' if use_log else 'OFF'} | scaler={scaler_type}")
    print(f"{'=' * 70}")

    df = df_sampled.copy()
    df = add_clinical_labels(df)

    df_use = df[cont_cols + CAT_COLS].copy()
    cat_indices = [df_use.columns.get_loc(c) for c in CAT_COLS]

    if use_log:
        log_targets = [c for c in LOG_COLS if c in df_use.columns]
        df_use[log_targets] = np.log1p(df_use[log_targets])
        print(f"  log1p 적용: {log_targets}")

    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    df_use[cont_cols] = scaler.fit_transform(df_use[cont_cols])
    X = df_use.values
    X_cont = df_use[cont_cols].values

    # gamma 탐색 — n_init=3
    print(f"\n[gamma 탐색] K={k}")
    results = []
    for gamma in GAMMA_COARSE:
        labels, sil, min_ratio, cost = run_clustering(X, X_cont, cat_indices, gamma, k, n_init=3)
        counts = np.bincount(labels)
        results.append({"gamma": gamma, "silhouette": sil, "min_cluster_ratio": min_ratio, "cost": cost})
        print(f"  gamma={gamma:.3f} | Sil: {sil:.4f} | 최소군집비율: {min_ratio:.3f} | Cost: {cost:,.0f} | 군집크기: {sorted(counts, reverse=True)}")

    results_df = pd.DataFrame(results)
    balanced = results_df[results_df["min_cluster_ratio"] > 0.1]
    best = balanced.loc[balanced["silhouette"].idxmax()] if len(balanced) > 0 else results_df.loc[results_df["silhouette"].idxmax()]
    best_gamma = best["gamma"]
    print(f"\n→ 최적 gamma: {best_gamma} (Sil: {best['silhouette']:.4f} | 최소군집비율: {best['min_cluster_ratio']:.3f})")

    # 최종 군집화 — n_init=20
    print(f"\n[최종 군집화] gamma={best_gamma}")
    labels, sil, min_ratio, cost = run_clustering(X, X_cont, cat_indices, best_gamma, k, n_init=20)
    print(f"  Silhouette: {sil:.4f} | 최소군집비율: {min_ratio:.3f} | 군집크기: {sorted(np.bincount(labels), reverse=True)}")

    summary, cont_mean, gender, age = detailed_cluster_analysis(df, labels, cont_cols)

    print(f"\n[군집별 임상 유병률]")
    print(summary.to_string())
    print(f"\n[군집별 연속형 변수 평균 (원본 스케일)]")
    print(cont_mean.to_string())
    print(f"\n[성별 분포]")
    print(gender.to_string())
    print(f"\n[연령대 분포]")
    print(age.to_string())

    exp_dir = output_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(exp_dir / "gamma_search_results.csv", index=False)
    summary.to_csv(exp_dir / "cluster_summary.csv")
    cont_mean.to_csv(exp_dir / "cluster_cont_mean.csv")
    gender.join(age).to_csv(exp_dir / "cluster_demographic.csv")
    df["cluster"] = labels
    df.to_csv(exp_dir / "clustered_data.csv", index=False)
    print(f"\n[저장 완료] → {exp_dir}")

    return {
        "tag": tag,
        "k": k,
        "n_cont": len(cont_cols),
        "use_log": use_log,
        "scaler": scaler_type,
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
    df = df.dropna(subset=CONT_COLS_WITH_HW + CAT_COLS)
    print(f"  전처리 후 전체: {len(df):,}명")

    df_sampled = df.sample(n=SAMPLE_N, random_state=SAMPLE_SEED).reset_index(drop=True)
    print(f"  샘플: {len(df_sampled):,}명 (seed={SAMPLE_SEED})")

    output_dir = OUTPUT_BASE / "ablation"
    summary_rows = []

    for exp in EXPERIMENTS:
        row = run_experiment(
            df_sampled,
            tag=exp["tag"],
            cont_cols=exp["cont_cols"],
            use_log=exp["use_log"],
            k=exp["k"],
            scaler_type=exp["scaler"],
            output_dir=output_dir,
        )
        summary_rows.append(row)

    print(f"\n{'=' * 70}")
    print("[실험 결과 요약]")
    print(f"{'=' * 70}")
    print(f"  {'실험':<14} {'K':<4} {'변수수':<8} {'log':<6} {'scaler':<12} {'best_gamma':<12} {'Silhouette':<12} {'최소군집비율':<14} 군집크기")
    for r in summary_rows:
        print(
            f"  {r['tag']:<14} {r['k']:<4} {r['n_cont']:<8} {'ON' if r['use_log'] else 'OFF':<6} "
            f"{r['scaler']:<12} {r['best_gamma']:<12} {r['silhouette']:.4f}{'':6} "
            f"{r['min_ratio']:.3f}{'':10} {r['cluster_sizes']}"
        )

    compare_df = pd.DataFrame(summary_rows)
    compare_path = output_dir / "ablation_summary.csv"
    compare_df.to_csv(compare_path, index=False)
    print(f"\n[비교 요약 저장] → {compare_path}")


if __name__ == "__main__":
    main()
