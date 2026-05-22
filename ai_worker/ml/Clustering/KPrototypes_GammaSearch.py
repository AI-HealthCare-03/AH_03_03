"""
건강검진 데이터 군집화 — K-Prototypes gamma 탐색 (K=4 고정)
Python 3.13 | kmodes>=0.12 | scikit-learn>=1.4

목표: 정상 / 고혈압 / 당뇨 / 이상지질혈증 4개 군집
전략: K=4 고정 + gamma 탐색 (0.01~1.0, step 0.01)
비만: 군집화 제외 → 사후 분석에서 확인
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────
DATA_PATH = str(Path(__file__).parent.parent.parent / "data" / "lipid_only.csv")
OUTPUT_DIR = str(Path(__file__).parent / "outputs" / "gamma_search")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
SEED = 42
K = 4  # 정상 / 고혈압 / 당뇨 / 이상지질혈증
SAMPLE_N = 50000  # 테스트용 샘플 수

# gamma 탐색 범위 (1단계: 굵게 / 2단계: 좁혀서 세밀하게)
GAMMA_COARSE = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]

# ── 변수 정의 ─────────────────────────────────────────────────
CONT_COLS = [
    "신장(5cm단위)",
    "체중(5kg단위)",
    "허리둘레",
    "수축기혈압",
    "이완기혈압",
    "식전혈당(공복혈당)",
    "총콜레스테롤",
    "트리글리세라이드",
    "HDL콜레스테롤",
    "LDL콜레스테롤",
    "혈색소",
    "혈청크레아티닌",
    "혈청지오티(AST)",
    "혈청지피티(ALT)",
    "감마지티피",
]

CAT_COLS = [
    "성별코드",
    "연령대코드(5세단위)",
    "흡연상태",
    "음주여부",
    "요단백",
]

CLINICAL_BOUNDS = {
    "수축기혈압": (50, 300),
    "이완기혈압": (20, 200),
    "식전혈당(공복혈당)": (30, 1000),
    "총콜레스테롤": (50, 1000),
    "트리글리세라이드": (10, 6000),
    "HDL콜레스테롤": (10, 200),
    "LDL콜레스테롤": (10, 1000),
    "혈색소": (3, 25),
    "혈청크레아티닌": (0.1, 50),
    "혈청지오티(AST)": (5, 5000),
    "혈청지피티(ALT)": (5, 5000),
    "감마지티피": (1, 5000),
    "허리둘레": (40, 200),
}


def add_clinical_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["고혈압_기준"] = ((df["수축기혈압"] >= 140) | (df["이완기혈압"] >= 90)).astype(int)
    df["당뇨_기준"] = (df["식전혈당(공복혈당)"] >= 126).astype(int)
    df["이상지질혈증_기준"] = (df["총콜레스테롤"] >= 240).astype(int)
    df["BMI"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
    df["비만_기준"] = (df["BMI"] >= 25).astype(int)
    return df


def remove_outliers_clinical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, (lower, upper) in CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        outlier_cnt = ((df[col] < lower) | (df[col] > upper)).sum()
        df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
        if outlier_cnt > 0:
            print(f"  {col}: {outlier_cnt}개 제거 (허용범위: {lower}~{upper})")
    return df


def analyze_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """군집별 임상 기준 유병률 분석"""
    df = df.copy()
    df["cluster"] = labels
    clinical_cols = ["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]
    summary = df.groupby("cluster")[clinical_cols].mean().round(3) * 100
    summary.insert(0, "샘플수", df["cluster"].value_counts().sort_index())
    return summary


def run_gamma_search(
    X: np.ndarray,
    X_cont: np.ndarray,
    cat_indices: list,
    df: pd.DataFrame,
    gammas: list,
    label: str = "1단계",
) -> pd.DataFrame:
    """gamma 탐색 실행"""
    print(f"\n[gamma 탐색] {label} | K={K} | gamma 범위: {gammas}")
    print("=" * 70)

    results = []
    for gamma in gammas:
        kp = KPrototypes(
            n_clusters=K,
            init="Huang",
            n_init=3,
            gamma=gamma,
            random_state=SEED,
            verbose=0,
        )
        labels = kp.fit_predict(X, categorical=cat_indices)
        sil = silhouette_score(X_cont, labels, sample_size=10000, random_state=SEED)

        # 군집별 임상 유병률
        summary = analyze_clusters(df, labels)
        htn_std = summary["고혈압_기준"].std()
        dm_std = summary["당뇨_기준"].std()
        dl_std = summary["이상지질혈증_기준"].std()
        diversity = round(float(htn_std + dm_std + dl_std), 4)

        results.append(
            {
                "gamma": gamma,
                "silhouette": round(sil, 4),
                "diversity": diversity,  # 군집간 임상 차이 (높을수록 잘 분리됨)
                "cost": round(kp.cost_, 0),
            }
        )

        print(f"  gamma={gamma:.3f} | Silhouette: {sil:.4f} | Diversity: {diversity:.4f} | Cost: {kp.cost_:,.0f}")

    return pd.DataFrame(results)


def main() -> None:
    # ── 데이터 로드 및 전처리 ─────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"[0] 데이터 로드 | shape: {df.shape}")

    drop_cols = [
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
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print("\n[1] 임상 기준 이상치 제거")
    df = remove_outliers_clinical(df)

    df["BMI"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
    CONT_COLS_FINAL = CONT_COLS + ["BMI"]

    for c in CONT_COLS_FINAL:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])

    df = add_clinical_labels(df)

    # # 샘플링 (전체 돌릴땐 주석)
    # df = df.sample(n=SAMPLE_N, random_state=SEED).reset_index(drop=True)
    # print(f"\n[2] 샘플링 완료 | {SAMPLE_N:,}명")

    # 스케일링
    scaler = StandardScaler()
    X_cont = scaler.fit_transform(df[CONT_COLS_FINAL])
    X_cat = df[CAT_COLS].values.astype(str)
    X = np.hstack([X_cont, X_cat])
    cat_indices = list(range(len(CONT_COLS_FINAL), len(CONT_COLS_FINAL) + len(CAT_COLS)))

    # ── 1단계: 굵은 탐색 ─────────────────────────────────────
    results_coarse = run_gamma_search(X, X_cont, cat_indices, df, GAMMA_COARSE, "1단계 (굵은 탐색)")
    results_coarse.to_csv(os.path.join(OUTPUT_DIR, "gamma_search_coarse.csv"), index=False)

    # 최적 gamma 구간 찾기 (diversity 최대)
    best_row = results_coarse.loc[results_coarse["diversity"].idxmax()]
    best_gamma_coarse = best_row["gamma"]
    print(f"\n  → 1단계 최적 gamma: {best_gamma_coarse} (diversity: {best_row['diversity']:.4f})")

    # ── 2단계: 세밀한 탐색 ───────────────────────────────────
    # best_gamma 주변 ±0.1 구간을 0.01 단위로 탐색
    fine_start = max(0.01, round(best_gamma_coarse - 0.1, 2))
    fine_end = round(best_gamma_coarse + 0.1, 2)
    GAMMA_FINE = [round(x, 2) for x in np.arange(fine_start, fine_end + 0.01, 0.01)]

    results_fine = run_gamma_search(X, X_cont, cat_indices, df, GAMMA_FINE, "2단계 (세밀한 탐색)")
    results_fine.to_csv(os.path.join(OUTPUT_DIR, "gamma_search_fine.csv"), index=False)

    # ── 최적 gamma로 최종 군집화 ──────────────────────────────
    best_fine = results_fine.loc[results_fine["diversity"].idxmax()]
    best_gamma_final = best_fine["gamma"]
    print(f"\n[최종] 최적 gamma: {best_gamma_final} (diversity: {best_fine['diversity']:.4f})")

    kp_final = KPrototypes(
        n_clusters=K,
        init="Huang",
        n_init=5,
        gamma=best_gamma_final,
        random_state=SEED,
        verbose=0,
    )
    df["cluster"] = kp_final.fit_predict(X, categorical=cat_indices)

    # ── 군집별 분석 ───────────────────────────────────────────
    print(f"\n[군집 분석] K={K} | gamma={best_gamma_final}")
    print("=" * 70)

    summary = analyze_clusters(df, df["cluster"].values)
    print(summary.T.to_string())

    print("\n[성별 분포]")
    print(df.groupby("cluster")["성별코드"].value_counts(normalize=True).unstack().round(3).to_string())

    print("\n[연령대 분포]")
    print(df.groupby("cluster")["연령대코드(5세단위)"].mean().round(1).to_string())

    # ── 저장 ─────────────────────────────────────────────────
    summary.to_csv(os.path.join(OUTPUT_DIR, "final_cluster_summary.csv"))
    df[
        ["cluster"] + CONT_COLS_FINAL + CAT_COLS + ["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]
    ].to_csv(os.path.join(OUTPUT_DIR, "final_clustered_data.csv"), index=False)

    print(f"\n[저장 완료] → {OUTPUT_DIR}")
    print("  gamma_search_coarse.csv / gamma_search_fine.csv / final_cluster_summary.csv")


if __name__ == "__main__":
    main()
