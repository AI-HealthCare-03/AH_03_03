"""
run_gender_split_clustering.py

성별 분리 K-Prototypes 군집화
- 남성 / 여성 각각 독립 모델 학습
- K=4, StandardScaler, 이완기혈압 제거(13개), 크레아티닌 ≤5
- 샘플: 각 성별 25,000명 (테스트) → 전체 실행 시 주석 처리
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
OUTPUT_DIR = Path(__file__).parent / "outputs" / "gender_split_v2_derived"

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────
K = 4
BEST_GAMMA_SEARCH = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]
N_INIT_SEARCH = 3
N_INIT_FINAL = 20
CLUSTER_SEED = 42
SAMPLE_SEED = 42
SAMPLE_N = 50000  # 성별당 샘플 수 (테스트용)

# ────────────────────────────────────────────────
# 변수 구성 (이완기혈압 제거, 13개)
# 성별코드는 분리 기준으로 사용 → 범주형에서 제거
# ────────────────────────────────────────────────
CONT_COLS = [
    "신장(5cm단위)",
    "체중(5kg단위)",
    "허리둘레",
    "수축기혈압",
    # 이완기혈압 제거 (수축기혈압과 r=0.68)
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
    # 트리글리세라이드는 TG_HDL비율 계산 후 제거 — CONT_COLS에 미포함
    "TG_HDL비율",       # 파생변수: 인슐린저항성 대리지표
    "비HDL콜레스테롤",   # 파생변수: 이상지질혈증 분류 지표
    "맥압",              # 파생변수: 수축기-이완기 (동맥경직도)
    "AST_ALT비율",       # 파생변수: 간경변 의심 지표
    "심혈관위험지수",     # 파생변수: 총콜레스테롤/HDL
]
CAT_COLS = ["연령대코드(5세단위)", "흡연상태", "음주여부", "요단백"]  # 성별코드 제거

# 대사 지표 선택적 로그 변환 대상 (코치님 권장)
LOG_COLS_SELECTIVE = [
    "식전혈당(공복혈당)",
    "트리글리세라이드",
    "혈청지오티(AST)",
    "혈청지피티(ALT)",
    "감마지티피",
    "혈청크레아티닌",
    "TG_HDL비율",
]

# ────────────────────────────────────────────────
# 임상 기준
# ────────────────────────────────────────────────
CLINICAL_BOUNDS = {
    "수축기혈압":         (60,  250),
    "이완기혈압":         (30,  150),
    "식전혈당(공복혈당)": (50,  500),
    "총콜레스테롤":       (50,  500),
    "트리글리세라이드":   (10,  500),
    "HDL콜레스테롤":      (10,  150),
    "LDL콜레스테롤":      (10,  400),
    "혈색소":             (5,    22),
    "혈청크레아티닌":     (0.3,   5),
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


def add_derived_features(df):
    """파생변수 생성 — TG/HDL 비율, 비HDL 콜레스테롤"""
    df = df.copy()
    df["TG_HDL비율"] = df["트리글리세라이드"] / df["HDL콜레스테롤"].replace(0, np.nan)
    df["비HDL콜레스테롤"] = df["총콜레스테롤"] - df["HDL콜레스테롤"]
    df["맥압"] = df["수축기혈압"] - df["이완기혈압"]
    df["AST_ALT비율"] = df["혈청지오티(AST)"] / df["혈청지피티(ALT)"].replace(0, np.nan)
    df["심혈관위험지수"] = df["총콜레스테롤"] / df["HDL콜레스테롤"].replace(0, np.nan)
    return df


def add_clinical_labels(df):
    df = df.copy()
    df["고혈압_기준"]       = ((df["수축기혈압"] >= 140) | (df["이완기혈압"] >= 90)).astype(int)
    df["당뇨_기준"]         = (df["식전혈당(공복혈당)"] >= 126).astype(int)
    df["이상지질혈증_기준"] = (df["총콜레스테롤"] >= 240).astype(int)
    df["비만_기준"]         = (df["BMI"] >= 25).astype(int)
    return df


# ────────────────────────────────────────────────
# gamma 탐색 + 최종 군집화
# ────────────────────────────────────────────────
def apply_selective_log(df, cont_cols):
    """대사 지표만 선택적 로그 변환"""
    df = df.copy()
    for col in LOG_COLS_SELECTIVE:
        if col in cont_cols and col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def run_clustering(X, X_cont, cat_indices, k, gamma, n_init):
    kp = KPrototypes(
        n_clusters=k,
        init="Huang",
        n_init=n_init,
        gamma=gamma,
        random_state=CLUSTER_SEED,
        verbose=0,
    )
    labels = kp.fit_predict(X, categorical=cat_indices)
    sil = silhouette_score(X_cont, labels, sample_size=5000, random_state=CLUSTER_SEED)
    counts = np.bincount(labels)
    min_ratio = counts.min() / len(labels)
    return labels, round(sil, 4), round(min_ratio, 3), round(kp.cost_, 0)


def calc_compare_silhouette(X_cont_orig, labels):
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X_cont_orig)
    cmp_sil = silhouette_score(X_std, labels, sample_size=5000, random_state=CLUSTER_SEED)
    return round(cmp_sil, 4)


# ────────────────────────────────────────────────
# 단일 성별 실험
# ────────────────────────────────────────────────
def run_one_gender(df_gender, gender_label):
    k_threshold = 0.04  # 코치님 권장: 0.03~0.05

    print(f"\n{'=' * 70}")
    print(f"[{gender_label}] N={len(df_gender):,} | K={K} | k_threshold={k_threshold:.3f}")
    print(f"{'=' * 70}")

    df = add_derived_features(df_gender)
    df = add_clinical_labels(df)
    df_orig = df.copy()

    # 대사 지표 선택적 로그 변환
    df = apply_selective_log(df, CONT_COLS)

    df_use = df[CONT_COLS + CAT_COLS].copy()
    cat_indices = [df_use.columns.get_loc(c) for c in CAT_COLS]
    scaler = StandardScaler()
    df_use[CONT_COLS] = scaler.fit_transform(df_use[CONT_COLS])
    X = df_use.values
    X_cont_scaled = df_use[CONT_COLS].values
    X_cont_orig = df_orig[CONT_COLS].values

    # gamma 탐색
    print(f"\n[gamma 탐색]")
    search_results = []
    for gamma in BEST_GAMMA_SEARCH:
        labels, train_sil, min_ratio, cost = run_clustering(
            X, X_cont_scaled, cat_indices, K, gamma, N_INIT_SEARCH
        )
        counts = np.bincount(labels)
        search_results.append({
            "gamma": gamma,
            "train_silhouette": train_sil,
            "min_cluster_ratio": min_ratio,
            "cost": cost,
        })
        print(f"  gamma={gamma:.3f} | train_sil={train_sil:.4f} | 최소군집비율={min_ratio:.3f} | "
              f"군집크기={sorted(counts, reverse=True)}")

    results_df = pd.DataFrame(search_results)
    balanced = results_df[results_df["min_cluster_ratio"] > k_threshold]
    if len(balanced) > 0:
        best_row = balanced.loc[balanced["train_silhouette"].idxmax()]
    else:
        best_row = results_df.loc[results_df["train_silhouette"].idxmax()]
    best_gamma = best_row["gamma"]
    print(f"\n→ 최적 gamma: {best_gamma} "
          f"(train_sil={best_row['train_silhouette']:.4f} | 최소군집비율={best_row['min_cluster_ratio']:.3f})")

    # 최종 군집화
    print(f"\n[최종 군집화] gamma={best_gamma} | n_init={N_INIT_FINAL}")
    labels, train_sil, min_ratio, cost = run_clustering(
        X, X_cont_scaled, cat_indices, K, best_gamma, N_INIT_FINAL
    )
    counts = np.bincount(labels)
    cmp_sil = calc_compare_silhouette(X_cont_orig, labels)

    print(f"  train_sil (학습용): {train_sil:.4f}")
    print(f"  cmp_sil   (비교용): {cmp_sil:.4f}")
    print(f"  최소군집비율: {min_ratio:.3f}")
    print(f"  군집 크기:    {sorted(counts, reverse=True)}")

    # 분석
    df_orig["cluster"] = labels
    df_orig["당뇨_고위험"]  = (df_orig["식전혈당(공복혈당)"] >= 126).astype(int)
    df_orig["당뇨_전단계"]  = ((df_orig["식전혈당(공복혈당)"] >= 100) & (df_orig["식전혈당(공복혈당)"] < 126)).astype(int)
    df_orig["고혈압_고위험"] = ((df_orig["수축기혈압"] >= 140) | (df_orig["이완기혈압"] >= 90)).astype(int)
    df_orig["고혈압_전단계"] = ((df_orig["수축기혈압"] >= 130) | (df_orig["이완기혈압"] >= 80)).astype(int)

    clinical_cols = ["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]
    cluster_count = df_orig["cluster"].value_counts().sort_index()
    summary = df_orig.groupby("cluster")[clinical_cols].mean().round(3) * 100
    summary.insert(0, "샘플수", cluster_count)
    summary.insert(1, "비율(%)", (cluster_count / len(df_orig) * 100).round(1))

    flag_cols = ["당뇨_고위험", "당뇨_전단계", "고혈압_고위험", "고혈압_전단계"]
    flag_summary = df_orig.groupby("cluster")[flag_cols].mean().round(3) * 100

    cont_mean = df_orig.groupby("cluster")[CONT_COLS].mean().round(2)
    age_mean = df_orig.groupby("cluster")["연령대코드(5세단위)"].mean().round(1).rename("평균연령대코드")

    print(f"\n[군집별 임상 유병률]")
    print(summary.to_string())
    print(f"\n[후처리 플래그 (당뇨/고혈압)]")
    print(flag_summary.to_string())
    print(f"\n[군집별 연속형 변수 평균]")
    print(cont_mean.to_string())
    print(f"\n[연령대 분포]")
    print(age_mean.to_string())

    # 저장
    out_dir = OUTPUT_DIR / gender_label
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "gamma_search_results.csv", index=False)
    summary.to_csv(out_dir / "cluster_summary.csv")
    flag_summary.to_csv(out_dir / "cluster_flag_summary.csv")
    cont_mean.to_csv(out_dir / "cluster_cont_mean.csv")
    age_mean.to_csv(out_dir / "cluster_demographic.csv")
    df_orig.to_csv(out_dir / "clustered_data.csv", index=False)
    print(f"\n[저장 완료] → {out_dir}")

    return {
        "gender":     gender_label,
        "n":          len(df_orig),
        "best_gamma": best_gamma,
        "train_sil":  train_sil,
        "cmp_sil":    cmp_sil,
        "min_ratio":  min_ratio,
        "cluster_sizes": sorted(counts, reverse=True),
    }


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def main():
    print("[0] 데이터 로드 및 전처리")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = remove_outliers_clinical(df)
    df = add_bmi(df)
    df = add_derived_features(df)
    df = df.dropna(subset=CONT_COLS + CAT_COLS + ["성별코드", "이완기혈압"])
    print(f"  전처리 후 전체: {len(df):,}명")
    df_male   = df[df["성별코드"] == 1].reset_index(drop=True)
    df_female = df[df["성별코드"] == 2].reset_index(drop=True)
    print(f"  남성: {len(df_male):,}명 | 여성: {len(df_female):,}명")

    # 테스트용 샘플링 — 전체 실행 시 아래 4줄 주석 처리
    df_male   = df_male.sample(n=SAMPLE_N, random_state=SAMPLE_SEED).reset_index(drop=True)
    df_female = df_female.sample(n=SAMPLE_N, random_state=SAMPLE_SEED).reset_index(drop=True)
    print(f"  샘플링 후 — 남성: {len(df_male):,}명 | 여성: {len(df_female):,}명 (seed={SAMPLE_SEED})")

    results = []
    for df_g, label in [(df_male, "male"), (df_female, "female")]:
        row = run_one_gender(df_g, label)
        results.append(row)

    print(f"\n{'=' * 70}")
    print("[전체 요약]")
    print(f"{'=' * 70}")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_DIR / "gender_split_summary.csv", index=False)
    print(f"\n[요약 저장] → {OUTPUT_DIR / 'gender_split_summary.csv'}")


if __name__ == "__main__":
    main()
