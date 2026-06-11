"""
run_gamma_reSearch_v3_ablation.py

K-Prototypes ablation 실험 스크립트
- 실험 조합(tag, k, scaler, use_log, cont_cols, cat_cols)만 EXPERIMENTS에 추가하면 자동 실행
- gamma coarse search (n_init=3) → 최적 gamma 선택 → 최종 군집화 (n_init=20)
- cont_mean은 항상 원본 스케일 기준 (로그/스케일 변환 전 df 기준)
- 람다 함수 사용 금지

[Silhouette 분리 정책]
- 학습용(train_sil): 실험 스케일러 공간에서 계산 → gamma 탐색 및 최종 군집화에 사용
- 비교용(cmp_sil):  StandardScaler 공간에서 재계산 → ablation 실험 간 공정 비교용
  (Robust vs Standard 스케일러 차이 제거)

[K-aware threshold]
- gamma 선택 시 최소군집비율 기준: 1 / (2 * k)
  K=4 → 0.125 / K=6 → 0.083 / K=7 → 0.071
"""

from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

# ────────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent  # ai_worker/
DATA_PATH = ROOT / "data/lipid_only.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "ablation"

# ────────────────────────────────────────────────
# 고정 설정
# ────────────────────────────────────────────────
SAMPLE_N = 50000
SAMPLE_SEED = 42
CLUSTER_SEED = 42
N_INIT_SEARCH = 3
N_INIT_FINAL = 20

GAMMA_COARSE = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0]

# ────────────────────────────────────────────────
# 변수 구성
# ────────────────────────────────────────────────
CONT_COLS_BASE = [
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

# CONT_COLS_WITH_HW  = 키+몸무게+BMI 포함 (15개)
# CONT_COLS_BMI_ONLY = 키/몸무게 제거, BMI만 (13개)
CONT_COLS_WITH_HW = CONT_COLS_BASE + ["BMI"]
CONT_COLS_BMI_ONLY = [c for c in CONT_COLS_BASE if c not in ["신장(5cm단위)", "체중(5kg단위)"]] + ["BMI"]

CAT_COLS = ["성별코드", "연령대코드(5세단위)", "흡연상태", "음주여부", "요단백"]

# 이완기혈압 제거 (수축기혈압과 r=0.68, 중복 신호)
CONT_COLS_NO_DBP = [c for c in CONT_COLS_WITH_HW if c != "이완기혈압"]  # 14개
CAT_COLS_NO_GENDER = ["연령대코드(5세단위)", "흡연상태", "음주여부", "요단백"]

# N2 베이스 + 파생변수 전체 포함 (19개)
# 트리글리세라이드: TG/HDL 계산용으로만 사용, CONT_COLS에 미포함
CONT_COLS_WITH_DERIVED = [
    c
    for c in CONT_COLS_WITH_HW
    if c != "이완기혈압"  # 이완기 제거
] + ["TG_HDL비율", "비HDL콜레스테롤", "맥압", "AST_ALT비율", "심혈관위험지수"]

# 로그 변환 대상 변수 (대사 지표만 선택적 적용)
LOG_COLS_SELECTIVE = [
    "식전혈당(공복혈당)",
    "트리글리세라이드",
    "혈청지오티(AST)",
    "혈청지피티(ALT)",
    "감마지티피",
    "혈청크레아티닌",
    "TG_HDL비율",
    # 맥압, AST_ALT비율, 심혈관위험지수는 로그 변환 불필요
]

# ────────────────────────────────────────────────
# 임상 기준
# ────────────────────────────────────────────────
CLINICAL_BOUNDS = {
    "수축기혈압": (60, 250),
    "이완기혈압": (30, 150),
    "식전혈당(공복혈당)": (50, 500),
    "총콜레스테롤": (50, 500),
    "트리글리세라이드": (10, 500),
    "HDL콜레스테롤": (10, 150),
    "LDL콜레스테롤": (10, 400),
    "혈색소": (5, 22),
    "혈청크레아티닌": (0.3, 5),  # 5 초과 = 말기 신부전 수준, 검진 데이터 신뢰도 낮음
    "혈청지오티(AST)": (5, 200),
    "혈청지피티(ALT)": (5, 200),
    "감마지티피": (1, 300),
    "허리둘레": (40, 160),
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

# ────────────────────────────────────────────────
# 실험 목록
# cat_cols: "default"(성별 포함) | "no_gender"(성별 제거)
# ────────────────────────────────────────────────
EXPERIMENTS = [
    # 기존 실험 (A~K) — 재실행 필요 없으면 주석 처리
    # {"tag": "A_base",        "cont_cols": CONT_COLS_WITH_HW,  "use_log": False, "k": 4, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "B_bmi",         "cont_cols": CONT_COLS_BMI_ONLY, "use_log": False, "k": 4, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "C_bmi_log",     "cont_cols": CONT_COLS_BMI_ONLY, "use_log": True,  "k": 4, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "D_base_log",    "cont_cols": CONT_COLS_WITH_HW,  "use_log": True,  "k": 4, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "E_k5",          "cont_cols": CONT_COLS_WITH_HW,  "use_log": False, "k": 5, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "F_robust",      "cont_cols": CONT_COLS_WITH_HW,  "use_log": False, "k": 4, "scaler": "robust",   "cat_cols": "default"},
    # {"tag": "G_robust_k5",   "cont_cols": CONT_COLS_WITH_HW,  "use_log": False, "k": 5, "scaler": "robust",   "cat_cols": "default"},
    # {"tag": "H_k6_standard", "cont_cols": CONT_COLS_WITH_HW,  "use_log": False, "k": 6, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "I_k6_robust",   "cont_cols": CONT_COLS_WITH_HW,  "use_log": False, "k": 6, "scaler": "robust",   "cat_cols": "default"},
    # {"tag": "J_k7_standard", "cont_cols": CONT_COLS_WITH_HW,  "use_log": False, "k": 7, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "K_k7_robust",   "cont_cols": CONT_COLS_WITH_HW,  "use_log": False, "k": 7, "scaler": "robust",   "cat_cols": "default"},
    # 신규 실험
    # {"tag": "L_no_gender", "cont_cols": CONT_COLS_WITH_HW, "use_log": False, "k": 6, "scaler": "standard", "cat_cols": "no_gender"},
    # {"tag": "M_k6_standard", "cont_cols": CONT_COLS_WITH_HW, "use_log": False, "k": 6, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "N_no_dbp", "cont_cols": CONT_COLS_NO_DBP, "use_log": False, "k": 6, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "O_k6_robust_cr5", "cont_cols": CONT_COLS_WITH_HW, "use_log": False, "k": 6, "scaler": "robust", "cat_cols": "default"},
    # {"tag": "P_k7_standard_no_dbp_cr5", "cont_cols": CONT_COLS_NO_DBP, "use_log": False, "k": 7, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "Q_k6_robust_no_dbp_cr5", "cont_cols": CONT_COLS_NO_DBP, "use_log": False, "k": 6, "scaler": "robust", "cat_cols": "default"},
    # 신규 — 코치님 피드백 반영
    # selective_log: 대사 지표만 선택적 로그 변환
    # 파생변수: TG/HDL비율, 비HDL콜레스테롤, 트리글리세라이드 포함
    # {"tag": "R_selective_log_derived", "cont_cols": CONT_COLS_WITH_DERIVED, "use_log": False, "selective_log": True, "k": 6, "scaler": "standard", "cat_cols": "default"},
    # {"tag": "S_full_derived_log", "cont_cols": CONT_COLS_WITH_DERIVED, "use_log": False, "selective_log": True, "k": 6, "scaler": "standard", "cat_cols": "default"},
    {
        "tag": "T_k7_full_derived_log",
        "cont_cols": CONT_COLS_WITH_DERIVED,
        "use_log": False,
        "selective_log": True,
        "k": 7,
        "scaler": "standard",
        "cat_cols": "default",
    },
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
    """파생변수 생성"""
    df = df.copy()
    df["TG_HDL비율"] = df["트리글리세라이드"] / df["HDL콜레스테롤"].replace(0, np.nan)
    df["비HDL콜레스테롤"] = df["총콜레스테롤"] - df["HDL콜레스테롤"]
    df["맥압"] = df["수축기혈압"] - df["이완기혈압"]
    df["AST_ALT비율"] = df["혈청지오티(AST)"] / df["혈청지피티(ALT)"].replace(0, np.nan)
    df["심혈관위험지수"] = df["총콜레스테롤"] / df["HDL콜레스테롤"].replace(0, np.nan)
    return df


def add_clinical_labels(df):
    df = df.copy()
    df["고혈압_기준"] = ((df["수축기혈압"] >= 140) | (df["이완기혈압"] >= 90)).astype(int)
    df["당뇨_기준"] = (df["식전혈당(공복혈당)"] >= 126).astype(int)
    df["이상지질혈증_기준"] = (df["총콜레스테롤"] >= 240).astype(int)
    df["비만_기준"] = (df["BMI"] >= 25).astype(int)
    return df


def apply_log_transform(df, cont_cols, selective=False):
    """
    selective=False: use_log=True 실험 — cont_cols 전체 로그 적용 (기존 방식)
    selective=True:  대사 지표만 선택적 로그 적용 (코치님 권장)
    원본 df는 건드리지 않음.
    """
    df = df.copy()
    target_cols = (
        LOG_COLS_SELECTIVE
        if selective
        else [
            "혈청지오티(AST)",
            "혈청지피티(ALT)",
            "감마지티피",
            "혈청크레아티닌",
            "식전혈당(공복혈당)",
        ]
    )
    for col in target_cols:
        if col in cont_cols and col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def get_scaler(scaler_name):
    if scaler_name == "robust":
        return RobustScaler()
    return StandardScaler()


# ────────────────────────────────────────────────
# Silhouette 재계산 (비교용 — StandardScaler 공간 고정)
# ────────────────────────────────────────────────
def calc_compare_silhouette(X_cont_original_scale, labels):
    """
    ablation 실험 간 공정 비교를 위해 StandardScaler 공간에서 Silhouette 재계산.
    X_cont_original_scale: 로그/스케일 변환 전 원본 스케일 연속형 배열
    labels: 군집 레이블
    """
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X_cont_original_scale)
    cmp_sil = silhouette_score(X_std, labels, sample_size=10000, random_state=CLUSTER_SEED)
    return round(cmp_sil, 4)


# ────────────────────────────────────────────────
# 군집화 함수
# ────────────────────────────────────────────────
def run_clustering(X, X_cont, cat_indices, k, gamma, n_init):
    """
    X_cont: 실험 스케일러 적용된 연속형 배열 (학습용 Silhouette 계산에 사용)
    반환 sil = 학습용 Silhouette (gamma 탐색 및 최종 군집화 내부 기준)
    """
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
    min_ratio = counts.min() / len(labels)
    return labels, round(sil, 4), round(min_ratio, 3), round(kp.cost_, 0)


# ────────────────────────────────────────────────
# 분석 함수
# ────────────────────────────────────────────────
def detailed_cluster_analysis(df_orig, labels, cont_cols):
    """
    df_orig: 원본 스케일 df (로그/스케일 변환 전)
    성별코드 없을 때 gender_dist는 빈 DataFrame 반환
    """
    df = df_orig.copy()
    df["cluster"] = labels
    clinical_cols = ["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]

    cluster_count = df["cluster"].value_counts().sort_index()
    cluster_ratio = (cluster_count / len(df) * 100).round(1)

    summary = df.groupby("cluster")[clinical_cols].mean().round(3) * 100
    summary.insert(0, "샘플수", cluster_count)
    summary.insert(1, "비율(%)", cluster_ratio)

    cont_mean = df.groupby("cluster")[cont_cols].mean().round(2)

    if "성별코드" in df.columns:
        gender_dist = df.groupby("cluster")["성별코드"].value_counts(normalize=True).unstack().fillna(0).round(3)
        gender_dist.columns = [f"성별_{int(c)}" for c in gender_dist.columns]
    else:
        gender_dist = pd.DataFrame(index=cluster_count.index)

    age_mean = df.groupby("cluster")["연령대코드(5세단위)"].mean().round(1).rename("평균연령대코드")

    return summary, cont_mean, gender_dist, age_mean


# ────────────────────────────────────────────────
# 단일 실험 실행
# ────────────────────────────────────────────────
def run_experiment(df_sampled, exp):
    tag = exp["tag"]
    cont_cols = exp["cont_cols"]
    use_log = exp["use_log"]
    k = exp["k"]
    scaler_nm = exp["scaler"]
    cat_cols = CAT_COLS_NO_GENDER if exp.get("cat_cols") == "no_gender" else CAT_COLS

    # K-aware threshold: 균형 기준을 K에 맞게 자동 조정
    k_threshold = 1 / (2 * k)

    print(f"\n{'=' * 70}")
    print(
        f"[실험] {tag} | K={k} | scaler={scaler_nm} | log={use_log} | "
        f"연속형={len(cont_cols)}개 | 범주형={len(cat_cols)}개 | k_threshold={k_threshold:.3f}"
    )
    print(f"{'=' * 70}")

    df = add_derived_features(df_sampled)
    df = add_clinical_labels(df)
    df_orig = df.copy()  # 원본 스케일 보존

    selective_log = exp.get("selective_log", False)
    if use_log or selective_log:
        df = apply_log_transform(df, cont_cols, selective=selective_log)

    # 실험 스케일러 적용 (학습용)
    df_use = df[cont_cols + cat_cols].copy()
    cat_indices = [df_use.columns.get_loc(c) for c in cat_cols]
    scaler = get_scaler(scaler_nm)
    df_use[cont_cols] = scaler.fit_transform(df_use[cont_cols])
    X = df_use.values
    X_cont_scaled = df_use[cont_cols].values  # 학습용 Silhouette에 사용

    # 비교용 StandardScaler 공간 준비 (원본 스케일 기준)
    X_cont_orig = df_orig[cont_cols].values

    # gamma coarse search
    print(f"\n[gamma 탐색] gamma 범위: {GAMMA_COARSE}")
    search_results = []
    for gamma in GAMMA_COARSE:
        labels, train_sil, min_ratio, cost = run_clustering(X, X_cont_scaled, cat_indices, k, gamma, N_INIT_SEARCH)
        counts = np.bincount(labels)
        search_results.append(
            {
                "gamma": gamma,
                "train_silhouette": train_sil,
                "min_cluster_ratio": min_ratio,
                "cost": cost,
            }
        )
        print(
            f"  gamma={gamma:.3f} | train_sil={train_sil:.4f} | 최소군집비율={min_ratio:.3f} | "
            f"Cost={cost:,.0f} | 군집크기={sorted(counts, reverse=True)}"
        )

    results_df = pd.DataFrame(search_results)

    # 최적 gamma: K-aware threshold 통과 중 train_silhouette 최대
    balanced = results_df[results_df["min_cluster_ratio"] > k_threshold]
    if len(balanced) > 0:
        best_row = balanced.loc[balanced["train_silhouette"].idxmax()]
    else:
        best_row = results_df.loc[results_df["train_silhouette"].idxmax()]
    best_gamma = best_row["gamma"]
    print(
        f"\n→ 최적 gamma: {best_gamma} "
        f"(train_sil={best_row['train_silhouette']:.4f} | 최소군집비율={best_row['min_cluster_ratio']:.3f})"
    )

    # 최종 군집화
    print(f"\n[최종 군집화] gamma={best_gamma} | n_init={N_INIT_FINAL}")
    labels, train_sil, min_ratio, cost = run_clustering(X, X_cont_scaled, cat_indices, k, best_gamma, N_INIT_FINAL)
    counts = np.bincount(labels)

    # 비교용 Silhouette 재계산 (StandardScaler 공간)
    cmp_sil = calc_compare_silhouette(X_cont_orig, labels)

    print(f"  train_sil (학습용):  {train_sil:.4f}  ← 실험 스케일러({scaler_nm}) 공간")
    print(f"  cmp_sil   (비교용):  {cmp_sil:.4f}  ← StandardScaler 공간 (ablation 비교 기준)")
    print(f"  최소군집비율: {min_ratio:.3f}")
    print(f"  군집 크기:    {sorted(counts, reverse=True)}")

    summary, cont_mean, gender_dist, age_mean = detailed_cluster_analysis(df_orig, labels, cont_cols)

    print(f"\n[군집별 임상 유병률]")
    print(summary.to_string())
    print(f"\n[군집별 연속형 변수 평균 (원본 스케일)]")
    print(cont_mean.to_string())
    if not gender_dist.empty:
        print(f"\n[성별 분포]")
        print(gender_dist.to_string())
    else:
        print(f"\n[성별 분포] — 성별코드 범주형 제외됨 (혈색소/신장 참고)")
    print(f"\n[연령대 분포]")
    print(age_mean.to_string())

    # 저장
    exp_dir = OUTPUT_DIR / tag
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(exp_dir / "gamma_search_results.csv", index=False)
    summary.to_csv(exp_dir / "cluster_summary.csv")
    cont_mean.to_csv(exp_dir / "cluster_cont_mean.csv")
    gender_dist.join(age_mean).to_csv(exp_dir / "cluster_demographic.csv")
    df_orig["cluster"] = labels
    df_orig.to_csv(exp_dir / "clustered_data.csv", index=False)
    print(f"\n[저장 완료] → {exp_dir}")

    return {
        "tag": tag,
        "k": k,
        "scaler": scaler_nm,
        "use_log": use_log,
        "cat_cols": exp.get("cat_cols", "default"),
        "best_gamma": best_gamma,
        "train_sil": train_sil,
        "cmp_sil": cmp_sil,
        "min_ratio": min_ratio,
        "cluster_sizes": sorted(counts, reverse=True),
    }


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────
def main():
    print("[0] 데이터 로드 및 공통 전처리")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = remove_outliers_clinical(df)
    df = add_bmi(df)

    all_cont = list(set(CONT_COLS_WITH_HW + CONT_COLS_BMI_ONLY + ["트리글리세라이드", "이완기혈압"]))
    df = df.dropna(subset=all_cont + CAT_COLS)
    print(f"  outlier 제거 + dropna 후: {len(df):,}명")

    df = add_derived_features(df)
    df_sampled = df.sample(n=SAMPLE_N, random_state=SAMPLE_SEED).reset_index(drop=True)
    print(f"  샘플 크기: {len(df_sampled):,}명 (seed={SAMPLE_SEED})")
    print(f"  실험 수: {len(EXPERIMENTS)}개")

    all_results = []
    for exp in EXPERIMENTS:
        row = run_experiment(df_sampled, exp)
        all_results.append(row)

    print(f"\n{'=' * 70}")
    print("[전체 실험 요약]")
    print(f"{'=' * 70}")
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)
    print(f"\n[요약 저장] → {OUTPUT_DIR / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()
