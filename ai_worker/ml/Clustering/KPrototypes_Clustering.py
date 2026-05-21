"""
건강검진 데이터 군집화 — K-Prototypes (혼합형)
Python 3.13 | kmodes>=0.12 | scikit-learn>=1.4

연속형: 혈압/혈당/콜레스테롤/간수치/신장/체중/BMI 등
범주형: 성별/연령대/흡연/음주/요단백
전처리: 임상 기준 이상치 제거 → StandardScaler → K-Prototypes
검증: Elbow (cost) + Silhouette
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────
DATA_PATH = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/data/lipid_only.csv"
OUTPUT_DIR = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/ml/Clustering/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────
SEED = 42
K_RANGE = range(2, 8)

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

# 임상 기준 상하한 (국제 학회 가이드라인 기반)
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
    """임상 진단 기준으로 사후 검증용 레이블 생성"""
    df["고혈압_기준"] = ((df["수축기혈압"] >= 140) | (df["이완기혈압"] >= 90)).astype(int)
    df["당뇨_기준"] = (df["식전혈당(공복혈당)"] >= 126).astype(int)
    df["이상지질혈증_기준"] = (df["총콜레스테롤"] >= 240).astype(int)
    df["BMI"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
    df["비만_기준"] = (df["BMI"] >= 25).astype(int)
    return df


def remove_outliers_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """임상적으로 불가능한 값만 제거 (국제 학회 가이드라인 기반)"""
    df = df.copy()
    for col, (lower, upper) in CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        outlier_cnt = ((df[col] < lower) | (df[col] > upper)).sum()
        df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
        if outlier_cnt > 0:
            print(f"  {col}: {outlier_cnt}개 제거 (허용범위: {lower}~{upper})")
    return df


def main() -> None:
    # ── 데이터 로드 ───────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"[0] 데이터 로드 | shape: {df.shape}")

    # 불필요 컬럼 제거
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
    print(f"[0] 불필요 컬럼 제거 후 | shape: {df.shape}")

    # ── 임상 기준 이상치 제거 ─────────────────────────────────
    print("\n[1] 임상 기준 이상치 제거 (국제 학회 가이드라인 기반)")
    df = remove_outliers_clinical(df)

    # ── BMI 추가 ──────────────────────────────────────────────
    df["BMI"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
    CONT_COLS_FINAL = CONT_COLS + ["BMI"]
    print(f"\n[2] BMI 추가 | 연속형: {len(CONT_COLS_FINAL)}개 | 범주형: {len(CAT_COLS)}개")

    # ── 결측치 처리 ───────────────────────────────────────────
    for c in CONT_COLS_FINAL:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])

    print(f"[2] 결측치 처리 완료 | 잔여 결측: {df[CONT_COLS_FINAL + CAT_COLS].isnull().sum().sum()}")

    # ── 임상 레이블 추가 (사후 검증용) ───────────────────────
    df = add_clinical_labels(df)
    print("\n[3] 임상 기준 레이블 추가")
    print(f"    고혈압_기준:       {df['고혈압_기준'].sum():,}명 ({df['고혈압_기준'].mean():.1%})")
    print(f"    당뇨_기준:         {df['당뇨_기준'].sum():,}명 ({df['당뇨_기준'].mean():.1%})")
    print(f"    이상지질혈증_기준: {df['이상지질혈증_기준'].sum():,}명 ({df['이상지질혈증_기준'].mean():.1%})")
    print(f"    비만_기준:         {df['비만_기준'].sum():,}명 ({df['비만_기준'].mean():.1%})")

    # 테스트용 샘플링 (전체 돌릴 때는 주석 처리)
    df = df.sample(n=50000, random_state=42).reset_index(drop=True)

    # ── 스케일링 + K-Prototypes 입력 구성 ────────────────────
    scaler = StandardScaler()
    X_cont = scaler.fit_transform(df[CONT_COLS_FINAL])
    X_cat = df[CAT_COLS].values.astype(str)
    X = np.hstack([X_cont, X_cat])
    cat_indices = list(range(len(CONT_COLS_FINAL), len(CONT_COLS_FINAL) + len(CAT_COLS)))

    print(f"\n[4] 전처리 완료 | 총 샘플: {len(df):,}명")

    # ── K 탐색 (Elbow + Silhouette) ───────────────────────────
    print(f"\n[5] K 탐색 | 범위: {list(K_RANGE)}")
    print("=" * 60)

    costs = []
    silhouettes = []

    for k in K_RANGE:
        kp = KPrototypes(n_clusters=k, init="Huang", n_init=3, random_state=SEED, verbose=0)
        labels = kp.fit_predict(X, categorical=cat_indices)
        costs.append(kp.cost_)
        sil = silhouette_score(X_cont, labels, sample_size=10000, random_state=SEED)
        silhouettes.append(sil)
        print(f"  K={k} | Cost: {kp.cost_:,.0f} | Silhouette: {sil:.4f}")

    best_k = K_RANGE.start + int(np.argmax(silhouettes))
    print(f"\n  → 최적 K: {best_k} (Silhouette 최대: {max(silhouettes):.4f})")

    # ── 최종 군집화 ───────────────────────────────────────────
    print(f"\n[6] 최종 군집화 | K={best_k}")
    print("=" * 60)

    kp_final = KPrototypes(n_clusters=best_k, init="Huang", n_init=5, random_state=SEED, verbose=0)
    df["cluster"] = kp_final.fit_predict(X, categorical=cat_indices)

    # ── 군집별 특성 분석 ──────────────────────────────────────
    print("\n[7] 군집별 특성 분석")
    print("=" * 60)

    analysis_cols = CONT_COLS_FINAL + ["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]
    cluster_summary = df.groupby("cluster")[analysis_cols].mean().round(2)
    cluster_size = df["cluster"].value_counts().sort_index()
    cluster_summary.insert(0, "샘플수", cluster_size)
    cluster_summary.insert(1, "비율(%)", (cluster_size / len(df) * 100).round(1))

    print(cluster_summary.T.to_string())

    print("\n[8] 군집별 범주형 분포")
    print("-" * 60)
    for c in ["성별코드", "흡연상태", "음주여부"]:
        print(f"\n  [{c}]")
        print(df.groupby("cluster")[c].value_counts(normalize=True).unstack().round(3).to_string())

    # ── 군집 시각화 (PCA 2D) ──────────────────────────────────

    print("\n[시각화] PCA 2D 군집 분포")
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_cont)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 군집별 색상
    colors = plt.cm.Set1(np.linspace(0, 1, best_k))

    # 군집 분포
    for k in range(best_k):
        mask = df["cluster"] == k
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[k]], label=f"군집 {k}", alpha=0.3, s=5)
    axes[0].set_title(f"군집 분포 (PCA 2D) | K={best_k}")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].legend(markerscale=3)

    # 임상 기준 유병률 히트맵
    clinical_rate = df.groupby("cluster")[["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]].mean() * 100

    import seaborn as sns

    sns.heatmap(clinical_rate, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title("군집별 임상 기준 유병률 (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_visualization.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("시각화 저장 완료")

    # ── 저장 ─────────────────────────────────────────────────
    cluster_summary.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv"))
    df[
        ["cluster"] + CONT_COLS_FINAL + CAT_COLS + ["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]
    ].to_csv(os.path.join(OUTPUT_DIR, "clustered_data.csv"), index=False)
    pd.DataFrame({"k": list(K_RANGE), "cost": costs, "silhouette": silhouettes}).to_csv(
        os.path.join(OUTPUT_DIR, "k_search.csv"), index=False
    )

    print(f"\n[9] 저장 완료 → {OUTPUT_DIR}")
    print("    cluster_summary.csv / clustered_data.csv / k_search.csv")


if __name__ == "__main__":
    main()
