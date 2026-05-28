from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent  # ai_worker/
DATA_PATH = ROOT / "data/lipid_only.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"

SAMPLE_N = 50000
SEED = 42
K = 4
GAMMA = 1.02

# ── 제거 변수: 신장, 수축기혈압, 이완기혈압, 트리글리세라이드 ──
CONT_COLS = [
    "체중(5kg단위)",
    "허리둘레",
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


def analyze_clusters(df, labels):
    df = df.copy()
    df["cluster"] = labels
    clinical_cols = ["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]
    summary = df.groupby("cluster")[clinical_cols].mean().round(3) * 100
    summary.insert(0, "샘플수", df["cluster"].value_counts().sort_index())
    return summary


def main():
    print("[0] 데이터 로드 및 전처리")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
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
    df = df.sample(n=SAMPLE_N, random_state=SEED).reset_index(drop=True)
    print(f"  샘플: {len(df):,}명")
    print(f"  연속형: {len(CONT_COLS_FINAL)}개 | 범주형: {len(CAT_COLS)}개")
    print(f"  사용 변수: {CONT_COLS_FINAL + CAT_COLS}")

    df_use = df[CONT_COLS_FINAL + CAT_COLS].copy()
    cat_indices = [df_use.columns.get_loc(c) for c in CAT_COLS]
    scaler = StandardScaler()
    df_use[CONT_COLS_FINAL] = scaler.fit_transform(df_use[CONT_COLS_FINAL])
    X = df_use.values
    X_cont = df_use[CONT_COLS_FINAL].values

    print(f"\n[군집화] K={K} | gamma={GAMMA}")
    kp = KPrototypes(
        n_clusters=K,
        init="Huang",
        n_init=3,
        gamma=GAMMA,
        random_state=SEED,
        verbose=0,
    )
    labels = kp.fit_predict(X, categorical=cat_indices)
    sil = silhouette_score(X_cont, labels, sample_size=10000, random_state=SEED)
    print(f"  Silhouette: {sil:.4f}  (베이스라인: 0.1034)")

    summary = analyze_clusters(df, labels)
    print(f"\n[군집 분석]")
    print(summary.to_string())
    print("\n[성별 분포]")
    print(df.assign(cluster=labels).groupby("cluster")["성별코드"].value_counts(normalize=True).unstack().round(3).to_string())
    print("\n[연령대 분포]")
    print(df.assign(cluster=labels).groupby("cluster")["연령대코드(5세단위)"].mean().round(1).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "selected_vars_cluster_summary.csv")
    print(f"\n[저장 완료] → {OUTPUT_DIR / 'selected_vars_cluster_summary.csv'}")


if __name__ == "__main__":
    main()
