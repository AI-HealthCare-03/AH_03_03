from pathlib import Path
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "ai_worker/data/lipid_only.csv"
OUTPUT_DIR = ROOT / "ai_worker/ml/Clustering/outputs/gamma_search"

K = 4
BEST_GAMMA = 1.02
SEED = 42

CONT_COLS = [
    "신장(5cm단위)", "체중(5kg단위)", "허리둘레",
    "수축기혈압", "이완기혈압", "식전혈당(공복혈당)",
    "총콜레스테롤", "트리글리세라이드", "HDL콜레스테롤", "LDL콜레스테롤",
    "혈색소", "혈청크레아티닌", "혈청지오티(AST)", "혈청지피티(ALT)", "감마지티피",
]
CAT_COLS = ["성별코드", "연령대코드(5세단위)", "흡연상태", "음주여부", "요단백"]

CLINICAL_BOUNDS = {
    "수축기혈압": (50, 300), "이완기혈압": (20, 200),
    "식전혈당(공복혈당)": (30, 1000), "총콜레스테롤": (50, 1000),
    "트리글리세라이드": (10, 6000), "HDL콜레스테롤": (10, 200),
    "LDL콜레스테롤": (10, 1000), "혈색소": (3, 25),
    "혈청크레아티닌": (0.1, 50), "혈청지오티(AST)": (5, 5000),
    "혈청지피티(ALT)": (5, 5000), "감마지티피": (1, 5000),
    "허리둘레": (40, 200),
}

DROP_COLS = [
    "치아우식증유무", "결손치 유무", "치아마모증유무", "제3대구치(사랑니) 이상",
    "치석", "기준년도", "가입자일련번호", "시도코드", "구강검진수검여부",
    "시력(좌)", "시력(우)", "청력(좌)", "청력(우)",
]

def remove_outliers_clinical(df):
    df = df.copy()
    for col, (lower, upper) in CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        outlier_cnt = ((df[col] < lower) | (df[col] > upper)).sum()
        df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
        if outlier_cnt > 0:
            print(f"  {col}: {outlier_cnt}개 제거 (허용범위: {lower}~{upper})")
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
    print("[0] 데이터 로드")
    df = pd.read_csv(DATA_PATH)
    print(f"  shape: {df.shape}")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    print(f"[0] 불필요 컬럼 제거 후 | shape: {df.shape}")

    print("\n[1] 임상 기준 이상치 제거")
    df = remove_outliers_clinical(df)

    df["BMI"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
    CONT_COLS_FINAL = CONT_COLS + ["BMI"]
    print(f"\n[2] BMI 추가 | 연속형: {len(CONT_COLS_FINAL)}개 | 범주형: {len(CAT_COLS)}개")

    for c in CONT_COLS_FINAL:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0])
    print(f"[2] 결측치 처리 완료 | 잔여: {df[CONT_COLS_FINAL + CAT_COLS].isnull().sum().sum()}")

    df = add_clinical_labels(df)

    df_use = df[CONT_COLS_FINAL + CAT_COLS].copy()
    cat_indices = [df_use.columns.get_loc(c) for c in CAT_COLS]
    scaler = StandardScaler()
    df_use[CONT_COLS_FINAL] = scaler.fit_transform(df_use[CONT_COLS_FINAL])
    X = df_use.values

    print(f"\n[최종 군집화] K={K} | gamma={BEST_GAMMA} | 전체 {len(df):,}명")
    kp = KPrototypes(
        n_clusters=K, init="Huang", n_init=5,
        gamma=BEST_GAMMA, random_state=SEED, verbose=1,
    )
    df["cluster"] = kp.fit_predict(X, categorical=cat_indices)

    summary = analyze_clusters(df, df["cluster"].values)
    print(f"\n[군집 분석]")
    print(summary.to_string())
    print("\n[성별 분포]")
    print(df.groupby("cluster")["성별코드"].value_counts(normalize=True).unstack().round(3).to_string())
    print("\n[연령대 분포]")
    print(df.groupby("cluster")["연령대코드(5세단위)"].mean().round(1).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "final_cluster_summary.csv")
    df[["cluster"] + CONT_COLS_FINAL + CAT_COLS + ["고혈압_기준", "당뇨_기준", "이상지질혈증_기준", "비만_기준"]].to_csv(
        OUTPUT_DIR / "final_clustered_data.csv", index=False
    )
    print(f"\n[저장 완료] → {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
