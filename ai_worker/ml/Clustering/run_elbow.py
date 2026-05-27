from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent  # ai_worker/
DATA_PATH = ROOT / "data/lipid_only.csv"
OUTPUT_DIR = Path(__file__).parent / "outputs"

SAMPLE_N = 50000
SEED = 42
K_RANGE = range(2, 8)  # K=2~7
GAMMA = 1.02  # 최적 gamma 고정

CONT_COLS = [
    "신장(5cm단위)",
    "체중(5kg단위)",
    "허리둘레",
    "수축기혈압",
    "이완기혈압",
    "식전혈당(공복혈당)",
    "총콜레스테롤",
    # "트리글리세라이드",
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


def main():
    print("[0] 데이터 로드 및 전처리")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = remove_outliers_clinical(df)
    df["BMI"] = df["체중(5kg단위)"] / ((df["신장(5cm단위)"] / 100) ** 2)
    CONT_COLS_FINAL = CONT_COLS + ["BMI"]

    # for c in CONT_COLS_FINAL:
    #     if c in df.columns:
    #         df[c] = df[c].fillna(df[c].median())
    # for c in CAT_COLS:
    #     if c in df.columns:
    #         df[c] = df[c].fillna(df[c].mode()[0])

    df = df.dropna(subset=CONT_COLS_FINAL + CAT_COLS)

    # 샘플링
    df = df.sample(n=SAMPLE_N, random_state=SEED).reset_index(drop=True)
    print(f"  샘플: {len(df):,}명")

    df_use = df[CONT_COLS_FINAL + CAT_COLS].copy()
    cat_indices = [df_use.columns.get_loc(c) for c in CAT_COLS]
    scaler = StandardScaler()
    df_use[CONT_COLS_FINAL] = scaler.fit_transform(df_use[CONT_COLS_FINAL])
    X = df_use.values

    print(f"\n[Elbow] gamma={GAMMA} | K={list(K_RANGE)}")
    print("=" * 50)

    results = []
    for k in K_RANGE:
        kp = KPrototypes(
            n_clusters=k,
            init="Huang",
            n_init=3,
            gamma=GAMMA,
            random_state=SEED,
            verbose=0,
        )
        kp.fit_predict(X, categorical=cat_indices)
        cost = kp.cost_
        results.append({"K": k, "cost": round(cost, 0)})
        print(f"  K={k} | Cost: {cost:,.0f}")

    results_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "elbow_results.csv", index=False)
    print(f"\n[저장 완료] → {OUTPUT_DIR / 'elbow_results.csv'}")
    print("\n[Elbow 결과 요약]")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
