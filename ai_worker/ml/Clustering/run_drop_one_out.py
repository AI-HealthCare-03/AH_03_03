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


def run_clustering(df, cont_cols, cat_cols):
    df_use = df[cont_cols + cat_cols].copy()
    cat_indices = [df_use.columns.get_loc(c) for c in cat_cols]
    scaler = StandardScaler()
    df_use[cont_cols] = scaler.fit_transform(df_use[cont_cols])
    X = df_use.values
    X_cont = df_use[cont_cols].values

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
    return round(sil, 4)


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

    df = df.sample(n=SAMPLE_N, random_state=SEED).reset_index(drop=True)
    print(f"  샘플: {len(df):,}명")

    # 베이스라인 (전체 변수)
    print(f"\n[베이스라인] 전체 {len(CONT_COLS_FINAL)}개 연속형 + {len(CAT_COLS)}개 범주형")
    baseline_sil = run_clustering(df, CONT_COLS_FINAL, CAT_COLS)
    print(f"  Silhouette: {baseline_sil}")

    # drop-one-out
    print(f"\n[Drop-one-out] 변수 1개씩 제거 후 Silhouette 측정")
    print("=" * 60)

    results = []
    all_vars = [(c, "연속형") for c in CONT_COLS_FINAL] + [(c, "범주형") for c in CAT_COLS]

    for var, var_type in all_vars:
        if var_type == "연속형":
            curr_cont = [c for c in CONT_COLS_FINAL if c != var]
            curr_cat = CAT_COLS
        else:
            curr_cont = CONT_COLS_FINAL
            curr_cat = [c for c in CAT_COLS if c != var]

        sil = run_clustering(df, curr_cont, curr_cat)
        diff = round(sil - baseline_sil, 4)
        results.append({
            "변수": var,
            "타입": var_type,
            "제거후_실루엣": sil,
            "변화량": diff,
            "중요도": "노이즈 가능" if diff > 0 else "중요",
        })
        print(f"  [{var_type}] {var:20s} | Silhouette: {sil} | 변화: {diff:+.4f}")

    results_df = pd.DataFrame(results).sort_values("변화량", ascending=False)

    print(f"\n[결과 요약] 베이스라인 Silhouette: {baseline_sil}")
    print("=" * 60)
    print("▶ 제거해도 되는 변수 (변화량 > 0, 오히려 올라감):")
    noise = results_df[results_df["변화량"] > 0]
    if len(noise) > 0:
        print(noise[["변수", "타입", "제거후_실루엣", "변화량"]].to_string(index=False))
    else:
        print("  없음")

    print("\n▶ 중요 변수 TOP 5 (제거 시 가장 많이 떨어짐):")
    print(results_df.tail(5)[["변수", "타입", "제거후_실루엣", "변화량"]].to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "drop_one_out_results.csv", index=False)
    print(f"\n[저장 완료] → {OUTPUT_DIR / 'drop_one_out_results.csv'}")


if __name__ == "__main__":
    main()
