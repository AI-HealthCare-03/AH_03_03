# ================================================================
# 건강검진 데이터 통합 변수 탐색기 (hn18~24)
# 목적: HTN / DM / DL 예측을 위한 추가 변수 후보 검토
# 실행환경: Python 3.9+  |  pyreadstat, pandas, numpy, matplotlib
# 사용법:
#   DATA_DIR 경로만 수정 후 실행
#   출력: 콘솔 요약 + plots/ 폴더에 png 저장
# ================================================================

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat

warnings.filterwarnings("ignore")

# ── 한글 폰트 (macOS 기준) ────────────────────────────────────
for font_name in ["AppleGothic", "NanumGothic", "Malgun Gothic"]:
    try:
        plt.rcParams["font.family"] = font_name
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

# ================================================================
# ★ 경로 설정 — 여기만 수정
# ================================================================
DATA_DIR = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/data"
PLOT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# 연도별 파일명 매핑
YEAR_FILES = {
    2018: "hn18_all.sas7bdat",
    2019: "hn19_all.sas7bdat",
    2020: "hn20_all.sas7bdat",
    2021: "hn21_all.sas7bdat",
    2022: "hn22_all.sas7bdat",
    2023: "hn23_all.sas7bdat",
    2024: "hn24_all.sas7bdat",
}

# ================================================================
# 타겟 변수 정의 — 다중분류 (수치 기반 단계 생성)
# ================================================================
# HTN: 0 정상 / 1 주의 / 2 고혈압전단계 / 3 1단계 / 4 2단계
# DM:  0 정상 / 1 공복혈당장애 / 2 당뇨
# DL:  0 정상 / 1 경계 / 2 위험 / 3 고위험
# OBE: 0 정상이하(1,2) / 1 비만전단계(3) / 2 1단계(4) / 3 2단계(5) / 4 3단계(6)

TARGET_MAP = {
    "HTN": "target_HTN",
    "DM": "target_DM",
    "DL": "target_DL",
    "OBE": "target_OBE",
}


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── HTN: sbp / dbp 기준 ──────────────────────────────────
    sbp = pd.to_numeric(df.get("HE_sbp"), errors="coerce")
    dbp = pd.to_numeric(df.get("HE_dbp"), errors="coerce")

    def classify_htn(row):
        s, d = row["_sbp"], row["_dbp"]
        if pd.isna(s) or pd.isna(d):
            return np.nan
        if s >= 160 or d >= 100:
            return 4  # 2단계 고혈압
        if s >= 140 or d >= 90:
            return 3  # 1단계 고혈압
        if s >= 130 or d >= 80:
            return 2  # 고혈압전단계
        if s >= 120:
            return 1  # 주의
        return 0  # 정상

    df["_sbp"], df["_dbp"] = sbp, dbp
    df["target_HTN"] = df[["_sbp", "_dbp"]].apply(classify_htn, axis=1)

    # ── DM: 공복혈당 / HbA1c 기준 (둘 중 높은 단계 적용) ────
    glu = pd.to_numeric(df.get("HE_glu"), errors="coerce")
    hba1c = pd.to_numeric(df.get("HE_HbA1c"), errors="coerce")

    def classify_dm(row):
        g, h = row["_glu"], row["_hba1c"]
        stages = []
        if not pd.isna(g):
            stages.append(2 if g >= 126 else 1 if g >= 100 else 0)
        if not pd.isna(h):
            stages.append(2 if h >= 6.5 else 1 if h >= 5.7 else 0)
        return max(stages) if stages else np.nan

    df["_glu"], df["_hba1c"] = glu, hba1c
    df["target_DM"] = df[["_glu", "_hba1c"]].apply(classify_dm, axis=1)

    # ── DL: LDL / 총콜레스테롤 / TG / HDL 기준 (최고 단계) ──
    df["_ldl"] = pd.to_numeric(df.get("HE_LDL_drct"), errors="coerce")
    df["_chol"] = pd.to_numeric(df.get("HE_chol"), errors="coerce")
    df["_tg"] = pd.to_numeric(df.get("HE_TG"), errors="coerce")
    df["_hdl"] = pd.to_numeric(df.get("HE_HDL_st2"), errors="coerce")

    def classify_dl(row):
        l, c, t, h = row["_ldl"], row["_chol"], row["_tg"], row["_hdl"]
        stages = []
        if not pd.isna(l):
            stages.append(3 if l >= 160 else 2 if l >= 130 else 1 if l >= 100 else 0)
        if not pd.isna(c):
            stages.append(3 if c >= 260 else 2 if c >= 240 else 1 if c >= 200 else 0)
        if not pd.isna(t):
            stages.append(3 if t >= 500 else 2 if t >= 200 else 1 if t >= 150 else 0)
        if not pd.isna(h):
            stages.append(2 if h < 40 else 1 if h < 60 else 0)
        return max(stages) if stages else np.nan

    df["target_DL"] = df[["_ldl", "_chol", "_tg", "_hdl"]].apply(classify_dl, axis=1)

    # ── OBE: HE_obe 원본 → 0~4 매핑 ─────────────────────────
    obe_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    df["target_OBE"] = pd.to_numeric(df.get("HE_obe"), errors="coerce").map(obe_map)

    # 임시 컬럼 제거
    tmp = ["_sbp", "_dbp", "_glu", "_hba1c", "_ldl", "_chol", "_tg", "_hdl"]
    df.drop(columns=[c for c in tmp if c in df.columns], inplace=True)

    return df


# ================================================================
# 기존 확정 변수 (이미 사용 중인 변수, 중복 제거용)
# ================================================================
EXISTING_FEATURES = {
    "공통": [
        "sex",
        "age",
        "HE_BMI",
        "HE_wc",
        "HE_ht",
        "HE_wt",
        "HE_sbp",
        "HE_dbp",
        "HE_glu",
        "HE_HbA1c",
        "HE_chol",
        "HE_HDL_st2",
        "HE_TG",
        "HE_LDL_drct",
        "HE_HPfh1",
        "HE_HPfh2",
        "HE_HPfh3",
        "HE_DMfh1",
        "HE_DMfh2",
        "HE_DMfh3",
        "HE_HLfh1",
        "HE_HLfh2",
        "HE_HLfh3",
        "BD1_11",  # 음주빈도
        "BD2_1",  # 음주량
        "BD2_31",  # 폭음빈도
        "BE3_31",  # 걷기일수
        "BE3_32",
        "BE3_33",  # 걷기시간
        "BS3_1",  # 흡연여부
        "incm",  # 소득분위
        "edu",  # 교육수준
    ]
}

# ================================================================
# 추가 후보 변수 그룹 (검토 대상)
# — 이 목록을 수정/확장해서 사용
# ================================================================
CANDIDATE_GROUPS = {
    # ── 검진 수치 ──────────────────────────────────────────────
    "검진_혈액": [
        "HE_ast",
        "HE_alt",  # 간수치
        "HE_BUN",
        "HE_crea",  # 신장기능
        "HE_WBC",
        "HE_RBC",  # 혈구수
        "HE_Bplt",  # 혈소판
        "HE_HB",
        "HE_HCT",  # 헤모글로빈/헤마토크리트
        "HE_Uacid",  # 요산
        "HE_hsCRP",  # 고감도 CRP
        "HE_Ca",
        "HE_P",
        "HE_Mg",  # 전해질
    ],
    "검진_소변": [
        "HE_Upro",  # 요단백
        "HE_Uglu",  # 요당
        "HE_Ualb",  # 요알부민
        "HE_Uph",  # 요산도
    ],
    "검진_폐기능": [
        "HE_fvc",
        "HE_fev1",
        "HE_fev1fvc",
    ],
    "검진_맥박_체성분": [
        "HE_mPLS",  # 맥박수
        "BIA_PBF",  # 체지방률
        "BIA_FFM",  # 제지방량
        "BIA_WBPA50",  # 위상각 (근육 건강 지표)
    ],
    # ── 생활습관 ───────────────────────────────────────────────
    "생활_수면": [
        "BP16_11",
        "BP16_13",  # 주중 취침/기상시각(시간)
        "BP16_21",
        "BP16_23",  # 주말 취침/기상시각(시간)
    ],
    "생활_신체활동_고강도": [
        "BE3_71",
        "BE3_72",  # 고강도(일) 여부/일수
        "BE3_75",
        "BE3_76",  # 고강도(여가) 여부/일수
    ],
    "생활_신체활동_중강도": [
        "BE3_81",
        "BE3_82",  # 중강도(일)
        "BE3_85",
        "BE3_86",  # 중강도(여가)
        "BE5_1",  # 근력운동 일수
        "BE8_1",  # 하루 앉아있는 시간(시간)
    ],
    "생활_식이": [
        "LS_VEG1",  # 채소 섭취빈도
        "LS_FRUIT",  # 과일 섭취빈도
        "L_BR_FQ",  # 아침식사 빈도
        "N_NA",  # 나트륨 섭취
        "N_TDF",  # 식이섬유
        "N_SUGAR",  # 당 섭취
        "N_CHOL",  # 콜레스테롤 섭취
        "N_FAT",  # 지방 섭취
        "N_SFA",  # 포화지방산
    ],
    "생활_정신건강": [
        "mh_PHQ_S",  # PHQ-9 우울 점수
        "mh_GAD_S",  # GAD-7 불안 점수
        "mh_stress",  # 스트레스 인지율
        "BP5",  # 2주이상 우울감
        "D_1_1",  # 주관적 건강인지
    ],
    "생활_수면_시간_계산용": [
        # 취침~기상으로 수면시간 직접 계산 가능
        "BP16_12",
        "BP16_14",  # 주중 취침/기상 분
        "BP16_22",
        "BP16_24",  # 주말 취침/기상 분
    ],
    # ── 사회경제 ───────────────────────────────────────────────
    "사회경제": [
        "marri_1",  # 결혼여부
        "occp",  # 직업
        "EC1_1",  # 경제활동상태
        "allownc",  # 기초생활수급
        "tins",  # 건강보험종류
    ],
    # ── 기타 질병력 / 가족력 ───────────────────────────────────
    "가족력_추가": [
        "HE_IHDfh1",
        "HE_IHDfh2",
        "HE_IHDfh3",  # 허혈성심장질환
        "HE_STRfh1",
        "HE_STRfh2",
        "HE_STRfh3",  # 뇌졸중
    ],
    "질병력": [
        "DI3_dg",  # 뇌졸중 진단
        "DI4_dg",  # 심근경색/협심증
        "DN1_dg",  # 신장질환
        "HE_anem",  # 빈혈
        "HE_obe",  # 비만
    ],
}


# ================================================================
# 데이터 로드 함수
# ================================================================
def load_year(year: int) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, YEAR_FILES[year])
    if not os.path.exists(path):
        print(f"  [SKIP] {year}년 파일 없음: {path}")
        return None
    print(f"  로딩 {year}년...", end=" ")
    df, _ = pyreadstat.read_sas7bdat(path)
    df["year"] = year
    print(f"{len(df):,}행 / {df.shape[1]}열")
    return df


def load_all() -> pd.DataFrame:
    print("=" * 60)
    print("데이터 로드 (hn18~24)")
    print("=" * 60)
    frames = []
    for year in sorted(YEAR_FILES.keys()):
        df = load_year(year)
        if df is not None:
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\n통합 완료: {combined.shape[0]:,}행 / {combined.shape[1]}열\n")
    return combined


# ================================================================
# 1. 기본 통계 요약
# ================================================================
def summary_basic(df: pd.DataFrame):
    print("=" * 60)
    print("[1] 연도별 샘플 수")
    print("=" * 60)
    print(df["year"].value_counts().sort_index().to_string())

    print("\n[타겟 단계별 분포 (전체)]")
    STAGE_LABEL = {
        "HTN": {0: "정상", 1: "주의", 2: "전단계", 3: "1단계", 4: "2단계"},
        "DM": {0: "정상", 1: "혈당장애", 2: "당뇨"},
        "DL": {0: "정상", 1: "경계", 2: "위험", 3: "고위험"},
        "OBE": {0: "정상이하", 1: "비만전단계", 2: "1단계", 3: "2단계", 4: "3단계"},
    }
    for tgt_name, col in TARGET_MAP.items():
        if col not in df.columns:
            print(f"  {tgt_name}: 컬럼 없음({col})")
            continue
        vc = df[col].value_counts(dropna=True).sort_index()
        total = vc.sum()
        labels = STAGE_LABEL.get(tgt_name, {})
        print(f"\n  ▶ {tgt_name} ({col})  총 유효 {total:,}명")
        for stage, cnt in vc.items():
            label = labels.get(int(stage), str(stage))
            print(f"    {int(stage)} {label:10s}: {cnt:6,}명 ({cnt / total * 100:5.1f}%)")


# ================================================================
# 2. 후보 변수 결측률 + 기초 통계
# ================================================================
def summary_candidates(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[2] 후보 변수 결측률 및 기초 통계")
    print("=" * 60)

    all_candidates = []
    for group, cols in CANDIDATE_GROUPS.items():
        for col in cols:
            if col in df.columns:
                all_candidates.append((group, col))

    rows = []
    for group, col in all_candidates:
        series = df[col]
        n_total = len(series)
        n_miss = series.isna().sum()
        miss_pct = n_miss / n_total * 100
        dtype = str(series.dtype)
        n_uniq = series.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(series):
            mean = series.mean()
            std = series.std()
            stat_str = f"mean={mean:.2f} std={std:.2f}"
        else:
            top_val = series.value_counts().index[0] if n_uniq > 0 else "-"
            stat_str = f"top={top_val} uniq={n_uniq}"

        rows.append(
            {
                "그룹": group,
                "변수": col,
                "dtype": dtype,
                "결측률(%)": round(miss_pct, 1),
                "고유값수": n_uniq,
                "기초통계": stat_str,
            }
        )

    result = pd.DataFrame(rows)
    # 결측률 30% 이상은 주의 표시
    result["주의"] = result["결측률(%)"].apply(lambda x: "⚠️ 고결측" if x >= 30 else "")
    print(result.to_string(index=False))

    # CSV 저장
    out_path = os.path.join(PLOT_DIR, "candidate_variables_summary.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  → CSV 저장: {out_path}")
    return result


# ================================================================
# 3. 후보 변수 × 타겟 상관 분석 (점이연 / 스피어만)
# ================================================================
def correlation_with_target(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[3] 후보 변수 × 타겟 상관관계")
    print("=" * 60)

    all_candidates = []
    for group, cols in CANDIDATE_GROUPS.items():
        for col in cols:
            if col in df.columns:
                all_candidates.append(col)
    all_candidates = list(dict.fromkeys(all_candidates))  # 중복 제거

    corr_rows = []
    for tgt_name, tgt_col in TARGET_MAP.items():
        if tgt_col not in df.columns:
            continue
        tgt = pd.to_numeric(df[tgt_col], errors="coerce")
        valid_tgt = tgt.notna()

        for col in all_candidates:
            if col not in df.columns:
                continue
            feat = pd.to_numeric(df[col], errors="coerce")
            mask = valid_tgt & feat.notna()
            n = mask.sum()
            if n < 100:
                continue
            try:
                from scipy.stats import spearmanr

                rho, pval = spearmanr(feat[mask], tgt[mask])
                corr_rows.append(
                    {
                        "타겟": tgt_name,
                        "변수": col,
                        "스피어만_rho": round(rho, 4),
                        "p값": round(pval, 4),
                        "n": n,
                        "유의": "✓" if pval < 0.05 else "",
                    }
                )
            except Exception:
                pass

    corr_df = pd.DataFrame(corr_rows)
    if corr_df.empty:
        print("  결과 없음")
        return

    corr_df["abs_rho"] = corr_df["스피어만_rho"].abs()
    corr_df = corr_df.sort_values(["타겟", "abs_rho"], ascending=[True, False])

    for tgt_name in TARGET_MAP:
        subset = corr_df[corr_df["타겟"] == tgt_name].head(20)
        if subset.empty:
            continue
        print(f"\n  ▶ {tgt_name} — 상위 20개")
        print(subset[["변수", "스피어만_rho", "p값", "유의", "n"]].to_string(index=False))

    # CSV 저장
    out_path = os.path.join(PLOT_DIR, "candidate_correlation.csv")
    corr_df.drop(columns="abs_rho").to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  → CSV 저장: {out_path}")
    return corr_df


# ================================================================
# 4. 주요 후보 변수 분포 시각화 (타겟별)
# ================================================================
def plot_top_candidates(df: pd.DataFrame, corr_df: pd.DataFrame, top_n: int = 10):
    print("\n" + "=" * 60)
    print(f"[4] 상위 {top_n}개 후보 변수 분포 시각화")
    print("=" * 60)

    for tgt_name, tgt_col in TARGET_MAP.items():
        if tgt_col not in df.columns:
            continue
        tgt = pd.to_numeric(df[tgt_col], errors="coerce")

        subset = corr_df[corr_df["타겟"] == tgt_name].head(top_n)
        top_cols = subset["변수"].tolist()
        if not top_cols:
            continue

        n_cols = 5
        n_rows = (len(top_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
        axes = axes.flatten()
        fig.suptitle(f"{tgt_name} — 상위 후보 변수 분포 (유병 0 vs 1)", fontsize=14)

        for i, col in enumerate(top_cols):
            ax = axes[i]
            feat = pd.to_numeric(df[col], errors="coerce")
            mask = tgt.notna() & feat.notna()
            g0 = feat[mask & (tgt == 0)]
            g1 = feat[mask & (tgt == 1)]

            ax.hist(g0, bins=30, alpha=0.5, label="유병 無(0)", density=True, color="#4C9BE8")
            ax.hist(g1, bins=30, alpha=0.5, label="유병 有(1)", density=True, color="#E8734C")
            rho_val = subset[subset["변수"] == col]["스피어만_rho"].values
            rho_str = f"rho={rho_val[0]:.3f}" if len(rho_val) else ""
            ax.set_title(f"{col}\n{rho_str}", fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel("")

        for j in range(len(top_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out_path = os.path.join(PLOT_DIR, f"top_candidates_{tgt_name}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  → 저장: {out_path}")


# ================================================================
# 5. 연도별 주요 변수 추이
# ================================================================
def plot_yearly_trend(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[5] 연도별 변수 추이 (타겟 유병율)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (tgt_name, col) in zip(axes, TARGET_MAP.items()):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        trend = df.groupby("year")[col].apply(
            lambda x: (pd.to_numeric(x, errors="coerce") == 1).sum()
            / pd.to_numeric(x, errors="coerce").notna().sum()
            * 100
        )
        ax.plot(trend.index, trend.values, marker="o", color="#4C9BE8")
        ax.set_title(f"{tgt_name} 유병율 추이 (%)")
        ax.set_xlabel("연도")
        ax.set_ylabel("%")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "yearly_trend.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → 저장: {out_path}")


# ================================================================
# 6. 컬럼 존재 여부 확인 (연도별)
# ================================================================
def check_column_availability():
    print("\n" + "=" * 60)
    print("[6] 후보 변수 연도별 존재 여부")
    print("=" * 60)

    all_candidates = []
    for group, cols in CANDIDATE_GROUPS.items():
        for col in cols:
            all_candidates.append((group, col))

    col_availability = {}
    for year in sorted(YEAR_FILES.keys()):
        path = os.path.join(DATA_DIR, YEAR_FILES[year])
        if not os.path.exists(path):
            continue
        _, meta = pyreadstat.read_sas7bdat(path)
        col_availability[year] = set(meta.column_names)

    rows = []
    for group, col in all_candidates:
        row = {"그룹": group, "변수": col}
        for year in sorted(col_availability.keys()):
            row[str(year)] = "O" if col in col_availability[year] else "-"
        rows.append(row)

    avail_df = pd.DataFrame(rows)
    print(avail_df.to_string(index=False))

    out_path = os.path.join(PLOT_DIR, "column_availability.csv")
    avail_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  → CSV 저장: {out_path}")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    # Step 1: 연도별 컬럼 존재 여부 먼저 확인 (데이터 안 읽어도 됨)
    check_column_availability()

    # Step 2: 전체 데이터 로드
    df = load_all()

    # Step 2-1: 다중분류 타겟 생성
    print("타겟 변수 생성 중...")
    df = create_targets(df)
    for tgt_name, col in TARGET_MAP.items():
        vc = df[col].value_counts(dropna=False).sort_index()
        print(f"  {tgt_name} ({col}): {vc.to_dict()}")

    # Step 3: 기본 요약
    summary_basic(df)

    # Step 4: 후보 변수 결측률
    summary_candidates(df)

    # Step 5: 타겟 상관관계
    corr_df = correlation_with_target(df)

    # Step 6: 시각화
    if corr_df is not None and not corr_df.empty:
        plot_top_candidates(df, corr_df, top_n=10)

    plot_yearly_trend(df)

    print("\n" + "=" * 60)
    print("완료! plots/ 폴더 확인하세요.")
    print("=" * 60)
