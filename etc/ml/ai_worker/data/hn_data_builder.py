"""
hn_data_builder.py
──────────────────────────────────────────────────────────────────────
KNHANES 연도별 SAS 파일 → 전체 통합 데이터셋 생성기

역할
  - 각 연도 SAS 원본 전체 컬럼 로드
  - 'year' 컬럼 추가 후 전체 연도 concat
  - hn_master.csv + hn_master.parquet 동시 저장
  - 연도별로 없는 컬럼은 NaN으로 채워 스키마 통일

사용법
  python hn_data_builder.py             # 전체 빌드
  python hn_data_builder.py --dry-run   # 파일 존재 여부만 확인

저장 파일
  hn_master.csv      : 범용 (Excel 등 외부 도구)
  hn_master.parquet  : 분석용 (빠름, 용량 1/5~1/10, 타입 보존)
──────────────────────────────────────────────────────────────────────
"""

import os
import argparse
import pandas as pd
import pyreadstat

# ── 경로 설정 ─────────────────────────────────────────────────────────
DATA_DIR       = "/Users/admin/PycharmProjects/AH_03_03/etc/ml/ai_worker/data"
SAVE_CSV       = os.path.join(DATA_DIR, "hn_master.csv")
SAVE_PARQUET   = os.path.join(DATA_DIR, "hn_master.parquet")

# ── 연도별 SAS 파일 목록 ──────────────────────────────────────────────
YEAR_PATHS = {
    2015: os.path.join(DATA_DIR, "hn15_all.sas7bdat"),
    2016: os.path.join(DATA_DIR, "hn16_all.sas7bdat"),
    2017: os.path.join(DATA_DIR, "hn17_all.sas7bdat"),
    2018: os.path.join(DATA_DIR, "hn18_all.sas7bdat"),
    2019: os.path.join(DATA_DIR, "hn19_all.sas7bdat"),
    2020: os.path.join(DATA_DIR, "hn20_all.sas7bdat"),
    2021: os.path.join(DATA_DIR, "hn21_all.sas7bdat"),
    2022: os.path.join(DATA_DIR, "hn22_all.sas7bdat"),
    2023: os.path.join(DATA_DIR, "hn23_all.sas7bdat"),
    2024: os.path.join(DATA_DIR, "hn24_all.sas7bdat"),
}


def check_files():
    """파일 존재 여부 체크"""
    print("=" * 55)
    print("파일 존재 여부 확인")
    print("=" * 55)
    found = 0
    for year, path in YEAR_PATHS.items():
        exists = os.path.exists(path)
        status = "✅ 있음" if exists else "❌ 없음"
        size   = ""
        if exists:
            size_mb = os.path.getsize(path) / 1024 / 1024
            size = f"({size_mb:.0f}MB)"
            found += 1
        print(f"  hn{year}: {status} {size}")
    print("=" * 55)
    print(f"  총 {found}/{len(YEAR_PATHS)}개 파일 존재")
    return found


def load_one_year(year, path, all_cols):
    """
    단일 연도 SAS 전체 로드
    - all_cols: 전체 연도 union 컬럼 목록 (스키마 통일용)
    - 해당 연도에 없는 컬럼은 NaN으로 채움
    - year 컬럼 추가
    - 성인(19세+) 필터
    """
    # 전체 컬럼 로드 (usecols 없이 — 해당 연도 모든 컬럼)
    df, _ = pyreadstat.read_sas7bdat(path, encoding="cp949")

    # 없는 컬럼 NaN으로 추가 — 한 번에 concat (루프 insert 시 fragmentation 경고 방지)
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        nan_df = pd.DataFrame(float("nan"), index=df.index, columns=missing_cols)
        df = pd.concat([df, nan_df], axis=1)

    # 컬럼 순서 통일 (all_cols 기준)
    df = df[all_cols]

    # 성인 필터
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[df["age"] >= 19].reset_index(drop=True)

    # ID 제거
    df = df.drop(columns=["ID"], errors="ignore")

    # year 컬럼 추가 (SAS 내부 year 컬럼과 구분하여 survey_year로 추가)
    df["survey_year"] = year

    return df


def get_all_columns():
    """
    전체 연도 컬럼 union 수집
    → 연도별로 컬럼이 다를 수 있으므로 전체를 먼저 파악
    """
    print("\n전체 연도 컬럼 union 수집 중...")
    all_cols = []
    seen = set()
    for year, path in YEAR_PATHS.items():
        if not os.path.exists(path):
            continue
        _, meta = pyreadstat.read_sas7bdat(path, metadataonly=True, encoding="cp949")
        for col in meta.column_names:
            if col not in seen:
                all_cols.append(col)
                seen.add(col)
        print(f"  hn{year}: {len(meta.column_names)}개 컬럼 확인")

    print(f"  → 전체 union 컬럼 수: {len(all_cols)}개")
    return all_cols


def build_master():
    """전체 연도 통합 → CSV + Parquet 저장"""
    print("\n전체 연도 데이터 빌드 시작")
    print("=" * 55)

    # 1단계: 전체 컬럼 union 수집
    all_cols = get_all_columns()

    # 2단계: 연도별 로드 및 통합
    print("\n연도별 데이터 로드 중...")
    dfs = []
    for year, path in YEAR_PATHS.items():
        if not os.path.exists(path):
            print(f"  hn{year}: 파일 없음 — skip")
            continue

        print(f"  hn{year} 로드 중...", end=" ", flush=True)
        df_year = load_one_year(year, path, all_cols)
        dfs.append(df_year)
        print(f"완료 | {len(df_year):,}행 × {len(df_year.columns)}컬럼")

    if not dfs:
        raise FileNotFoundError(
            f"로드 가능한 파일이 없습니다. DATA_DIR 확인: {DATA_DIR}"
        )

    # 3단계: concat
    print("\n전체 연도 concat 중...")
    df_master = pd.concat(dfs, ignore_index=True)

    # survey_year를 앞으로 이동
    cols_order = ["survey_year"] + [c for c in df_master.columns if c != "survey_year"]
    df_master = df_master[cols_order]

    print(f"통합 완료 | {len(df_master):,}행 × {len(df_master.columns)}컬럼")
    print("\n연도별 샘플 수:")
    year_vc = df_master["survey_year"].value_counts().sort_index()
    for yr, cnt in year_vc.items():
        print(f"  {yr}년: {cnt:,}명")

    # 4단계: CSV 저장
    print(f"\nCSV 저장 중... → {SAVE_CSV}")
    df_master.to_csv(SAVE_CSV, index=False, encoding="utf-8-sig")
    csv_mb = os.path.getsize(SAVE_CSV) / 1024 / 1024
    print(f"CSV 저장 완료 | {csv_mb:.0f}MB")

    # 5단계: Parquet 저장
    print(f"\nParquet 저장 중... → {SAVE_PARQUET}")
    try:
        df_master.to_parquet(SAVE_PARQUET, index=False, engine="pyarrow")
        pq_mb = os.path.getsize(SAVE_PARQUET) / 1024 / 1024
        print(f"Parquet 저장 완료 | {pq_mb:.0f}MB")
        print(f"압축률: {pq_mb/csv_mb*100:.0f}% (CSV 대비)")
    except ImportError:
        print("⚠️  pyarrow 없음 — Parquet 저장 skip")
        print("   설치: pip install pyarrow")

    print("\n" + "=" * 55)
    print("빌드 완료")
    print(f"  CSV     : {SAVE_CSV}")
    print(f"  Parquet : {SAVE_PARQUET}")
    print("=" * 55)

    return df_master


def summary(df_master):
    """마스터 데이터 요약"""
    print("\n[마스터 데이터 요약]")
    print(f"  총 샘플  : {len(df_master):,}행")
    print(f"  총 컬럼  : {len(df_master.columns)}개")

    # 주요 질환 유병 분포
    target_map = {
        "DI1_pr": "고혈압유병",
        "DE1_pr": "당뇨유병",
        "DI2_pr": "이상지질혈증유병",
        "DL1_pr": "간질환유병",
        "DL1_dg": "간질환진단",
    }
    print("\n  주요 질환 분포 (원본 코드 기준 — 이진화 전):")
    for col, label in target_map.items():
        if col not in df_master.columns:
            print(f"    {label}({col}): 컬럼 없음")
            continue
        n1  = (df_master[col] == 1).sum()
        n8  = (df_master[col] == 8).sum()
        nan = df_master[col].isna().sum()
        print(f"    {label}: 유병(1)={n1:,} / 정상(8)={n8:,} / NaN={nan:,}")

    # 결측률 상위 10개
    miss = (df_master.isnull().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    print(f"\n  결측률 상위 10개:")
    for col, pct in miss.head(10).items():
        print(f"    {col}: {pct:.1f}%")


# ── 실행 진입점 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNHANES 전체 연도 데이터 빌더")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 존재 여부만 확인 (실제 로드 안 함)"
    )
    args = parser.parse_args()

    if args.dry_run:
        check_files()
    else:
        check_files()
        df_master = build_master()
        summary(df_master)
