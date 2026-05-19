"""
데이터셋 레지스트리
어떤 실험이 어떤 CSV를 사용했는지 추적하기 위한 단일 진실 공급원(SSOT)
경로 변경 시 여기만 수정하면 됨
"""

from __future__ import annotations

import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터셋 등록
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BASE = "/Users/admin/PycharmProjects/AH_03_03/ai_worker/data"

DATASETS: dict[str, str] = {
    "hn_all":  os.path.join(_BASE, "hn_all_preprocessed.csv"),   # hn18~24 통합 (현재 실험 기준)
    "hn2224":  os.path.join(_BASE, "hn2224_preprocessed.csv"),   # hn22~24 통합
    "hn24":    os.path.join(_BASE, "hn24_file1_preprocessed.csv"), # hn24 단독
}

# 현재 실험 기본값
DEFAULT_DATASET = "hn_all"


def get_dataset_path(name: str = DEFAULT_DATASET) -> str:
    """
    데이터셋 이름으로 경로 반환

    Parameters
    ----------
    name : DATASETS 키 (default: "hn_all")

    Returns
    -------
    CSV 절대 경로

    Raises
    ------
    KeyError : 등록되지 않은 데이터셋 이름
    FileNotFoundError : 경로에 파일이 없을 때
    """
    if name not in DATASETS:
        raise KeyError(f"등록되지 않은 데이터셋: '{name}'. 등록된 목록: {list(DATASETS.keys())}")

    path = DATASETS[name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"데이터셋 파일 없음: {path}")

    return path


def list_datasets() -> None:
    """등록된 데이터셋 목록 출력"""
    print("[ 등록된 데이터셋 ]")
    for name, path in DATASETS.items():
        exists = "✅" if os.path.exists(path) else "❌ 파일 없음"
        print(f"  {name:<10} {exists}  {path}")
