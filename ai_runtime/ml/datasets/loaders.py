from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_runtime.ml.datasets.registry import get_dataset_path


def resolve_dataset_path(config: dict[str, Any]) -> Path:
    dataset_path = config.get("dataset_path")
    if dataset_path:
        path = Path(str(dataset_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {path}")
        return path

    dataset_name = str(config.get("dataset_name") or "hn_all")
    return Path(get_dataset_path(dataset_name))


def load_csv_dataset(config: dict[str, Any], sample_size: int | None = None):
    import pandas as pd

    path = resolve_dataset_path(config)
    df = pd.read_csv(path)
    if sample_size is not None and sample_size > 0:
        df = df.head(sample_size).copy()
    return df
