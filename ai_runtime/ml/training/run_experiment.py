from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ai_runtime.ml.training.train_catboost import train_catboost_from_config
from ai_runtime.ml.training.train_xgboost import train_xgboost_from_config


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def run_experiment(config: dict[str, Any], *, dry_run: bool = False, sample_size: int | None = None) -> dict[str, Any]:
    model_type = str(config.get("model_type") or "").lower()
    if model_type == "catboost":
        return train_catboost_from_config(config, dry_run=dry_run, sample_size=sample_size)
    if model_type == "xgboost":
        return train_xgboost_from_config(config, dry_run=dry_run, sample_size=sample_size)
    raise ValueError(f"지원하지 않는 model_type입니다: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Config 기반 ML 실험 실행")
    parser.add_argument("--config", required=True, help="experiment config JSON 경로")
    parser.add_argument("--dry-run", action="store_true", help="데이터/피처 스키마만 검증하고 학습하지 않음")
    parser.add_argument("--sample-size", type=int, default=None, help="앞쪽 N개 row만 사용")
    args = parser.parse_args()

    result = run_experiment(
        load_experiment_config(args.config),
        dry_run=args.dry_run,
        sample_size=args.sample_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
