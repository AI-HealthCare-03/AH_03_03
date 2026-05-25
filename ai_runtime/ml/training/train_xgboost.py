from __future__ import annotations

from typing import Any


def train_xgboost_from_config(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    sample_size: int | None = None,
) -> dict[str, Any]:
    _ = sample_size
    if dry_run:
        return {
            "status": "dry_run",
            "model_type": "xgboost",
            "disease": config.get("disease"),
            "message": "XGBoost 학습 경로는 config 인터페이스만 준비되어 있습니다.",
        }
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("XGBoost 학습을 실행하려면 xgboost 패키지가 필요합니다.") from exc
    raise NotImplementedError("XGBoost 학습 구현은 CatBoost 파이프라인 안정화 후 확장합니다.")
