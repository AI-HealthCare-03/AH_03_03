"""Warm up CatBoost chronic disease artifacts before a demo.

This script intentionally does not run at FastAPI startup. Use it manually when
you want to pay the model load cost before the first PRECISION analysis request.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ai_runtime.ml.inference.disease_risk_service import warmup_chronic_disease_models  # noqa: E402


def main() -> int:
    results = warmup_chronic_disease_models()
    failed = False
    for disease_key, result in results.items():
        status = result.get("status")
        print(
            "{disease}: status={status}, models={models}, features={features}, artifact_dir={artifact_dir}".format(
                disease=disease_key,
                status=status,
                models=result.get("model_count", "-"),
                features=result.get("feature_count", "-"),
                artifact_dir=result.get("artifact_dir", "-"),
            )
        )
        failed = failed or status == "failed"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
