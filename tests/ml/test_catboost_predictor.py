from __future__ import annotations

import pytest

from ai_worker.ml.inference.catboost_predictor import FeatureSchemaError, _validate_feature_schema


def test_validate_feature_schema_rejects_missing_feature() -> None:
    with pytest.raises(FeatureSchemaError):
        _validate_feature_schema({"성별": 1.0}, ["성별", "나이"])


def test_validate_feature_schema_rejects_unexpected_feature() -> None:
    with pytest.raises(FeatureSchemaError):
        _validate_feature_schema({"성별": 1.0, "나이": 45.0, "예상외_feature": 0.0}, ["성별", "나이"])


def test_validate_feature_schema_accepts_exact_feature_columns() -> None:
    _validate_feature_schema({"성별": 1.0, "나이": 45.0}, ["성별", "나이"])
