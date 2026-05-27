from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ai_runtime.ml.inference.catboost_predictor import CatBoostDiseasePredictor
from ai_runtime.ml.inference.feature_mapper import FeatureMappingError, map_service_features
from ai_runtime.ml.inference.schemas import DiseasePrediction

logger = logging.getLogger(__name__)

DISEASE_ARTIFACTS = {
    "DM": Path("ai_runtime/ml/artifacts/dm/catboost"),
    "HTN": Path("ai_runtime/ml/artifacts/htn/catboost"),
    "DL": Path("ai_runtime/ml/artifacts/dl/catboost"),
}


def predict_chronic_disease_risks(
    user: Any,
    health_record: Any,
    diseases: list[str] | None = None,
) -> dict[str, DiseasePrediction]:
    requested = diseases or list(DISEASE_ARTIFACTS)
    predictions: dict[str, DiseasePrediction] = {}
    for disease in requested:
        disease_key = disease.upper()
        artifact_dir = DISEASE_ARTIFACTS.get(disease_key)
        if artifact_dir is None:
            continue
        predictor = CatBoostDiseasePredictor(disease_key, artifact_dir)
        feature_columns = predictor.load_feature_columns()
        if not feature_columns:
            continue
        try:
            mapping = map_service_features(user, health_record, feature_columns, strict=True)
        except FeatureMappingError as exc:
            logger.info(
                "ML feature mapping skipped",
                extra={
                    "disease_key": disease_key,
                    "missing_sources": exc.missing_sources,
                    "warnings": exc.warnings,
                },
            )
            continue
        prediction = predictor.predict(mapping.features)
        if prediction is not None:
            predictions[disease_key] = prediction
    return predictions


def warmup_chronic_disease_models(diseases: list[str] | None = None) -> dict[str, dict[str, Any]]:
    requested = diseases or list(DISEASE_ARTIFACTS)
    results: dict[str, dict[str, Any]] = {}
    for disease in requested:
        disease_key = disease.upper()
        artifact_dir = DISEASE_ARTIFACTS.get(disease_key)
        if artifact_dir is None:
            results[disease_key] = {"status": "skipped", "reason": "unknown_disease"}
            continue

        predictor = CatBoostDiseasePredictor(disease_key, artifact_dir)
        try:
            feature_columns = predictor.load_feature_columns()
            if not predictor.available or not feature_columns:
                results[disease_key] = {
                    "status": "skipped",
                    "reason": "artifact_unavailable",
                    "artifact_dir": str(artifact_dir),
                }
                continue

            warmed = predictor.warmup()
        except Exception as exc:
            logger.exception(
                "ML warmup failed",
                extra={"disease_key": disease_key, "artifact_dir": str(artifact_dir)},
            )
            results[disease_key] = {
                "status": "failed",
                "reason": exc.__class__.__name__,
                "artifact_dir": str(artifact_dir),
            }
            continue

        results[disease_key] = {
            "status": "ok",
            "model_count": warmed,
            "feature_count": len(feature_columns),
            "artifact_dir": str(artifact_dir),
        }
        logger.info(
            "ML warmup completed",
            extra={
                "disease_key": disease_key,
                "model_count": warmed,
                "feature_count": len(feature_columns),
                "artifact_dir": str(artifact_dir),
            },
        )
    return results
