from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ai_worker.ml.inference.catboost_predictor import CatBoostDiseasePredictor
from ai_worker.ml.inference.feature_mapper import FeatureMappingError, map_service_features
from ai_worker.ml.inference.schemas import DiseasePrediction

logger = logging.getLogger(__name__)

DISEASE_ARTIFACTS = {
    "DM": Path("ai_worker/ml/artifacts/dm/catboost"),
    "HTN": Path("ai_worker/ml/artifacts/htn/catboost"),
    "DL": Path("ai_worker/ml/artifacts/dl/catboost"),
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
