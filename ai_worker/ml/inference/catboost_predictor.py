from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_worker.ml.common.artifacts import read_json
from ai_worker.ml.inference.schemas import DiseasePrediction


class FeatureSchemaError(ValueError):
    pass


class CatBoostDiseasePredictor:
    def __init__(self, disease: str, artifact_dir: str | Path):
        self.disease = disease
        self.artifact_dir = Path(artifact_dir)
        self.model_paths = sorted(self.artifact_dir.glob("model_fold*.cbm"))
        self.feature_columns_path = self.artifact_dir / "feature_columns.json"
        self.threshold_path = self.artifact_dir / "threshold.json"
        self.metrics_path = self.artifact_dir / "metrics.json"
        self.config_path = self.artifact_dir / "experiment_config.json"
        self._models: list[Any] | None = None

    @property
    def available(self) -> bool:
        return bool(self.model_paths) and self.feature_columns_path.exists() and self.threshold_path.exists()

    def load_feature_columns(self) -> list[str]:
        if not self.feature_columns_path.exists():
            return []
        return list(read_json(self.feature_columns_path))

    def warmup(self) -> int:
        if not self.available:
            return 0

        try:
            from catboost import CatBoostClassifier
        except ImportError:
            return 0

        if self._models is None:
            self._models = []
            for model_path in self.model_paths:
                model = CatBoostClassifier()
                model.load_model(str(model_path))
                self._models.append(model)
        return len(self._models)

    def predict(self, features: dict[str, Any]) -> DiseasePrediction | None:
        if not self.available:
            return None
        feature_columns = self.load_feature_columns()
        if not feature_columns:
            return None

        _validate_feature_schema(features, feature_columns)

        try:
            import pandas as pd
        except ImportError:
            return None

        self.warmup()

        row = {column: features[column] for column in feature_columns}
        frame = pd.DataFrame([row], columns=feature_columns)
        if frame.shape[1] != len(feature_columns):
            raise FeatureSchemaError(
                f"{self.disease} feature column 개수 불일치: expected={len(feature_columns)}, actual={frame.shape[1]}"
            )
        probabilities = [float(model.predict_proba(frame)[0][1]) for model in self._models]
        probability = sum(probabilities) / max(len(probabilities), 1)
        threshold_payload = read_json(self.threshold_path)
        threshold = float(threshold_payload.get("threshold", 0.5))
        return DiseasePrediction(
            disease=self.disease,
            probability=probability,
            threshold=threshold,
            risk_level=_risk_level(probability, threshold),
            model_name="catboost",
            model_version=_model_version(self.config_path, self.disease),
            artifact_dir=str(self.artifact_dir),
            factors=_top_factors(self.metrics_path),
        )


def _risk_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "HIGH"
    if probability >= max(0.25, threshold * 0.65):
        return "MEDIUM"
    return "LOW"


def _model_version(config_path: Path, disease: str) -> str:
    if not config_path.exists():
        return f"{disease.lower()}-catboost"
    payload = read_json(config_path)
    return str(payload.get("experiment_id") or f"{disease.lower()}-catboost")


def _top_factors(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    payload = read_json(metrics_path)
    feature_importance = payload.get("feature_importance")
    if not isinstance(feature_importance, list):
        return []
    return feature_importance[:5]


def _validate_feature_schema(features: dict[str, Any], feature_columns: list[str]) -> None:
    missing = [column for column in feature_columns if column not in features]
    if missing:
        raise FeatureSchemaError(f"feature_columns.json 기준 누락 feature가 있습니다: {missing}")

    unexpected = [column for column in features if column not in feature_columns]
    if unexpected:
        raise FeatureSchemaError(f"feature_columns.json 기준 허용되지 않은 feature가 있습니다: {unexpected}")

    expected = len(feature_columns)
    actual = len(features)
    if actual != expected:
        raise FeatureSchemaError(f"feature column 개수 불일치: expected={expected}, actual={actual}")
