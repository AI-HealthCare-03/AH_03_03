from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from ai_runtime.ml.common.artifacts import read_json

SUPPORTED_SCREENING_DISEASES = frozenset({"HTN", "DM", "DL"})
SCREENING_ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "screening"
MODEL_DIR_NAME = "screening_catboost"
HTN_FAMILY_FEATURE_PREFIX = "고혈압가족력_"


class ScreeningArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class ScreeningArtifact:
    disease_code: str
    artifact_dir: Path
    feature_columns: list[str]
    threshold: float
    models: list[Any]

    @property
    def model_count(self) -> int:
        return len(self.models)


@dataclass(frozen=True)
class ScreeningPrediction:
    disease_code: str
    probability: float
    threshold: float
    screening_high: bool
    missing_features: list[str]
    neutralized_features: list[str]
    model_count: int


def load_screening_artifact(disease_code: str) -> ScreeningArtifact:
    normalized_disease_code = _normalize_disease_code(disease_code)
    return _load_screening_artifact_cached(normalized_disease_code)


def predict_screening_risk(disease_code: str, features: Mapping[str, Any]) -> ScreeningPrediction:
    artifact = load_screening_artifact(disease_code)
    row, missing_features, neutralized_features = _build_feature_row(
        disease_code=artifact.disease_code,
        feature_columns=artifact.feature_columns,
        features=features,
    )
    frame = pd.DataFrame([row], columns=artifact.feature_columns)
    fold_probabilities = [float(model.predict_proba(frame)[0][1]) for model in artifact.models]
    probability = sum(fold_probabilities) / max(len(fold_probabilities), 1)

    # Probability is returned for internal policy/debug use only. Product responses
    # should expose the 4-step service band, not raw model probability.
    return ScreeningPrediction(
        disease_code=artifact.disease_code,
        probability=probability,
        threshold=artifact.threshold,
        screening_high=probability >= artifact.threshold,
        missing_features=missing_features,
        neutralized_features=neutralized_features,
        model_count=artifact.model_count,
    )


@lru_cache(maxsize=len(SUPPORTED_SCREENING_DISEASES))
def _load_screening_artifact_cached(disease_code: str) -> ScreeningArtifact:
    artifact_dir = SCREENING_ARTIFACT_ROOT / disease_code.lower() / MODEL_DIR_NAME
    feature_columns_path = artifact_dir / "feature_columns.json"
    threshold_path = artifact_dir / "threshold.json"
    model_paths = sorted(artifact_dir.glob("model_fold*.cbm"))

    if not artifact_dir.exists():
        raise ScreeningArtifactError(f"screening artifact directory not found: {artifact_dir}")
    if not feature_columns_path.exists():
        raise ScreeningArtifactError(f"feature_columns.json not found: {feature_columns_path}")
    if not threshold_path.exists():
        raise ScreeningArtifactError(f"threshold.json not found: {threshold_path}")
    if len(model_paths) != 5:
        raise ScreeningArtifactError(
            f"{disease_code} screening artifact requires 5 fold models, found={len(model_paths)}"
        )

    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ScreeningArtifactError("catboost is required to load screening artifacts") from exc

    feature_columns_payload = read_json(feature_columns_path)
    if not isinstance(feature_columns_payload, list) or not all(
        isinstance(column, str) for column in feature_columns_payload
    ):
        raise ScreeningArtifactError(f"invalid feature_columns.json: {feature_columns_path}")

    threshold_payload = read_json(threshold_path)
    threshold = float(threshold_payload.get("threshold", 0.5))

    models = []
    for model_path in model_paths:
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        models.append(model)

    return ScreeningArtifact(
        disease_code=disease_code,
        artifact_dir=artifact_dir,
        feature_columns=list(feature_columns_payload),
        threshold=threshold,
        models=models,
    )


def _build_feature_row(
    *,
    disease_code: str,
    feature_columns: list[str],
    features: Mapping[str, Any],
) -> tuple[dict[str, float], list[str], list[str]]:
    row: dict[str, float] = {}
    missing_features: list[str] = []
    neutralized_features: list[str] = []

    for column in feature_columns:
        value = features.get(column)
        if value is None:
            missing_features.append(column)
            neutralized_features.append(column)
            # Temporary adapter strategy: HTN family-history inputs are being
            # removed from the final policy, so missing HTN family features are
            # neutralized at runtime until the screening model is retrained.
            row[column] = _neutral_feature_value(disease_code=disease_code, feature_name=column)
            continue
        row[column] = _coerce_numeric_feature(value)

    return row, missing_features, neutralized_features


def _normalize_disease_code(disease_code: str) -> str:
    normalized = disease_code.upper()
    if normalized not in SUPPORTED_SCREENING_DISEASES:
        supported = ", ".join(sorted(SUPPORTED_SCREENING_DISEASES))
        raise ScreeningArtifactError(f"unsupported screening disease_code={disease_code!r}; supported={supported}")
    return normalized


def _neutral_feature_value(*, disease_code: str, feature_name: str) -> float:
    if disease_code == "HTN" and feature_name.startswith(HTN_FAMILY_FEATURE_PREFIX):
        return 0.0
    return 0.0


def _coerce_numeric_feature(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
