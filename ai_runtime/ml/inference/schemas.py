from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiseaseRiskInput:
    features: dict[str, Any]


@dataclass(frozen=True)
class DiseasePrediction:
    disease: str
    probability: float
    threshold: float
    risk_level: str
    model_name: str
    model_version: str
    artifact_dir: str
    factors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease": self.disease,
            "probability": self.probability,
            "threshold": self.threshold,
            "risk_level": self.risk_level,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "artifact_dir": self.artifact_dir,
            "factors": self.factors,
        }
