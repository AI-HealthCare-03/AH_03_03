from __future__ import annotations

"""Compare BASIC X1 rules and KDU screening artifacts.

Run:
    uv run python scripts/compare_dual_stage_screening.py
    uv run python scripts/compare_dual_stage_screening.py --mode synthetic
    uv run python scripts/compare_dual_stage_screening.py --mode real --n-samples 30000
    uv run python scripts/compare_dual_stage_screening.py --mode real --input-path <x1_csv_or_dir>
"""

# ruff: noqa: E402,I001

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.models.analysis import AnalysisType, RiskLevel
from app.services.analysis import (
    _basic_diabetes_score,
    _basic_dyslipidemia_score,
    _basic_hypertension_score,
    _risk_level,
)
from ai_runtime.ml.inference.feature_mapper import map_service_features


DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "experiment/ml/kdu/screening/artifacts"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "tmp/dual_stage_screening_compare_real.csv"
DEFAULT_SUMMARY_MD = REPO_ROOT / "tmp/dual_stage_screening_summary_real.md"

SERVICE_FEATURES = [
    "성별",
    "나이",
    "음주빈도",
    "음주량",
    "현재흡연",
    "걷기일수",
    "근력운동일수",
    "고혈압가족력_부",
    "고혈압가족력_모",
    "고혈압가족력_형제",
    "고지혈증가족력_부",
    "고지혈증가족력_모",
    "고지혈증가족력_형제",
    "당뇨가족력_부",
    "당뇨가족력_모",
    "당뇨가족력_형제",
    "키",
    "체중",
    "BMI",
    "직업_관리전문",
    "직업_사무",
    "직업_서비스판매",
    "직업_농림어업",
    "직업_기능노무",
    "직업_주부학생",
    "직업_무직",
    "직업_작업미상",
]

FAMILY_COLUMNS = [
    "고혈압가족력_부",
    "고혈압가족력_모",
    "고혈압가족력_형제",
    "고지혈증가족력_부",
    "고지혈증가족력_모",
    "고지혈증가족력_형제",
    "당뇨가족력_부",
    "당뇨가족력_모",
    "당뇨가족력_형제",
]

HTN_FAMILY_COLUMNS = ["고혈압가족력_부", "고혈압가족력_모", "고혈압가족력_형제"]

DEFAULT_REAL_INPUTS = {
    "HTN": REPO_ROOT
    / "experiment/ml/kdu/data/processed/x1_service_runtime_hypertension_hn13_24_screening/hypertension_hn13_24_screening.csv",
    "DM": REPO_ROOT
    / "experiment/ml/kdu/data/processed/x1_service_runtime_diabetes_hn13_24_screening/diabetes_hn13_24_screening.csv",
    "DL": REPO_ROOT
    / "experiment/ml/kdu/data/processed/x1_service_runtime_dyslipidemia_hn13_24_screening_v2/dyslipidemia_hn13_24_screening_v2.csv",
}

DISEASES = {
    "HTN": {
        "label": "고혈압",
        "analysis_type": AnalysisType.HYPERTENSION,
        "artifact_name": "htn",
        "base_score": _basic_hypertension_score,
        "target_candidates": ["hypertension_screening_risk", "htn_screening_risk", "고혈압유병", "HTN", "target"],
    },
    "DM": {
        "label": "당뇨",
        "analysis_type": AnalysisType.DIABETES,
        "artifact_name": "dm",
        "base_score": _basic_diabetes_score,
        "target_candidates": ["diabetes_screening_risk", "dm_screening_risk", "당뇨유병", "DM", "target"],
    },
    "DL": {
        "label": "이상지질혈증",
        "analysis_type": AnalysisType.DYSLIPIDEMIA,
        "artifact_name": "dl",
        "base_score": _basic_dyslipidemia_score,
        "target_candidates": [
            "dyslipidemia_screening_v2",
            "dl_screening_risk",
            "이상지질혈증유병",
            "DL",
            "target",
        ],
    },
}

POLICY_LABELS = {
    (False, False): "낮음",
    (False, True): "관심 필요",
    (True, False): "주의",
    (True, True): "높은 주의",
}


@dataclass(frozen=True)
class DummyCase:
    case_id: str
    description: str
    user: SimpleNamespace
    record: SimpleNamespace


@dataclass(frozen=True)
class SyntheticRow:
    disease: str
    case_id: str
    description: str
    base_score: Decimal
    base_risk_level: RiskLevel
    base_high: bool
    screening_probability: float
    screening_threshold: float
    screening_high: bool
    service_band: str


class ScreeningArtifact:
    def __init__(self, disease: str, artifact_dir: Path):
        self.disease = disease
        self.artifact_dir = artifact_dir
        self.model_paths = sorted(artifact_dir.glob("model_fold*.cbm"))
        self.feature_columns = _read_json(artifact_dir / "feature_columns.json")
        threshold_payload = _read_json(artifact_dir / "threshold.json")
        self.threshold = float(threshold_payload.get("threshold", 0.5))
        self._models: list[Any] | None = None

    @property
    def available(self) -> bool:
        return bool(self.model_paths) and bool(self.feature_columns)

    def predict_dummy(self, user: SimpleNamespace, record: SimpleNamespace) -> tuple[float, bool]:
        mapping = map_service_features(user, record, self.feature_columns, strict=False)
        features = _neutralize_missing_family(mapping.features)
        frame = pd.DataFrame([{column: features[column] for column in self.feature_columns}], columns=self.feature_columns)
        probabilities = self.predict_frame(frame)
        probability = float(probabilities[0])
        return probability, probability >= self.threshold

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.available:
            raise RuntimeError(f"{self.disease} screening artifact가 없습니다: {self.artifact_dir}")
        self._load_models()
        x = frame[self.feature_columns].copy()
        for column in self.feature_columns:
            x[column] = pd.to_numeric(x[column], errors="coerce")
        fold_probabilities = [model.predict_proba(x)[:, 1].astype(float) for model in self._models or []]
        if not fold_probabilities:
            return np.zeros(len(x), dtype=float)
        return np.column_stack(fold_probabilities).mean(axis=1)

    def _load_models(self) -> None:
        if self._models is not None:
            return
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("catboost가 필요합니다. uv sync 후 다시 실행하세요.") from exc

        self._models = []
        for model_path in self.model_paths:
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            self._models.append(model)


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    artifacts = {
        disease: ScreeningArtifact(disease, artifact_root / config["artifact_name"] / "screening_catboost")
        for disease, config in DISEASES.items()
    }

    print_policy_truth_table()

    output_frames: list[pd.DataFrame] = []
    summary_lines: list[str] = []

    if args.mode in {"synthetic", "both"}:
        synthetic_rows = run_synthetic_mode(artifacts, args.base_high_level)
        print_synthetic_rows(synthetic_rows)
        summary_lines.extend(synthetic_summary_lines(synthetic_rows))

    if args.mode in {"real", "both"}:
        real_frame, lines = run_real_mode(
            artifacts=artifacts,
            input_path=Path(args.input_path) if args.input_path else None,
            n_samples=args.n_samples,
            seed=args.seed,
            base_high_level=args.base_high_level,
            htn_family_mode=args.htn_family_mode,
        )
        output_frames.append(real_frame)
        summary_lines.extend(lines)
        print("\n".join(lines))

    if args.mode == "fixed":
        summary_lines.extend(policy_truth_table_lines())

    if output_frames:
        output = pd.concat(output_frames, ignore_index=True)
        write_output_csv(output, Path(args.output_csv))
    if summary_lines and args.mode != "synthetic":
        write_summary_md(summary_lines, Path(args.summary_md))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BASIC X1 룰 결과와 KDU screening CatBoost 모델 결과를 비교합니다. "
            "실제 서비스/DB 코드는 수정하지 않는 실험용 스크립트입니다."
        )
    )
    parser.add_argument("--mode", choices=["fixed", "synthetic", "real", "both"], default="real")
    parser.add_argument("--input-path", default=None, help="real mode 입력 CSV/parquet/pkl 파일 또는 디렉터리")
    parser.add_argument("--n-samples", type=int, default=None, help="real mode 샘플 수. 미지정 시 전체 사용")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--summary-md", default=str(DEFAULT_SUMMARY_MD))
    parser.add_argument("--htn-family-mode", choices=["raw", "neutral"], default="neutral")
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="screening artifacts root. 기본값: experiment/ml/kdu/screening/artifacts",
    )
    parser.add_argument(
        "--base-high-level",
        choices=["medium", "high"],
        default="medium",
        help="base 높음 기준. medium이면 MEDIUM/HIGH를 높음으로 보고, high이면 HIGH만 높음으로 봅니다.",
    )
    return parser.parse_args()


def run_real_mode(
    *,
    artifacts: dict[str, ScreeningArtifact],
    input_path: Path | None,
    n_samples: int | None,
    seed: int,
    base_high_level: str,
    htn_family_mode: str,
) -> tuple[pd.DataFrame, list[str]]:
    inputs = resolve_real_inputs(input_path)
    outputs: list[pd.DataFrame] = []
    lines = [
        "\n# Real KNHANES/X1 dual-stage comparison",
        "",
        f"- n_samples: {n_samples if n_samples is not None else 'all'}",
        f"- seed: {seed}",
        f"- htn_family_mode: {htn_family_mode}",
        f"- base_high_level: {base_high_level}",
        "",
    ]

    for disease in DISEASES:
        source_path = inputs[disease]
        raw = read_table(source_path)
        if n_samples is not None and len(raw) > n_samples:
            raw = raw.sample(n=n_samples, random_state=seed).sort_index()
        prepared = prepare_real_features(raw, disease=disease, htn_family_mode=htn_family_mode)
        artifact = artifacts[disease]
        missing = [column for column in artifact.feature_columns if column not in prepared.columns]
        if missing:
            raise ValueError(f"{disease} screening feature 매핑 실패: missing={missing}, input={source_path}")

        probabilities = artifact.predict_frame(prepared[artifact.feature_columns])
        label_column = find_label_column(raw, disease)
        result = build_real_result_frame(
            raw=raw,
            features=prepared,
            disease=disease,
            source_path=source_path,
            probabilities=probabilities,
            threshold=artifact.threshold,
            label_column=label_column,
            base_high_level=base_high_level,
            htn_family_mode=htn_family_mode,
        )
        outputs.append(result)
        lines.extend(real_disease_summary_lines(result, disease=disease, source_path=source_path, label_column=label_column))

    return pd.concat(outputs, ignore_index=True), lines


def resolve_real_inputs(input_path: Path | None) -> dict[str, Path]:
    if input_path is None:
        return dict(DEFAULT_REAL_INPUTS)
    if input_path.is_file():
        return {disease: input_path for disease in DISEASES}
    if input_path.is_dir():
        return {disease: find_disease_file(input_path, disease) for disease in DISEASES}
    raise FileNotFoundError(f"--input-path를 찾을 수 없습니다: {input_path}")


def find_disease_file(root: Path, disease: str) -> Path:
    patterns = {
        "HTN": ["hypertension", "htn", "고혈압"],
        "DM": ["diabetes", "dm", "당뇨"],
        "DL": ["dyslipidemia", "dl", "지질"],
    }[disease]
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in {".csv", ".parquet", ".pkl", ".pickle"}
        and any(pattern.lower() in path.name.lower() for pattern in patterns)
    )
    if not candidates:
        raise FileNotFoundError(f"{disease} 입력 파일을 {root} 아래에서 찾지 못했습니다.")
    return candidates[0]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"지원하지 않는 입력 파일 형식입니다: {path}")


def prepare_real_features(raw: pd.DataFrame, *, disease: str, htn_family_mode: str) -> pd.DataFrame:
    frame = raw.copy()
    apply_column_aliases(frame)
    if "BMI" not in frame.columns and {"키", "체중"}.issubset(frame.columns):
        height_m = pd.to_numeric(frame["키"], errors="coerce") / 100.0
        frame["BMI"] = pd.to_numeric(frame["체중"], errors="coerce") / (height_m**2)
    add_missing_occupation_one_hot(frame)
    missing = [column for column in SERVICE_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(
            "real input에서 service X1 feature를 만들 수 없습니다. "
            f"missing={missing}. --input-path는 x1_service_runtime 계열 CSV를 권장합니다."
        )

    features = frame[SERVICE_FEATURES].copy()
    for column in SERVICE_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in FAMILY_COLUMNS:
        features[column] = features[column].fillna(0.0)
    if disease == "HTN" and htn_family_mode == "neutral":
        # HTN screening artifact가 가족력 feature를 요구하면 neutral value(0.0)를 채운다.
        # 이는 가족력 제외 HTN screening artifact가 준비되기 전까지의 실험용 adapter 전략이다.
        for column in HTN_FAMILY_COLUMNS:
            features[column] = 0.0
    return features


def apply_column_aliases(frame: pd.DataFrame) -> None:
    aliases = {
        "성별": ["gender", "sex", "SEX"],
        "나이": ["age", "AGE"],
        "키": ["height_cm", "HE_ht", "HE_HT"],
        "체중": ["weight_kg", "HE_wt", "HE_WT"],
        "BMI": ["bmi", "HE_BMI"],
        "음주빈도": ["drinking_frequency"],
        "음주량": ["drinking_amount"],
        "현재흡연": ["current_smoking", "smoking_status"],
        "걷기일수": ["walking_days_per_week"],
        "근력운동일수": ["strength_days_per_week"],
    }
    for target, sources in aliases.items():
        if target in frame.columns:
            continue
        source = next((column for column in sources if column in frame.columns), None)
        if source is not None:
            frame[target] = frame[source]


def add_missing_occupation_one_hot(frame: pd.DataFrame) -> None:
    occupation_columns = [column for column in SERVICE_FEATURES if column.startswith("직업_")]
    if all(column in frame.columns for column in occupation_columns):
        return
    for column in occupation_columns:
        frame[column] = 0.0
    frame["직업_작업미상"] = 1.0


def build_real_result_frame(
    *,
    raw: pd.DataFrame,
    features: pd.DataFrame,
    disease: str,
    source_path: Path,
    probabilities: np.ndarray,
    threshold: float,
    label_column: str | None,
    base_high_level: str,
    htn_family_mode: str,
) -> pd.DataFrame:
    base_scores = calculate_base_scores(features, disease)
    base_levels = base_scores.map(score_to_level)
    base_high = base_levels.map(lambda level: is_base_high_value(level, base_high_level))
    screening_high = probabilities >= threshold
    service_band = [POLICY_LABELS[(bool(base), bool(screening))] for base, screening in zip(base_high, screening_high, strict=True)]
    output = pd.DataFrame(
        {
            "case_id": [f"{source_path.stem}:{index}" for index in raw.index],
            "source_path": str(source_path),
            "source_index": raw.index,
            "disease": disease,
            "base_score": base_scores.round(6),
            "base_level": base_levels,
            "base_high": base_high.astype(bool),
            "screening_probability": probabilities,
            "screening_threshold": threshold,
            "screening_high": screening_high.astype(bool),
            "service_band": service_band,
            "htn_family_mode": htn_family_mode,
        }
    )
    if label_column is not None:
        output["label_column"] = label_column
        output["label_value"] = pd.to_numeric(raw[label_column], errors="coerce")
    else:
        output["label_column"] = None
        output["label_value"] = np.nan
    return output


def calculate_base_scores(features: pd.DataFrame, disease: str) -> pd.Series:
    age_adj = np.select(
        [features["나이"] >= 60, features["나이"] >= 45],
        [0.14, 0.08],
        default=0.0,
    )
    lifestyle = (
        (features["현재흡연"] == 1).astype(float) * 0.08
        + (features["음주빈도"] >= 4).astype(float) * 0.06
        + (features["음주량"] >= 3).astype(float) * 0.06
        + (features["걷기일수"] <= 2).astype(float) * 0.05
        + (features["근력운동일수"] == 0).astype(float) * 0.04
    )
    bmi_risk = (features["BMI"] >= 25).astype(float) * 0.10
    if disease == "HTN":
        family = features[HTN_FAMILY_COLUMNS].fillna(0).sum(axis=1).clip(upper=1.0) * 0.16
        return pd.Series(np.minimum(0.20 + age_adj + family + bmi_risk + lifestyle, 0.86), index=features.index)
    if disease == "DM":
        family = features[["당뇨가족력_부", "당뇨가족력_모", "당뇨가족력_형제"]].fillna(0).sum(axis=1).clip(upper=1.0) * 0.16
        return pd.Series(np.minimum(0.20 + age_adj + family + bmi_risk + lifestyle, 0.88), index=features.index)
    if disease == "DL":
        family = (
            features[["고지혈증가족력_부", "고지혈증가족력_모", "고지혈증가족력_형제"]]
            .fillna(0)
            .sum(axis=1)
            .clip(upper=1.0)
            * 0.14
        )
        return pd.Series(np.minimum(0.18 + age_adj + family + bmi_risk + lifestyle, 0.82), index=features.index)
    raise ValueError(f"지원하지 않는 disease입니다: {disease}")


def score_to_level(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def is_base_high_value(level: str, base_high_level: str) -> bool:
    if base_high_level == "high":
        return level == "HIGH"
    return level in {"MEDIUM", "HIGH"}


def find_label_column(frame: pd.DataFrame, disease: str) -> str | None:
    for candidate in DISEASES[disease]["target_candidates"]:
        if candidate in frame.columns:
            return candidate
    return None


def real_disease_summary_lines(
    result: pd.DataFrame,
    *,
    disease: str,
    source_path: Path,
    label_column: str | None,
) -> list[str]:
    label = DISEASES[disease]["label"]
    total = len(result)
    lines = [
        f"## {disease} / {label}",
        "",
        f"- source: `{source_path}`",
        f"- rows: {total}",
        f"- label_column: `{label_column}`" if label_column else "- label_column: 없음",
        "",
        "| base | screening | service_band | count | rate | label_positive_rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for base_high, screening_high in [(False, False), (False, True), (True, False), (True, True)]:
        cell = result[(result["base_high"] == base_high) & (result["screening_high"] == screening_high)]
        label_rate = cell["label_value"].mean() if cell["label_value"].notna().any() else np.nan
        lines.append(
            "| "
            + " | ".join(
                [
                    _state(base_high),
                    _state(screening_high),
                    POLICY_LABELS[(base_high, screening_high)],
                    str(len(cell)),
                    f"{len(cell) / total:.4f}" if total else "0.0000",
                    f"{label_rate:.4f}" if pd.notna(label_rate) else "N/A",
                ]
            )
            + " |"
        )
    base_high_screening_low = float(((result["base_high"]) & (~result["screening_high"])).mean())
    screening_high_base_low = float(((~result["base_high"]) & (result["screening_high"])).mean())
    lines.extend(
        [
            "",
            f"- base high + screening low 비율: {base_high_screening_low:.4f}",
            f"- screening high + base low 비율: {screening_high_base_low:.4f}",
            *interpretation_lines(has_label=label_column is not None, base_high_screening_low=base_high_screening_low),
            "",
        ]
    )
    return lines


def interpretation_lines(*, has_label: bool, base_high_screening_low: float) -> list[str]:
    lines = []
    if base_high_screening_low > 0:
        lines.append("- base high + screening low가 존재하므로 screening을 gate로 쓰면 downgrade 위험이 있습니다.")
    else:
        lines.append("- 이번 샘플에서는 base high + screening low 충돌이 거의 없었습니다.")
    lines.append("- base high + screening low가 있어도 최종 band는 '주의'로 유지해야 합니다.")
    lines.append("- screening high + base low는 '관심 필요' band의 후보 근거로 볼 수 있습니다.")
    if not has_label:
        lines.append("- label이 없어 정답 기반 평가는 하지 못했고, 모델 간 충돌 비율만 확인했습니다.")
    return lines


def write_output_csv(output: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(f"\n[write] CSV: {path}")


def write_summary_md(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[write] summary: {path}")


def run_synthetic_mode(artifacts: dict[str, ScreeningArtifact], base_high_level: str) -> list[SyntheticRow]:
    rows: list[SyntheticRow] = []
    for case in build_dummy_cases():
        for disease, disease_config in DISEASES.items():
            base_score = disease_config["base_score"](case.record, case.user)
            base_risk_level = _risk_level(base_score)
            probability, screening_high = artifacts[disease].predict_dummy(case.user, case.record)
            base_high = _is_base_high(base_risk_level, base_high_level)
            rows.append(
                SyntheticRow(
                    disease=disease,
                    case_id=case.case_id,
                    description=case.description,
                    base_score=base_score,
                    base_risk_level=base_risk_level,
                    base_high=base_high,
                    screening_probability=probability,
                    screening_threshold=artifacts[disease].threshold,
                    screening_high=screening_high,
                    service_band=POLICY_LABELS[(base_high, screening_high)],
                )
            )
    return rows


def build_dummy_cases() -> list[DummyCase]:
    return [
        make_case(
            "low_risk",
            "젊고 BMI/생활습관 위험이 낮은 케이스",
            age=28,
            gender="FEMALE",
            height_cm="165.0",
            weight_kg="55.0",
            occupation_code="OFFICE",
            smoking_status="NEVER",
            drinking_frequency="NONE",
            drinking_amount="NONE",
            walking_days_per_week=6,
            strength_days_per_week=2,
        ),
        make_case(
            "lifestyle_risk",
            "흡연/음주/운동부족 중심 위험 케이스",
            age=42,
            gender="MALE",
            height_cm="173.0",
            weight_kg="72.0",
            occupation_code="OFFICE",
            smoking_status="CURRENT_SMOKER",
            drinking_frequency="WEEKLY_4_PLUS",
            drinking_amount="FIVE_TO_SIX",
            walking_days_per_week=1,
            strength_days_per_week=0,
        ),
        make_case(
            "obesity_risk",
            "BMI 30 이상 비만 위험 케이스",
            age=38,
            gender="MALE",
            height_cm="170.0",
            weight_kg="90.0",
            occupation_code="SERVICE",
            smoking_status="FORMER_SMOKER",
            drinking_frequency="MONTHLY_2_4",
            drinking_amount="THREE_TO_FOUR",
            walking_days_per_week=2,
            strength_days_per_week=0,
        ),
        make_case(
            "age_risk",
            "고령이지만 생활습관 위험은 낮은 케이스",
            age=67,
            gender="FEMALE",
            height_cm="158.0",
            weight_kg="58.0",
            occupation_code="HOMEMAKER",
            smoking_status="NEVER",
            drinking_frequency="NONE",
            drinking_amount="NONE",
            walking_days_per_week=5,
            strength_days_per_week=1,
        ),
        make_case(
            "mixed_risk",
            "나이/BMI/생활습관이 함께 높은 케이스",
            age=58,
            gender="MALE",
            height_cm="168.0",
            weight_kg="82.0",
            occupation_code="MANUAL",
            smoking_status="CURRENT_SMOKER",
            drinking_frequency="DAILY",
            drinking_amount="SEVEN_PLUS",
            walking_days_per_week=0,
            strength_days_per_week=0,
        ),
        make_case(
            "htn_likely_high",
            "고혈압 BASIC high가 나올 법한 케이스",
            age=66,
            gender="MALE",
            height_cm="171.0",
            weight_kg="86.0",
            occupation_code="MANAGER",
            smoking_status="CURRENT_SMOKER",
            drinking_frequency="DAILY",
            drinking_amount="HEAVY",
            walking_days_per_week=1,
            strength_days_per_week=0,
            family_htn="YES",
        ),
        make_case(
            "dm_likely_high",
            "당뇨 BASIC high가 나올 법한 케이스",
            age=63,
            gender="FEMALE",
            height_cm="160.0",
            weight_kg="74.0",
            occupation_code="OFFICE",
            smoking_status="CURRENT_SMOKER",
            drinking_frequency="WEEKLY_2_3",
            drinking_amount="FIVE_TO_SIX",
            walking_days_per_week=1,
            strength_days_per_week=0,
            family_dm="YES",
        ),
        make_case(
            "dl_likely_high",
            "이상지질혈증 BASIC high가 나올 법한 케이스",
            age=64,
            gender="MALE",
            height_cm="175.0",
            weight_kg="88.0",
            occupation_code="OFFICE",
            smoking_status="CURRENT_SMOKER",
            drinking_frequency="WEEKLY_4_PLUS",
            drinking_amount="FIVE_TO_SIX",
            walking_days_per_week=1,
            strength_days_per_week=0,
            family_dyslipidemia="YES",
        ),
    ]


def make_case(
    case_id: str,
    description: str,
    *,
    age: int,
    gender: str,
    height_cm: str,
    weight_kg: str,
    occupation_code: str,
    smoking_status: str,
    drinking_frequency: str,
    drinking_amount: str,
    walking_days_per_week: int,
    strength_days_per_week: int,
    family_htn: str = "NO",
    family_dm: str = "NO",
    family_dyslipidemia: str = "NO",
) -> DummyCase:
    height = Decimal(height_cm)
    weight = Decimal(weight_kg)
    bmi = weight / ((height / Decimal("100")) ** 2)
    user = SimpleNamespace(gender=gender, birthday=_birthday_for_age(age))
    record = SimpleNamespace(
        height_cm=height,
        weight_kg=weight,
        bmi=bmi.quantize(Decimal("0.01")),
        occupation_code=occupation_code,
        family_htn=family_htn,
        family_dm=family_dm,
        family_dyslipidemia=family_dyslipidemia,
        smoking_status=smoking_status,
        drinking_frequency=drinking_frequency,
        drinking_amount=drinking_amount,
        walking_days_per_week=walking_days_per_week,
        strength_days_per_week=strength_days_per_week,
    )
    return DummyCase(case_id=case_id, description=description, user=user, record=record)


def print_synthetic_rows(rows: list[SyntheticRow]) -> None:
    print("\n# Synthetic dummy comparison")
    print("disease | case | base_score | base_level | base_high | screen_p | screen_high | policy")
    print("--- | --- | --- | --- | --- | --- | --- | ---")
    for row in rows:
        print(
            " | ".join(
                [
                    row.disease,
                    row.case_id,
                    f"{row.base_score:.3f}",
                    row.base_risk_level.value,
                    str(row.base_high),
                    f"{row.screening_probability:.3f}/{row.screening_threshold:.2f}",
                    str(row.screening_high),
                    row.service_band,
                ]
            )
        )


def synthetic_summary_lines(rows: list[SyntheticRow]) -> list[str]:
    lines = ["# Synthetic dummy comparison", ""]
    for disease in DISEASES:
        disease_rows = [row for row in rows if row.disease == disease]
        lines.extend([f"## {disease}", ""])
        for base_high, screening_high in [(False, False), (False, True), (True, False), (True, True)]:
            matched = [
                row.case_id for row in disease_rows if row.base_high == base_high and row.screening_high == screening_high
            ]
            lines.append(
                f"- base_{_state(base_high)} + screening_{_state(screening_high)} "
                f"→ {POLICY_LABELS[(base_high, screening_high)]}: {len(matched)}"
            )
        lines.append("")
    return lines


def print_policy_truth_table() -> None:
    print("## Policy truth table")
    print("base | screening | policy")
    print("--- | --- | ---")
    for base_high, screening_high in [(False, False), (False, True), (True, False), (True, True)]:
        print(f"{_state(base_high)} | {_state(screening_high)} | {POLICY_LABELS[(base_high, screening_high)]}")
    print()


def policy_truth_table_lines() -> list[str]:
    lines = ["# Policy truth table", "", "| base | screening | policy |", "|---|---|---|"]
    for base_high, screening_high in [(False, False), (False, True), (True, False), (True, True)]:
        lines.append(f"| {_state(base_high)} | {_state(screening_high)} | {POLICY_LABELS[(base_high, screening_high)]} |")
    lines.append("")
    return lines


def _neutralize_missing_family(features: dict[str, Any]) -> dict[str, Any]:
    for column in FAMILY_COLUMNS:
        if features.get(column) is None:
            features[column] = 0.0
    return features


def _is_base_high(risk_level: RiskLevel, base_high_level: str) -> bool:
    if base_high_level == "high":
        return risk_level == RiskLevel.HIGH
    return risk_level in {RiskLevel.MEDIUM, RiskLevel.HIGH}


def _state(value: bool) -> str:
    return "high" if value else "low"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _birthday_for_age(age: int) -> date:
    today = date.today()
    return date(today.year - age, today.month, today.day)


if __name__ == "__main__":
    main()
