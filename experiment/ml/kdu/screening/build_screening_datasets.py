"""Build only the KDU X1 service screening candidate datasets.

Label policy and feature-list definitions are screening-local. Raw KNHANES reading
and service-schema transformation still delegate to the project base builder so
the screening package remains project-internal reproducible rather than fully
standalone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from screening_config import (
    AUXILIARY_TARGETS,
    MAIN_SCREENING_TARGETS,
    SCREENING_AND_AUXILIARY_TARGETS,
    SERVICE_POLICY_ALLOWED,
)
from screening_feature_set import FORBIDDEN_LEAKAGE_COLUMNS, X1_SERVICE_RUNTIME_FEATURES
from screening_label_policy import SOURCE_FILES_HN13_24, build_label, get_label_definition


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    raise RuntimeError("Cannot find repo root. Run this script inside AH_03_03.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KDU screening datasets only.")
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Build only the three main screening datasets, excluding strict auxiliary labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the build plan without writing processed datasets.",
    )
    return parser.parse_args()


def _load_base_builder(repo_root: Path) -> Any:
    script_dir = repo_root / "experiment" / "ml" / "kdu" / "scripts"
    src_dir = repo_root / "experiment" / "ml" / "kdu" / "src"
    for path in (script_dir, src_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import build_x1_service_basic_datasets as base_builder

    return base_builder


def _metadata_for(target: Any, dataset: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    positive_count = int((dataset["target"] == 1).sum())
    negative_count = int((dataset["target"] == 0).sum())
    return {
        "disease": target.disease,
        "feature_set": "x1_service_runtime",
        "data_version": target.data_version,
        "label_policy": target.label_policy,
        "policy_suffix": target.policy_suffix,
        "role": target.role,
        "source_years": "HN13-HN24",
        "row_count": int(len(dataset)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": round(positive_count / len(dataset), 4) if len(dataset) else None,
        "feature_count": len(X1_SERVICE_RUNTIME_FEATURES),
        "feature_columns": X1_SERVICE_RUNTIME_FEATURES,
        "excluded_leakage_columns": FORBIDDEN_LEAKAGE_COLUMNS,
        "recommended_threshold": target.recommended_threshold,
        "recommended_gray_zone": target.recommended_gray_zone,
        "service_expression": target.service_expression,
        "service_policy_allowed": SERVICE_POLICY_ALLOWED,
        "medical_safety_note": "Screening/check-needed label only. Do not use confirmed diagnosis wording.",
        "output_path": str(output_path),
    }


def _runtime_columns() -> list[str]:
    return ["survey_year", *X1_SERVICE_RUNTIME_FEATURES, "target"]


def _build_target(repo_root: Path, base_builder: Any, target: Any) -> dict[str, Any]:
    raw_dir = repo_root / "experiment" / "ml" / "kdu" / "data" / "raw"
    definition = get_label_definition(target.disease, target.policy_suffix)
    frames: list[pd.DataFrame] = []
    yearly_rows: list[dict[str, Any]] = []

    for file_name in SOURCE_FILES_HN13_24:
        path = raw_dir / file_name
        year = base_builder.detect_year(file_name)
        available = base_builder.available_columns(path)
        needed = [*base_builder.RAW_CANDIDATE_COLUMNS, *definition.source_columns]
        usecols = [column for column in needed if column in available]
        raw = base_builder.read_columns(path, usecols)
        label = build_label(raw, target.disease, target.policy_suffix)
        valid = label.notna()
        service_frame = base_builder.transform_to_service_schema(raw.loc[valid].copy())
        service_frame.insert(0, "survey_year", year)
        service_frame["target"] = label.loc[valid].astype(bool).astype(int).to_numpy()
        service_frame = service_frame.reindex(columns=_runtime_columns())
        missing_features = [column for column in X1_SERVICE_RUNTIME_FEATURES if column not in service_frame.columns]
        if missing_features:
            raise ValueError(f"{target.disease}/{target.policy_suffix} missing runtime features: {missing_features}")
        frames.append(service_frame)

        positive_count = int((service_frame["target"] == 1).sum())
        yearly_rows.append(
            {
                "disease": target.disease,
                "label_policy": target.label_policy,
                "policy_suffix": target.policy_suffix,
                "year": year,
                "file_name": file_name,
                "rows": int(len(service_frame)),
                "positive_count": positive_count,
                "negative_count": int((service_frame["target"] == 0).sum()),
                "positive_rate": round(positive_count / len(service_frame), 4) if len(service_frame) else None,
            }
        )

    dataset = pd.concat(frames, ignore_index=True)
    output_dir = repo_root / "experiment" / "ml" / "kdu" / "data" / "processed" / target.data_version
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / target.file_name
    dataset.to_csv(output_path, index=False)
    metadata = _metadata_for(target, dataset, output_path)
    metadata["positive_condition"] = definition.positive_condition
    metadata["label_source_columns"] = list(definition.source_columns)
    metadata["yearly_distribution"] = yearly_rows
    (output_dir / "screening_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd().resolve())
    targets = MAIN_SCREENING_TARGETS if args.main_only else SCREENING_AND_AUXILIARY_TARGETS

    print(f"[SCREENING DATASET BUILD] targets={len(targets)} main_only={args.main_only}", flush=True)
    for target in targets:
        print(
            f"- {target.disease}/{target.label_policy}: data_version={target.data_version} role={target.role}",
            flush=True,
        )
    if args.dry_run:
        return 0

    base_builder = _load_base_builder(repo_root)
    rows = [_build_target(repo_root, base_builder, target) for target in targets]

    summary = pd.DataFrame(rows)
    summary_path = (
        repo_root / "experiment" / "ml" / "kdu" / "outputs" / "summary" / "screening_dataset_build_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"[SCREENING DATASET BUILD] summary={summary_path}", flush=True)
    print(
        summary[["disease", "label_policy", "role", "row_count", "positive_rate", "feature_count"]].to_string(
            index=False
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
