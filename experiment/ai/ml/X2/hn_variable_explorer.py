"""EDA helpers for health exam source variables and stage targets.

This script is separated from service runtime code. It is intended for local
analysis notebooks or one-off CSV exploration only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ai_worker.ml.X2.health_stage_classifier import SOURCE_VARIABLE_MAP

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR.parents[0] / "data"
OUTPUT_DIR = MODULE_DIR / "outputs"

TARGET_MAP = {
    "HTN": "고혈압 단계",
    "DM": "당뇨병 범위 단계",
    "DL": "이상지질혈증 수치 단계",
    "OBE": "비만 단계",
    "ANEM": "빈혈 범위 단계",
}


def load_csv(path: Path) -> Any:
    import pandas as pd

    return pd.read_csv(path)


def summarize_columns(csv_path: Path) -> dict[str, Any]:
    data = load_csv(csv_path)
    source_columns = [column for column in data.columns if column in SOURCE_VARIABLE_MAP]
    return {
        "rows": int(len(data)),
        "columns": list(data.columns),
        "known_source_columns": source_columns,
        "mapped_service_fields": {column: SOURCE_VARIABLE_MAP[column] for column in source_columns},
    }


def plot_stage_distributions(csv_path: Path, target_columns: list[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    data = load_csv(csv_path)
    valid_targets = [column for column in target_columns if column in data.columns]
    if not valid_targets:
        raise ValueError("CSV에 지정한 target column이 없습니다.")

    fig, axes = plt.subplots(len(valid_targets), 1, figsize=(10, max(4, 3 * len(valid_targets))))
    if len(valid_targets) == 1:
        axes = [axes]

    for axis, target in zip(axes, valid_targets, strict=True):
        counts = data[target].value_counts(dropna=False).sort_index()
        axis.bar([str(index) for index in counts.index], counts.values)
        axis.set_title(f"{target} stage distribution")
        axis.set_xlabel("stage")
        axis.set_ylabel("count")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_stage_boxplots(csv_path: Path, target_column: str, value_columns: list[str], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    data = load_csv(csv_path)
    if target_column not in data.columns:
        raise ValueError(f"CSV에 target column이 없습니다: {target_column}")
    valid_values = [column for column in value_columns if column in data.columns]
    if not valid_values:
        raise ValueError("CSV에 지정한 value column이 없습니다.")

    fig, axes = plt.subplots(len(valid_values), 1, figsize=(10, max(4, 3 * len(valid_values))))
    if len(valid_values) == 1:
        axes = [axes]

    for axis, value_column in zip(axes, valid_values, strict=True):
        grouped = [
            group[value_column].dropna().values
            for _, group in data[[target_column, value_column]].dropna(subset=[target_column]).groupby(target_column)
        ]
        labels = [str(label) for label in sorted(data[target_column].dropna().unique())]
        axis.boxplot(grouped, tick_labels=labels, showfliers=False)
        axis.set_title(f"{value_column} by {target_column}")
        axis.set_xlabel("stage")
        axis.set_ylabel(value_column)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_yearly_trends(csv_path: Path, year_column: str, target_columns: list[str], output_path: Path) -> None:
    import math

    import matplotlib.pyplot as plt

    data = load_csv(csv_path)
    valid_targets = [column for column in target_columns if column in data.columns]
    if year_column not in data.columns:
        raise ValueError(f"CSV에 year column이 없습니다: {year_column}")
    if not valid_targets:
        raise ValueError("CSV에 지정한 target column이 없습니다.")

    columns = min(2, len(valid_targets))
    rows = math.ceil(len(valid_targets) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 4 * rows), squeeze=False)
    flat_axes = axes.flatten()

    for axis, target in zip(flat_axes, valid_targets, strict=False):
        trend = data.groupby([year_column, target]).size().unstack(fill_value=0)
        trend.plot(ax=axis)
        axis.set_title(f"{TARGET_MAP.get(target, target)} yearly trend")
        axis.set_xlabel("year")
        axis.set_ylabel("count")

    for axis in flat_axes[len(valid_targets) :]:
        axis.set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore health exam source variables and stage targets.")
    parser.add_argument("csv", type=Path, nargs="?", help="CSV file to inspect")
    parser.add_argument("--summary", action="store_true", help="Print source variable mapping summary")
    parser.add_argument("--targets", nargs="*", default=list(TARGET_MAP), help="Target columns for stage plots")
    parser.add_argument("--year-column", default="year", help="Year column for trend plots")
    args = parser.parse_args()

    if args.csv is None:
        print(f"DATA_DIR={DATA_DIR}")
        print(f"OUTPUT_DIR={OUTPUT_DIR}")
        print(f"TARGET_MAP={TARGET_MAP}")
        return

    csv_path = args.csv if args.csv.is_absolute() else DATA_DIR / args.csv
    if args.summary:
        print(summarize_columns(csv_path))
        return

    plot_stage_distributions(csv_path, args.targets, OUTPUT_DIR / "stage_distributions.png")
    plot_yearly_trends(csv_path, args.year_column, args.targets, OUTPUT_DIR / "yearly_trends.png")
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
