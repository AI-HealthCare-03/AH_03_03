"""Train KDU X1 service screening CatBoost candidates.

This is a narrow wrapper around the existing training script. It only exposes
the screening candidates and optional strict auxiliary labels.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from screening_config import AUXILIARY_TARGETS, FEATURE_SET, MAIN_SCREENING_TARGETS, TRAINING_MODEL_NAME


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    raise RuntimeError("Cannot find repo root. Run this script inside AH_03_03.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KDU screening CatBoost models.")
    parser.add_argument(
        "--include-auxiliary",
        action="store_true",
        help="Also train strict/high-attention auxiliary labels.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable SQLite registry logging in the underlying training script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print training commands without running model training.",
    )
    return parser.parse_args()


def command_for(repo_root: Path, target: object, no_tracking: bool) -> list[str]:
    script = repo_root / "experiment" / "ml" / "kdu" / "scripts" / "train_diagnosis_baseline.py"
    command = [
        sys.executable,
        str(script),
        "--disease",
        target.disease,
        "--model",
        TRAINING_MODEL_NAME,
        "--feature-set",
        FEATURE_SET,
        "--data-version",
        target.data_version,
        "--missing-policy",
        "native",
    ]
    if no_tracking:
        command.append("--no-tracking")
    return command


def main() -> int:
    args = parse_args()
    repo_root = find_repo_root(Path.cwd().resolve())
    targets = list(MAIN_SCREENING_TARGETS)
    if args.include_auxiliary:
        targets.extend(AUXILIARY_TARGETS)

    print(f"[SCREENING CATBOOST TRAIN] targets={len(targets)} include_auxiliary={args.include_auxiliary}", flush=True)
    for index, target in enumerate(targets, start=1):
        command = command_for(repo_root, target, args.no_tracking)
        print(f"[{index}/{len(targets)}] {target.disease}/{target.label_policy}", flush=True)
        print("  " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
