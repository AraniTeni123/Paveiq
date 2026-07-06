"""Build and persist the scoring model artifact.

Stage 3 of the pipeline. There is no labeled footpath-condition
data yet (see ``paveiq.models.heuristic``), so "training" here
means instantiating a :class:`~paveiq.models.heuristic.HeuristicScorer`
with its default (or CLI-overridden) weights and saving it to
``artifacts/``. Once real labels exist, this module is where a real
fit-and-evaluate step replaces the instantiate-and-save step; the
artifact/CLI shape is deliberately built now so that swap is the
only thing that changes.

As a stand-in for training metrics, ``main`` runs the freshly-built
scorer over the latest features and prints a score-distribution
report.

CLI
---
::

    python -m paveiq.models.train
    python -m paveiq.models.train --features path/to/features_with_wards.parquet
    python -m paveiq.models.train --out artifacts/heuristic_scorer_v1.json
    python -m paveiq.models.train --weight-surface-quality 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np

from paveiq.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR
from paveiq.models.heuristic import DEFAULT_WEIGHTS, HeuristicScorer

DEFAULT_ARTIFACT_NAME = "heuristic_scorer_v1.json"


def _find_latest_features_with_wards(processed_dir: Optional[Path] = None) -> Path:
    """Return the newest ``*_with_wards.parquet`` in ``processed_dir``."""
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed dir does not exist: {processed_dir}")
    candidates = sorted(processed_dir.glob("*_with_wards.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"no `*_with_wards.parquet` files in {processed_dir}; run ward_boundaries first"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def score_distribution_report(scores: np.ndarray) -> str:
    """Multi-line summary of a score array, standing in for training metrics."""
    n = len(scores)
    if n == 0:
        return "Score distribution: 0 rows."
    percentiles = np.percentile(scores, [0, 25, 50, 75, 100])
    lines = [
        f"Score distribution ({n:,} segments):",
        f"  mean:              {scores.mean():6.2f}",
        f"  min / p25 / median / p75 / max: "
        f"{percentiles[0]:6.2f} / {percentiles[1]:6.2f} / {percentiles[2]:6.2f} / "
        f"{percentiles[3]:6.2f} / {percentiles[4]:6.2f}",
    ]
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paveiq.models.train",
        description=(
            "Build the heuristic scorer artifact from the default (or "
            "CLI-overridden) weights and print a score-distribution report."
        ),
    )
    p.add_argument(
        "--features",
        default=None,
        help="Path to a *_with_wards.parquet (default: latest in data/processed/).",
    )
    p.add_argument(
        "--out",
        default=None,
        help=f"Output artifact path (default: artifacts/{DEFAULT_ARTIFACT_NAME}).",
    )
    for feature in DEFAULT_WEIGHTS:
        p.add_argument(
            f"--weight-{feature.replace('_', '-')}",
            type=float,
            default=None,
            help=f"Override the {feature} weight (default: {DEFAULT_WEIGHTS[feature]}).",
        )
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    try:
        features_path = Path(args.features) if args.features else _find_latest_features_with_wards()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    weights = dict(DEFAULT_WEIGHTS)
    for feature in DEFAULT_WEIGHTS:
        override = getattr(args, f"weight_{feature}")
        if override is not None:
            weights[feature] = override

    out_path = Path(args.out) if args.out else ARTIFACTS_DIR / DEFAULT_ARTIFACT_NAME

    try:
        scorer = HeuristicScorer(weights=weights)
        scorer.save(out_path)
        features = gpd.read_parquet(features_path)
        scores = scorer.predict(features)
    except Exception as e:
        print(f"error: failed to build scorer: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {out_path}")
    print(score_distribution_report(scores))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
