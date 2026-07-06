"""Apply the scoring model to the full city network.

Stage 4 of the pipeline. Reads the scorer artifact produced by
``models.train`` and the latest ``*_with_wards.parquet``, computes a
``score`` (and, for scorers that expose it, sub-score/confidence
columns) per segment, and writes a scored Parquet that the
dashboard reads directly.

Output schema
-------------
All columns from the input ``*_with_wards.parquet`` (14, including
the ward columns), plus ``score`` (float32, clipped to
``[SCORE_MIN, SCORE_MAX]``). If the loaded scorer exposes an
``explain(df)`` method (as :class:`~paveiq.models.heuristic.HeuristicScorer`
does), the sub-score and ``n_features_observed`` columns it returns
are merged in too, so a future opaque model that lacks ``explain``
degrades gracefully to the bare ``score`` column.

CLI
---
::

    python -m paveiq.scoring.score_city
    python -m paveiq.scoring.score_city --features path/to/features_with_wards.parquet
    python -m paveiq.scoring.score_city --model artifacts/heuristic_scorer_v1.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from paveiq.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, SCORE_MAX, SCORE_MIN
from paveiq.models.registry import Scorer, load_scorer


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


def _find_latest_artifact(artifacts_dir: Optional[Path] = None) -> Path:
    """Return the newest ``*_scorer*.json`` in ``artifacts_dir``."""
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"artifacts dir does not exist: {artifacts_dir}")
    candidates = sorted(artifacts_dir.glob("*_scorer*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no `*_scorer*.json` files in {artifacts_dir}; run models.train first"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def score_dataframe(df: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    """Return ``df`` with a ``score`` column (and sub-scores, if available) added."""
    out = df.copy()
    if hasattr(scorer, "explain"):
        explained = scorer.explain(df)
        for col in explained.columns:
            out[col] = explained[col]
    else:
        out["score"] = np.clip(scorer.predict(df), SCORE_MIN, SCORE_MAX).astype("float32")
    return out


def scoring_report(scored_df: pd.DataFrame) -> str:
    """Multi-line summary of a scored GeoDataFrame."""
    n = len(scored_df)
    if n == 0:
        return "Scoring summary: 0 rows."
    scores = scored_df["score"]
    lines = [
        f"Scoring summary ({n:,} segments):",
        f"  mean score:                 {scores.mean():6.2f}",
        f"  median score:                {scores.median():6.2f}",
        f"  min / max score:             {scores.min():6.2f} / {scores.max():6.2f}",
    ]
    if "n_features_observed" in scored_df.columns:
        counts = scored_df["n_features_observed"].value_counts().sort_index()
        lines.append("  segments by n_features_observed:")
        for k, v in counts.items():
            lines.append(f"    {k}: {v:,}  ({100 * v / n:5.2f}%)")
    if "highway" in scored_df.columns:
        by_highway = scored_df.groupby("highway")["score"].mean().sort_values()
        lines.append("  mean score by highway (worst 5):")
        for highway, mean_score in by_highway.head(5).items():
            lines.append(f"    {highway:<15} {mean_score:6.2f}")
    return "\n".join(lines)


def score_and_save(
    features_path: Path,
    model_path: Path,
    out_dir: Optional[Path] = None,
) -> Path:
    """End-to-end: load features + scorer, score, write Parquet.

    Output path: ``<out_dir>/<features_stem>_scored.parquet``.
    """
    out_dir = Path(out_dir) if out_dir is not None else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    features = gpd.read_parquet(features_path)
    scorer = load_scorer(model_path)
    scored = score_dataframe(features, scorer)

    out_name = features_path.stem + "_scored.parquet"
    out_path = out_dir / out_name
    scored.to_parquet(out_path)
    return out_path


# --- CLI -------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paveiq.scoring.score_city",
        description=(
            "Apply the scoring model artifact to the latest features_with_wards "
            "Parquet and write a scored Parquet."
        ),
    )
    p.add_argument(
        "--features",
        default=None,
        help="Path to a *_with_wards.parquet (default: latest in data/processed/).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Path to a scorer artifact (default: latest *_scorer*.json in artifacts/).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: data/processed/).",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else None

    try:
        features_path = Path(args.features) if args.features else _find_latest_features_with_wards()
        model_path = Path(args.model) if args.model else _find_latest_artifact()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        out_path = score_and_save(features_path, model_path, out_dir=out_dir)
    except Exception as e:
        print(f"error: failed to score city: {e}", file=sys.stderr)
        return 1

    scored = gpd.read_parquet(out_path)
    print(f"Wrote {out_path}")
    print(scoring_report(scored))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
