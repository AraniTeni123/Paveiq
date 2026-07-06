"""Transparent, hand-tuned scorer used until labeled data exists.

Stage 3 of the pipeline needs "labeled footpath-condition data" to
fit a real regression/classifier — none exists in the repo yet (no
citizen reports, no field verification). ``HeuristicScorer`` is a
stand-in: a weighted combination of the four engineered features
from ``build_features.py``, transparent enough to sanity-check by
eye. It implements the same ``predict(df) -> np.ndarray`` shape a
trained sklearn/xgboost model would (see ``models.registry``), so
swapping it out later touches no downstream code.

Missing-data policy
--------------------
Real coverage on the Koramangala bbox: ``surface_quality`` 21.5%,
``width_m`` 0.4%, ``highway_likelihood`` 99.98%, ``sidewalk_presence``
100% (never null by construction). Missing sub-scores are
neutral-imputed to 0.5 rather than excluded-and-renormalized:
renormalizing per row would make two segments with identical
*observed* values score differently depending on which fields
happen to be missing, which is confusing for the advocacy audience
this tool serves. The practical consequence is that with
``width_m`` populated for well under 1% of segments, its weight is
an almost-constant offset for most rows — a known limitation, not
a hidden one. ``n_features_observed`` (0-4) is exposed alongside
the score so a consumer can see how much signal actually went into
it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from paveiq.config import SCORE_MAX, SCORE_MIN

MODEL_TYPE = "heuristic_v1"

# Raw feature columns this scorer reads, in the order sub-scores are
# computed (also the columns `n_features_observed` counts over).
FEATURE_COLS = ("surface_quality", "sidewalk_presence", "width_m", "highway_likelihood")

DEFAULT_WEIGHTS = {
    "surface_quality": 0.35,
    "sidewalk_presence": 0.30,
    "width_m": 0.15,
    "highway_likelihood": 0.20,
}

DEFAULT_SIDEWALK_SCORE_MAP = {
    "explicit_present": 1.0,
    "implicit_present": 0.9,
    "likely_present": 0.5,
    "unlikely": 0.2,
    "explicit_absent": 0.0,
}

DEFAULT_WIDTH_MIN_M = 0.5
DEFAULT_WIDTH_MAX_M = 2.0
DEFAULT_NEUTRAL_IMPUTE = 0.5


class HeuristicScorer:
    """Weighted-sum scorer over ``FEATURE_COLS``. See module docstring."""

    def __init__(
        self,
        weights: Optional[dict] = None,
        sidewalk_score_map: Optional[dict] = None,
        width_min_m: float = DEFAULT_WIDTH_MIN_M,
        width_max_m: float = DEFAULT_WIDTH_MAX_M,
        neutral_impute: float = DEFAULT_NEUTRAL_IMPUTE,
        score_min: float = SCORE_MIN,
        score_max: float = SCORE_MAX,
    ):
        weights = dict(weights) if weights is not None else dict(DEFAULT_WEIGHTS)
        missing = set(FEATURE_COLS) - set(weights)
        if missing:
            raise ValueError(f"weights is missing entries for: {sorted(missing)}")
        self.weights = weights
        self.sidewalk_score_map = (
            dict(sidewalk_score_map) if sidewalk_score_map is not None else dict(DEFAULT_SIDEWALK_SCORE_MAP)
        )
        self.width_min_m = width_min_m
        self.width_max_m = width_max_m
        self.neutral_impute = neutral_impute
        self.score_min = score_min
        self.score_max = score_max

    # --- sub-scores ----------------------------------------------------

    def _surface_subscore(self, df: pd.DataFrame) -> pd.Series:
        return df["surface_quality"].fillna(self.neutral_impute)

    def _sidewalk_subscore(self, df: pd.DataFrame) -> pd.Series:
        mapped = df["sidewalk_presence"].map(self.sidewalk_score_map)
        return mapped.fillna(self.neutral_impute)

    def _width_subscore(self, df: pd.DataFrame) -> pd.Series:
        span = self.width_max_m - self.width_min_m
        normalized = (df["width_m"] - self.width_min_m) / span
        return normalized.clip(0.0, 1.0).fillna(self.neutral_impute)

    def _highway_subscore(self, df: pd.DataFrame) -> pd.Series:
        return df["highway_likelihood"].fillna(self.neutral_impute)

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the per-row sub-scores, ``n_features_observed``, and ``score``.

        Not part of the ``Scorer`` protocol — callers that want
        transparency (``score_city.py``, the dashboard's what-if
        panel) check ``hasattr(scorer, "explain")`` so a future
        opaque model degrades gracefully to bare ``predict()``.
        """
        self._validate_columns(df)
        surface_score = self._surface_subscore(df)
        sidewalk_score = self._sidewalk_subscore(df)
        width_score = self._width_subscore(df)
        highway_score = self._highway_subscore(df)

        n_features_observed = (
            df["surface_quality"].notna().astype(int)
            + df["sidewalk_presence"].notna().astype(int)
            + df["width_m"].notna().astype(int)
            + df["highway_likelihood"].notna().astype(int)
        )

        weighted = (
            self.weights["surface_quality"] * surface_score
            + self.weights["sidewalk_presence"] * sidewalk_score
            + self.weights["width_m"] * width_score
            + self.weights["highway_likelihood"] * highway_score
        )
        score = (self.score_min + (self.score_max - self.score_min) * weighted).clip(
            self.score_min, self.score_max
        )

        return pd.DataFrame(
            {
                "surface_score": surface_score,
                "sidewalk_score": sidewalk_score,
                "width_score": width_score,
                "highway_score": highway_score,
                "n_features_observed": n_features_observed.astype("int8"),
                "score": score.astype("float32"),
            },
            index=df.index,
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return a ``float`` score in ``[score_min, score_max]`` per row."""
        return self.explain(df)["score"].to_numpy()

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"input is missing required columns: {missing}")

    # --- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "model_type": MODEL_TYPE,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "weights": self.weights,
            "sidewalk_score_map": self.sidewalk_score_map,
            "width_score_min_m": self.width_min_m,
            "width_score_max_m": self.width_max_m,
            "neutral_impute": self.neutral_impute,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "HeuristicScorer":
        data = json.loads(Path(path).read_text())
        model_type = data.get("model_type")
        if model_type != MODEL_TYPE:
            raise ValueError(f"expected model_type={MODEL_TYPE!r}, got {model_type!r} in {path}")
        return cls(
            weights=data["weights"],
            sidewalk_score_map=data["sidewalk_score_map"],
            width_min_m=data["width_score_min_m"],
            width_max_m=data["width_score_max_m"],
            neutral_impute=data["neutral_impute"],
            score_min=data.get("score_min", SCORE_MIN),
            score_max=data.get("score_max", SCORE_MAX),
        )
