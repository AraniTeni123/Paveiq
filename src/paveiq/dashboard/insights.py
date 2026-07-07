"""Auto-generated interpretive sentences, computed live from the scored data.

Pure functions — no ``streamlit`` import — same convention as
``data.py``/``mapview.py``/``whatif.py``. Every sentence here is computed
from the actual DataFrame passed in; nothing is a hardcoded template with
placeholder numbers.
"""

from __future__ import annotations

import pandas as pd

from paveiq.dashboard.theme import GOOD_SCORE_THRESHOLD

# The four heuristic sub-score columns HeuristicScorer.explain() adds to a
# scored table, mapped to a human label for the "driven primarily by"
# clause in header_insight().
_SUBSCORE_LABELS = {
    "surface_score": "surface quality",
    "sidewalk_score": "sidewalk presence",
    "width_score": "path width",
    "highway_score": "road-type confidence",
}


def header_insight(scored_df: pd.DataFrame, ward_summary_df: pd.DataFrame) -> str:
    """The worst ward's gap vs. the citywide mean, and what's driving it.

    "Driving it" = the sub-score column with the largest negative
    deviation between the worst ward's mean and the citywide mean for
    that same column. Degrades gracefully (drops the "driven by" clause)
    if the sub-score columns aren't present — e.g. a future non-heuristic
    scorer without ``.explain()``.
    """
    if len(ward_summary_df) == 0 or len(scored_df) == 0:
        return "No scored data available."

    worst = ward_summary_df.iloc[0]
    city_mean = scored_df["score"].mean()
    gap = city_mean - worst["mean_score"]

    dominant_clause = ""
    subscore_cols = [c for c in _SUBSCORE_LABELS if c in scored_df.columns]
    if subscore_cols and "ward_id" in scored_df.columns:
        ward_rows = scored_df.loc[scored_df["ward_id"] == worst["ward_id"]]
        if len(ward_rows) > 0:
            deviations = {col: scored_df[col].mean() - ward_rows[col].mean() for col in subscore_cols}
            dominant_col = max(deviations, key=deviations.get)
            dominant_clause = f" — driven primarily by weak {_SUBSCORE_LABELS[dominant_col]}"

    return (
        f"The worst-scoring ward, {worst['ward_name']}, sits {gap:.1f} points below "
        f"the citywide mean of {city_mean:.1f}{dominant_clause}."
    )


def leaderboard_insight(ward_summary_df: pd.DataFrame) -> str:
    """How many wards fall below the scale midpoint, plus the best/worst spread.

    ``ward_summary_df`` is expected worst-first (as ``data.ward_summary``
    returns it) — ``.iloc[0]``/``.iloc[-1]`` are the worst/best wards.
    """
    if len(ward_summary_df) == 0:
        return "No ward data available."

    total = len(ward_summary_df)
    n_below = int((ward_summary_df["mean_score"] < GOOD_SCORE_THRESHOLD).sum())
    worst = ward_summary_df.iloc[0]
    best = ward_summary_df.iloc[-1]
    return (
        f"{n_below} of {total} wards score below the citywide midpoint of "
        f"{GOOD_SCORE_THRESHOLD} — {worst['ward_name']} is lowest at {worst['mean_score']:.1f}, "
        f"{best['ward_name']} highest at {best['mean_score']:.1f}."
    )


def ward_rank(ward_summary_df: pd.DataFrame, ward_id) -> int:
    """1-indexed rank of ``ward_id`` (1 = worst), matching the displayed worst-first table."""
    sorted_df = ward_summary_df.sort_values("mean_score", ascending=True).reset_index(drop=True)
    matches = sorted_df.index[sorted_df["ward_id"] == ward_id]
    if len(matches) == 0:
        raise ValueError(f"ward_id {ward_id!r} not found in ward_summary_df")
    return int(matches[0]) + 1


def segment_rank(scored_df: pd.DataFrame, osmid) -> int:
    """1-indexed rank of ``osmid`` (1 = worst) among all segments by score."""
    sorted_df = scored_df.sort_values("score", ascending=True).reset_index(drop=True)
    matches = sorted_df.index[sorted_df["osmid"] == osmid]
    if len(matches) == 0:
        raise ValueError(f"osmid {osmid!r} not found in scored_df")
    return int(matches[0]) + 1


def rank_change_insight(label: str, scope: str, before_rank: int, after_rank: int, total: int) -> str:
    """e.g. "This hypothetical change would move HAL Airport (93) from rank 1 to rank 14 of 68 wards."."""
    noun = "wards" if scope == "ward" else "segments"
    if before_rank == after_rank:
        return f"This hypothetical change would not change {label}'s rank ({before_rank} of {total} {noun})."
    return (
        f"This hypothetical change would move {label} from rank {before_rank} "
        f"to rank {after_rank} of {total} {noun}."
    )
