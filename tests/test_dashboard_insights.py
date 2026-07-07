"""Offline tests for ``paveiq.dashboard.insights``."""

from __future__ import annotations

import pandas as pd
import pytest

from paveiq.dashboard import insights


def _scored_df():
    return pd.DataFrame(
        {
            "osmid": ["1", "2", "3", "4"],
            "ward_id": ["A", "A", "B", "B"],
            "score": [90.0, 30.0, 60.0, 10.0],
            "surface_score": [1.0, 0.2, 0.6, 0.2],
            "sidewalk_score": [1.0, 0.2, 0.6, 0.0],
            "width_score": [0.5, 0.5, 0.5, 0.5],
            "highway_score": [1.0, 0.0, 0.5, 0.0],
        }
    )


def _ward_summary_df():
    # Worst-first, matching data.ward_summary's contract.
    return pd.DataFrame(
        {
            "ward_id": ["B", "A"],
            "ward_name": ["Beta", "Alpha"],
            "mean_score": [35.0, 60.0],
            "segment_count": [2, 2],
            "pct_poor": [50.0, 50.0],
            "total_length_m": [20.0, 10.0],
        }
    )


# --- header_insight ------------------------------------------------------


def test_header_insight_reports_gap_and_dominant_factor():
    text = insights.header_insight(_scored_df(), _ward_summary_df())
    assert "Beta" in text
    # city mean = (90+30+60+10)/4 = 47.5; worst (Beta) mean_score = 35.0; gap = 12.5
    assert "12.5" in text
    assert "47.5" in text
    # Beta's rows (3,4) deviations vs city mean: surface=0.10, sidewalk=0.15,
    # width=0.0, highway=0.125 -> sidewalk_score is the largest.
    assert "sidewalk presence" in text


def test_header_insight_degrades_gracefully_without_subscore_columns():
    scored = _scored_df()[["osmid", "ward_id", "score"]]
    text = insights.header_insight(scored, _ward_summary_df())
    assert "Beta" in text
    assert "driven primarily by" not in text


def test_header_insight_handles_empty_ward_summary():
    empty = pd.DataFrame({"ward_id": [], "ward_name": [], "mean_score": []})
    text = insights.header_insight(_scored_df(), empty)
    assert "No" in text


# --- leaderboard_insight ---------------------------------------------------


def test_leaderboard_insight_reports_count_below_midpoint_and_spread():
    text = insights.leaderboard_insight(_ward_summary_df())
    assert "1 of 2 wards" in text  # only Beta (35.0) is below GOOD_SCORE_THRESHOLD=50
    assert "Beta" in text and "35.0" in text
    assert "Alpha" in text and "60.0" in text


def test_leaderboard_insight_handles_empty():
    empty = pd.DataFrame({"ward_id": [], "ward_name": [], "mean_score": []})
    text = insights.leaderboard_insight(empty)
    assert "No" in text


# --- ward_rank / segment_rank -----------------------------------------------


def test_ward_rank_worst_is_rank_1():
    ws = _ward_summary_df()
    assert insights.ward_rank(ws, "B") == 1
    assert insights.ward_rank(ws, "A") == 2


def test_ward_rank_unsorted_input_still_correct():
    """ward_rank should sort internally, not trust the caller's row order."""
    ws = _ward_summary_df().sort_values("ward_id")  # now alphabetical, not worst-first
    assert insights.ward_rank(ws, "B") == 1
    assert insights.ward_rank(ws, "A") == 2


def test_ward_rank_missing_raises():
    with pytest.raises(ValueError, match="not found"):
        insights.ward_rank(_ward_summary_df(), "Z")


def test_segment_rank_worst_is_rank_1():
    scored = _scored_df()
    assert insights.segment_rank(scored, "4") == 1  # score 10.0, worst
    assert insights.segment_rank(scored, "1") == 4  # score 90.0, best


def test_segment_rank_missing_raises():
    with pytest.raises(ValueError, match="not found"):
        insights.segment_rank(_scored_df(), "nonexistent")


# --- rank_change_insight ---------------------------------------------------


def test_rank_change_insight_improvement():
    text = insights.rank_change_insight("HAL Airport (93)", "ward", before_rank=1, after_rank=14, total=68)
    assert "HAL Airport (93)" in text
    assert "from rank 1 to rank 14 of 68 wards" in text


def test_rank_change_insight_segment_scope_uses_segments_noun():
    text = insights.rank_change_insight("segment 12345", "segment", before_rank=5, after_rank=3, total=21947)
    assert "segments" in text


def test_rank_change_insight_no_change():
    text = insights.rank_change_insight("Alpha", "ward", before_rank=10, after_rank=10, total=68)
    assert "would not change" in text
    assert "10 of 68 wards" in text
