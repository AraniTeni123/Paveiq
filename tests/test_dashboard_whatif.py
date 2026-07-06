"""Offline tests for ``paveiq.dashboard.whatif``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paveiq.dashboard import whatif as wi
from paveiq.models.heuristic import HeuristicScorer


def _df():
    return pd.DataFrame(
        {
            "osmid": ["1", "2", "3", "4"],
            "ward_id": ["A", "A", "B", "B"],
            "surface_quality": [0.2, 1.0, 0.2, np.nan],
            "sidewalk_presence": ["explicit_absent", "explicit_present", "unlikely", "explicit_absent"],
            "width_m": [0.5, 2.0, 0.5, 0.5],
            "highway_likelihood": [0.5, 1.0, 0.3, 0.3],
        }
    )


# --- select_scope ------------------------------------------------------


def test_select_scope_segment():
    out = wi.select_scope(_df(), "segment", "2")
    assert list(out["osmid"]) == ["2"]


def test_select_scope_ward():
    out = wi.select_scope(_df(), "ward", "B")
    assert list(out["osmid"]) == ["3", "4"]


def test_select_scope_invalid_raises():
    with pytest.raises(ValueError, match="scope must be"):
        wi.select_scope(_df(), "city", "X")


# --- apply_hypothetical: segment scope --------------------------------


def test_apply_hypothetical_segment_surface_upgrade():
    modified, affected = wi.apply_hypothetical(
        _df(), {"surface_upgrade": 1.0}, scope="segment", target="1"
    )
    assert list(modified["surface_quality"]) == [1.0]
    assert affected == {"surface_upgrade": 1}


def test_apply_hypothetical_already_matching_target_not_counted():
    """Segment 2 is already surface_quality=1.0 -> upgrading to 1.0 affects 0 rows."""
    modified, affected = wi.apply_hypothetical(
        _df(), {"surface_upgrade": 1.0}, scope="segment", target="2"
    )
    assert affected == {"surface_upgrade": 0}
    assert list(modified["surface_quality"]) == [1.0]


def test_apply_hypothetical_missing_value_counts_as_affected():
    """Segment 4 has NaN surface_quality; upgrading it should count as affected."""
    modified, affected = wi.apply_hypothetical(
        _df(), {"surface_upgrade": 1.0}, scope="segment", target="4"
    )
    assert affected == {"surface_upgrade": 1}
    assert modified["surface_quality"].iloc[0] == 1.0


# --- apply_hypothetical: ward scope (bulk simulate) -----------------------


def test_apply_hypothetical_ward_bulk_applies_to_all_matching_segments():
    modified, affected = wi.apply_hypothetical(
        _df(), {"add_sidewalk": "explicit_present"}, scope="ward", target="B"
    )
    assert len(modified) == 2
    assert list(modified["sidewalk_presence"]) == ["explicit_present", "explicit_present"]
    # Both segment 3 (unlikely) and 4 (explicit_absent) differ from target -> both affected.
    assert affected == {"add_sidewalk": 2}


def test_apply_hypothetical_ward_scope_only_changes_differing_rows():
    """Ward A: segment 1 has highway_likelihood 0.5, segment 2 already 1.0.
    pedestrianize -> only segment 1 should be counted/changed."""
    modified, affected = wi.apply_hypothetical(
        _df(), {"pedestrianize": 1.0}, scope="ward", target="A"
    )
    assert affected == {"pedestrianize": 1}
    assert list(modified["highway_likelihood"]) == [1.0, 1.0]


def test_apply_hypothetical_widen_path():
    modified, affected = wi.apply_hypothetical(
        _df(), {"widen_path": 1.8}, scope="segment", target="1"
    )
    assert modified["width_m"].iloc[0] == 1.8
    assert affected == {"widen_path": 1}


def test_apply_hypothetical_multiple_toggles_at_once():
    modified, affected = wi.apply_hypothetical(
        _df(),
        {"surface_upgrade": 1.0, "add_sidewalk": "explicit_present"},
        scope="segment",
        target="1",
    )
    assert modified["surface_quality"].iloc[0] == 1.0
    assert modified["sidewalk_presence"].iloc[0] == "explicit_present"
    assert affected == {"surface_upgrade": 1, "add_sidewalk": 1}


def test_apply_hypothetical_unknown_toggle_raises():
    with pytest.raises(ValueError, match="unknown toggle"):
        wi.apply_hypothetical(_df(), {"teleport": True}, scope="segment", target="1")


def test_apply_hypothetical_does_not_mutate_original_df():
    df = _df()
    original_surface = df["surface_quality"].copy()
    wi.apply_hypothetical(df, {"surface_upgrade": 1.0}, scope="segment", target="1")
    pd.testing.assert_series_equal(df["surface_quality"], original_surface)


# --- before/after recompute via the real scorer (integration-style) --------


def test_whatif_before_after_recompute_uses_same_scorer_interface():
    df = _df()
    scorer = HeuristicScorer()
    original = wi.select_scope(df, "segment", "1")
    modified, affected = wi.apply_hypothetical(
        df, {"surface_upgrade": 1.0, "add_sidewalk": "explicit_present"}, scope="segment", target="1"
    )
    before = scorer.predict(original)
    after = scorer.predict(modified)
    assert after[0] > before[0]  # upgrading surface + adding a sidewalk can only help
    assert affected == {"surface_upgrade": 1, "add_sidewalk": 1}
