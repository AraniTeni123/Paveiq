"""Offline tests for ``paveiq.models.heuristic``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paveiq.models.heuristic import (
    DEFAULT_WEIGHTS,
    HeuristicScorer,
)


def _row(
    surface_quality=1.0,
    sidewalk_presence="explicit_present",
    width_m=2.0,
    highway_likelihood=1.0,
):
    return pd.DataFrame(
        {
            "surface_quality": [surface_quality],
            "sidewalk_presence": [sidewalk_presence],
            "width_m": [width_m],
            "highway_likelihood": [highway_likelihood],
        }
    )


def test_predict_best_case_is_near_max_score():
    scorer = HeuristicScorer()
    out = scorer.predict(_row())
    assert out[0] == pytest.approx(100.0, abs=0.01)


def test_predict_worst_case_is_near_min_score():
    scorer = HeuristicScorer()
    df = _row(
        surface_quality=0.2,
        sidewalk_presence="explicit_absent",
        width_m=0.0,
        highway_likelihood=0.0,
    )
    out = scorer.predict(df)
    assert out[0] == pytest.approx(0.35 * 0.2 * 100, abs=0.5)


def test_all_missing_yields_neutral_score():
    """Every sub-score neutral-imputed to 0.5 -> score is exactly the midpoint."""
    scorer = HeuristicScorer()
    df = _row(
        surface_quality=np.nan,
        sidewalk_presence=None,
        width_m=np.nan,
        highway_likelihood=np.nan,
    )
    out = scorer.predict(df)
    assert out[0] == pytest.approx(50.0, abs=0.01)


def test_score_bounded_within_score_min_max():
    scorer = HeuristicScorer()
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "surface_quality": rng.choice([0.2, 0.6, 1.0, np.nan], size=n),
            "sidewalk_presence": rng.choice(
                ["explicit_present", "explicit_absent", "unlikely", None], size=n
            ),
            "width_m": rng.uniform(-1, 5, size=n),
            "highway_likelihood": rng.uniform(-0.5, 1.5, size=n),
        }
    )
    out = scorer.predict(df)
    assert (out >= 0).all()
    assert (out <= 100).all()


def test_n_features_observed_counts_non_null_inputs():
    scorer = HeuristicScorer()
    df = pd.concat(
        [
            _row(),  # all 4 present
            _row(surface_quality=np.nan),  # 3 present
            _row(surface_quality=np.nan, width_m=np.nan),  # 2 present
        ],
        ignore_index=True,
    )
    explained = scorer.explain(df)
    assert list(explained["n_features_observed"]) == [4, 3, 2]
    assert str(explained["n_features_observed"].dtype) == "int8"


def test_missing_does_not_get_renormalized_away():
    """A row missing surface_quality should differ from a row with it
    observed-but-neutral only by that one weighted sub-score term —
    i.e. missingness does NOT redistribute weight onto other features."""
    scorer = HeuristicScorer()
    complete = _row(surface_quality=1.0)
    missing = _row(surface_quality=np.nan)
    diff = scorer.predict(complete)[0] - scorer.predict(missing)[0]
    # surface weight 0.35, observed value 1.0 vs neutral-imputed 0.5:
    # difference should be 0.35 * (1.0 - 0.5) * 100 = 17.5
    assert diff == pytest.approx(17.5, abs=0.01)


def test_predict_raises_on_missing_columns():
    scorer = HeuristicScorer()
    df = pd.DataFrame({"surface_quality": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        scorer.predict(df)


def test_constructor_raises_on_incomplete_weights():
    with pytest.raises(ValueError, match="weights is missing"):
        HeuristicScorer(weights={"surface_quality": 1.0})


def test_save_load_round_trip(tmp_path):
    scorer = HeuristicScorer()
    path = tmp_path / "scorer.json"
    scorer.save(path)
    loaded = HeuristicScorer.load(path)
    assert loaded.weights == scorer.weights
    assert loaded.sidewalk_score_map == scorer.sidewalk_score_map
    assert loaded.width_min_m == scorer.width_min_m
    assert loaded.width_max_m == scorer.width_max_m
    df = _row()
    np.testing.assert_allclose(loaded.predict(df), scorer.predict(df))


def test_load_rejects_wrong_model_type(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"model_type": "sklearn_gbm_v1"}))
    with pytest.raises(ValueError, match="expected model_type"):
        HeuristicScorer.load(path)


def test_custom_weights_are_respected():
    scorer = HeuristicScorer(weights={**DEFAULT_WEIGHTS, "surface_quality": 1.0, "sidewalk_presence": 0.0,
                                       "width_m": 0.0, "highway_likelihood": 0.0})
    df = _row(surface_quality=1.0, sidewalk_presence="explicit_absent", width_m=0.0, highway_likelihood=0.0)
    out = scorer.predict(df)
    assert out[0] == pytest.approx(100.0, abs=0.01)
