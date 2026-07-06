"""Offline tests for ``paveiq.scoring.score_city``."""

from __future__ import annotations

import os
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from paveiq.models.heuristic import HeuristicScorer
from paveiq.scoring import score_city as sc


def _features_with_wards_gdf(n: int = 5) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "osmid": [str(i) for i in range(n)],
            "highway": ["footway"] * n,
            "highway_likelihood": [1.0] * n,
            "length_m": [10.0] * n,
            "width_m": [1.5] * n,
            "surface_quality": [0.2, 0.6, 1.0, None, 1.0],
            "sidewalk_presence": ["explicit_present"] * n,
            "ward_id": ["1"] * n,
            "ward_name": ["Test Ward"] * n,
            "ward_lgd_code": [101] * n,
            "geometry": [LineString([(i, 0), (i, 1)]) for i in range(n)],
        },
        crs="EPSG:32643",
    )


# --- score_dataframe / scoring_report --------------------------------------


def test_score_dataframe_adds_score_and_subscore_columns():
    features = _features_with_wards_gdf()
    scorer = HeuristicScorer()
    scored = sc.score_dataframe(features, scorer)
    assert "score" in scored.columns
    assert {"surface_score", "sidewalk_score", "width_score", "highway_score", "n_features_observed"} <= set(
        scored.columns
    )
    assert len(scored) == len(features)
    # Original columns preserved.
    assert "osmid" in scored.columns and "ward_id" in scored.columns


def test_score_dataframe_scores_within_bounds():
    features = _features_with_wards_gdf()
    scored = sc.score_dataframe(features, HeuristicScorer())
    assert (scored["score"] >= 0).all()
    assert (scored["score"] <= 100).all()


def test_score_dataframe_falls_back_to_bare_score_without_explain():
    class BareScorer:
        def predict(self, df):
            return [42.0] * len(df)

    features = _features_with_wards_gdf()
    scored = sc.score_dataframe(features, BareScorer())
    assert list(scored["score"]) == [42.0] * len(features)
    assert "surface_score" not in scored.columns


def test_scoring_report_includes_summary_stats():
    features = _features_with_wards_gdf()
    scored = sc.score_dataframe(features, HeuristicScorer())
    report = sc.scoring_report(scored)
    assert "Scoring summary" in report
    assert "mean score" in report
    assert "n_features_observed" in report


def test_scoring_report_handles_empty():
    empty = pd.DataFrame({"score": []})
    report = sc.scoring_report(empty)
    assert "0 rows" in report


# --- score_and_save / end-to-end -------------------------------------------


def test_score_and_save_writes_expected_output(tmp_path):
    features_path = tmp_path / "koramangala_features_with_wards.parquet"
    _features_with_wards_gdf().to_parquet(features_path)

    model_path = tmp_path / "heuristic_scorer_v1.json"
    HeuristicScorer().save(model_path)

    out_dir = tmp_path / "out"
    out_path = sc.score_and_save(features_path, model_path, out_dir=out_dir)

    assert out_path == out_dir / "koramangala_features_with_wards_scored.parquet"
    assert out_path.exists()
    scored = gpd.read_parquet(out_path)
    assert "score" in scored.columns
    assert len(scored) == 5


# --- CLI ---------------------------------------------------------------


def test_main_end_to_end(tmp_path, capsys):
    features_path = tmp_path / "features_with_wards.parquet"
    _features_with_wards_gdf().to_parquet(features_path)
    model_path = tmp_path / "heuristic_scorer_v1.json"
    HeuristicScorer().save(model_path)
    out_dir = tmp_path / "out"

    rc = sc.main(
        [
            "--features",
            str(features_path),
            "--model",
            str(model_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "Wrote" in captured.out
    assert "Scoring summary" in captured.out


def test_main_missing_features_dir_returns_exit_code_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sc, "PROCESSED_DATA_DIR", tmp_path / "does_not_exist")
    monkeypatch.setattr(sc, "ARTIFACTS_DIR", tmp_path)
    rc = sc.main([])
    assert rc == 2
    assert "error" in capsys.readouterr().err


def test_main_missing_model_returns_exit_code_2(tmp_path, capsys):
    features_path = tmp_path / "features_with_wards.parquet"
    _features_with_wards_gdf().to_parquet(features_path)
    rc = sc.main(["--features", str(features_path), "--model", str(tmp_path / "no_model.json")])
    # An explicit missing --model path fails inside score_and_save (exit 1),
    # distinct from "no artifacts found via auto-discovery" (exit 2).
    assert rc == 1
    assert "error" in capsys.readouterr().err


# --- _find_latest_* helpers -------------------------------------------------


def test_find_latest_features_with_wards_picks_newest(tmp_path):
    older = tmp_path / "older_features_with_wards.parquet"
    newer = tmp_path / "newer_features_with_wards.parquet"
    older.write_text("")
    newer.write_text("")
    os.utime(older, (time.time() - 60, time.time() - 60))
    os.utime(newer, (time.time(), time.time()))
    assert sc._find_latest_features_with_wards(tmp_path) == newer


def test_find_latest_features_with_wards_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        sc._find_latest_features_with_wards(tmp_path / "no_such_dir")


def test_find_latest_artifact_picks_newest(tmp_path):
    older = tmp_path / "older_scorer.json"
    newer = tmp_path / "heuristic_scorer_v1.json"
    older.write_text("{}")
    newer.write_text("{}")
    os.utime(older, (time.time() - 60, time.time() - 60))
    os.utime(newer, (time.time(), time.time()))
    assert sc._find_latest_artifact(tmp_path) == newer


def test_find_latest_artifact_no_candidates(tmp_path):
    (tmp_path / "unrelated.txt").write_text("")
    with pytest.raises(FileNotFoundError, match="scorer"):
        sc._find_latest_artifact(tmp_path)
