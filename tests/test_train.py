"""Offline tests for ``paveiq.models.train``."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from paveiq.models import train as train_mod
from paveiq.models.heuristic import DEFAULT_WEIGHTS, HeuristicScorer


def _write_features_with_wards(path: Path, n: int = 5) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "osmid": [str(i) for i in range(n)],
            "highway": ["footway"] * n,
            "highway_likelihood": [1.0] * n,
            "length_m": [10.0] * n,
            "width_m": [1.5] * n,
            "surface_quality": [1.0] * n,
            "sidewalk_presence": ["explicit_present"] * n,
            "ward_id": ["1"] * n,
            "ward_name": ["Test Ward"] * n,
            "ward_lgd_code": [101] * n,
            "geometry": [LineString([(i, 0), (i, 1)]) for i in range(n)],
        },
        crs="EPSG:32643",
    )
    gdf.to_parquet(path)
    return path


def test_main_writes_artifact_and_reports_distribution(tmp_path, capsys):
    features_path = _write_features_with_wards(tmp_path / "features_with_wards.parquet")
    out_path = tmp_path / "scorer.json"
    rc = train_mod.main(["--features", str(features_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()

    loaded = HeuristicScorer.load(out_path)
    assert loaded.weights == DEFAULT_WEIGHTS

    captured = capsys.readouterr()
    assert f"Wrote {out_path}" in captured.out
    assert "Score distribution" in captured.out


def test_main_applies_weight_overrides(tmp_path):
    features_path = _write_features_with_wards(tmp_path / "features_with_wards.parquet")
    out_path = tmp_path / "scorer.json"
    rc = train_mod.main(
        [
            "--features",
            str(features_path),
            "--out",
            str(out_path),
            "--weight-surface-quality",
            "0.9",
        ]
    )
    assert rc == 0
    data = json.loads(out_path.read_text())
    assert data["weights"]["surface_quality"] == 0.9
    # Untouched weights stay at their defaults.
    assert data["weights"]["highway_likelihood"] == DEFAULT_WEIGHTS["highway_likelihood"]


def test_main_explicit_missing_features_file_returns_exit_code_1(tmp_path, capsys):
    """An explicit --features path that doesn't exist is a read failure (exit 1),
    distinct from the auto-discovery 'no candidates in directory' case (exit 2)."""
    rc = train_mod.main(["--features", str(tmp_path / "does_not_exist.parquet")])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_main_no_candidates_in_processed_dir_returns_exit_code_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_mod, "PROCESSED_DATA_DIR", tmp_path)
    rc = train_mod.main([])
    assert rc == 2
    assert "error" in capsys.readouterr().err


def test_find_latest_features_with_wards_picks_newest(tmp_path):
    older = tmp_path / "older_features_with_wards.parquet"
    newer = tmp_path / "newer_features_with_wards.parquet"
    older.write_text("")
    newer.write_text("")
    os.utime(older, (time.time() - 60, time.time() - 60))
    os.utime(newer, (time.time(), time.time()))
    assert train_mod._find_latest_features_with_wards(tmp_path) == newer


def test_find_latest_features_with_wards_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_mod._find_latest_features_with_wards(tmp_path / "no_such_dir")


def test_find_latest_features_with_wards_no_candidates(tmp_path):
    (tmp_path / "unrelated.txt").write_text("")
    with pytest.raises(FileNotFoundError, match="with_wards.parquet"):
        train_mod._find_latest_features_with_wards(tmp_path)


def test_score_distribution_report_handles_empty():
    import numpy as np

    report = train_mod.score_distribution_report(np.array([]))
    assert "0 rows" in report
