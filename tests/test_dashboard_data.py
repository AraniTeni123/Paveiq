"""Offline tests for ``paveiq.dashboard.data``."""

from __future__ import annotations

import os
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from paveiq.dashboard import data as dd


def _scored_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "osmid": ["1", "2", "3", "4"],
            "highway": ["footway", "residential", "footway", "trunk"],
            "ward_id": ["A", "A", "B", "B"],
            "ward_name": ["Alpha", "Alpha", "Beta", "Beta"],
            "score": [90.0, 30.0, 60.0, 10.0],
            "length_m": [10.0, 20.0, 5.0, 15.0],
            "geometry": [LineString([(i, 0), (i, 1)]) for i in range(4)],
        },
        crs="EPSG:32643",
    )


# --- ward_summary ------------------------------------------------------


def test_ward_summary_aggregates_correctly():
    summary = dd.ward_summary(_scored_gdf())
    beta = summary[summary["ward_id"] == "B"].iloc[0]
    assert beta["mean_score"] == pytest.approx(35.0)
    assert beta["segment_count"] == 2
    assert beta["total_length_m"] == pytest.approx(20.0)
    # Both Beta segments (60, 10) -> one below the 40 threshold -> 50%.
    assert beta["pct_poor"] == pytest.approx(50.0)


def test_ward_summary_sorted_worst_first():
    summary = dd.ward_summary(_scored_gdf())
    assert list(summary["ward_id"]) == ["B", "A"]  # B mean=35 < A mean=60


def test_ward_summary_all_poor_ward():
    poor_ward = dd.ward_summary(_scored_gdf())
    alpha = poor_ward[poor_ward["ward_id"] == "A"].iloc[0]
    # Alpha scores are 90, 30 -> one below 40 -> 50%.
    assert alpha["pct_poor"] == pytest.approx(50.0)


# --- leaderboard ---------------------------------------------------------


def test_leaderboard_orders_worst_first():
    board = dd.leaderboard(_scored_gdf(), n=2)
    assert list(board["osmid"]) == ["4", "2"]  # scores 10, 30
    assert len(board) == 2


def test_leaderboard_respects_n():
    board = dd.leaderboard(_scored_gdf(), n=1)
    assert len(board) == 1


# --- to_wgs84 --------------------------------------------------------------


def test_to_wgs84_reprojects_known_point():
    # A point in UTM 43N known to correspond to roughly (77.6, 12.9) lon/lat.
    gdf = gpd.GeoDataFrame({"geometry": [Point(77.6, 12.9)]}, crs="EPSG:4326").to_crs("EPSG:32643")
    out = dd.to_wgs84(gdf)
    assert out.crs.to_epsg() == 4326
    pt = out.geometry.iloc[0]
    assert pt.x == pytest.approx(77.6, abs=1e-6)
    assert pt.y == pytest.approx(12.9, abs=1e-6)


def test_to_wgs84_is_noop_if_already_wgs84():
    gdf = gpd.GeoDataFrame({"geometry": [Point(77.6, 12.9)]}, crs="EPSG:4326")
    out = dd.to_wgs84(gdf)
    assert out.crs.to_epsg() == 4326


def test_to_wgs84_raises_without_crs():
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})
    with pytest.raises(ValueError, match="no CRS"):
        dd.to_wgs84(gdf)


# --- _find_latest_scored / load_scored_segments -----------------------------


def test_find_latest_scored_picks_newest(tmp_path):
    older = tmp_path / "older_scored.parquet"
    newer = tmp_path / "newer_scored.parquet"
    older.write_text("")
    newer.write_text("")
    os.utime(older, (time.time() - 60, time.time() - 60))
    os.utime(newer, (time.time(), time.time()))
    assert dd._find_latest_scored(tmp_path) == newer


def test_find_latest_scored_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        dd._find_latest_scored(tmp_path / "no_such_dir")


def test_find_latest_scored_no_candidates(tmp_path):
    (tmp_path / "unrelated.txt").write_text("")
    with pytest.raises(FileNotFoundError, match="scored.parquet"):
        dd._find_latest_scored(tmp_path)


def test_load_scored_segments_reads_explicit_path(tmp_path):
    path = tmp_path / "features_with_wards_scored.parquet"
    _scored_gdf().to_parquet(path)
    loaded = dd.load_scored_segments(path)
    assert len(loaded) == 4
    assert "score" in loaded.columns


# --- demo-dir fallback -------------------------------------------------


def test_find_latest_scored_falls_back_to_demo_when_processed_empty(tmp_path, monkeypatch):
    empty_processed = tmp_path / "processed"
    empty_processed.mkdir()
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    demo_file = demo_dir / "demo_scored.parquet"
    demo_file.write_text("")

    monkeypatch.setattr(dd, "PROCESSED_DATA_DIR", empty_processed)
    monkeypatch.setattr(dd, "DEMO_DATA_DIR", demo_dir)

    assert dd._find_latest_scored() == demo_file


def test_find_latest_scored_prefers_processed_over_demo(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    real_file = processed_dir / "real_scored.parquet"
    real_file.write_text("")
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    (demo_dir / "demo_scored.parquet").write_text("")

    monkeypatch.setattr(dd, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(dd, "DEMO_DATA_DIR", demo_dir)

    assert dd._find_latest_scored() == real_file


def test_find_latest_scored_raises_when_both_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "PROCESSED_DATA_DIR", tmp_path / "no_processed")
    monkeypatch.setattr(dd, "DEMO_DATA_DIR", tmp_path / "no_demo")
    with pytest.raises(FileNotFoundError, match="scored.parquet"):
        dd._find_latest_scored()


def test_find_latest_scored_explicit_dir_does_not_fall_back_to_demo(tmp_path, monkeypatch):
    """An explicitly-passed processed_dir (as tests use) must never silently
    fall back to the real demo dir — only the no-argument default path does."""
    empty_processed = tmp_path / "processed"
    empty_processed.mkdir()
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    (demo_dir / "demo_scored.parquet").write_text("")

    monkeypatch.setattr(dd, "DEMO_DATA_DIR", demo_dir)

    with pytest.raises(FileNotFoundError, match="scored.parquet"):
        dd._find_latest_scored(empty_processed)
