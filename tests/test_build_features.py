"""Offline tests for ``paveiq.features.build_features``.

Pure-function tests where possible; the integration tests build
small synthetic GeoDataFrames in memory so nothing touches
``data/raw/`` or ``data/processed/``.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from paveiq.features import build_features as bf
from paveiq.features.build_features import (
    ENGINEERED_FEATURES,
    build_features,
    feature_coverage_report,
    find_latest_raw,
    highway_likelihood,
    parse_width_m,
    process_file,
    sidewalk_presence,
    surface_quality,
)


# --- parse_width_m ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2", 2.0),
        ("2.5", 2.5),
        ("2 m", 2.0),
        ("100 cm", 1.0),
        ("500 mm", 0.5),
        ("25 ft", pytest.approx(7.62, rel=1e-9)),
        ("20'", pytest.approx(6.096, rel=1e-9)),
        ('24"', pytest.approx(0.6096, rel=1e-9)),
        ("24.38", 24.38),  # bare float, in plausible range
        ("", float("nan")),
        (None, float("nan")),
        ("not a number", float("nan")),
        ("2 meters tall", float("nan")),  # trailing garbage
        ("-1", float("nan")),  # implausible
        ("0.05", float("nan")),  # implausible
        ("200", float("nan")),  # implausible (>100 m)
        (3, 3.0),  # already a number
        (0, float("nan")),  # implausible
    ],
)
def test_parse_width_m(raw, expected):
    out = parse_width_m(raw)
    if isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(out)
    else:
        assert out == pytest.approx(expected, rel=1e-9)


# --- highway_likelihood ----------------------------------------------------


@pytest.mark.parametrize(
    "highway,expected",
    [
        ("footway", 1.00),
        ("path", 0.95),
        ("pedestrian", 0.90),
        ("living_street", 0.85),
        ("steps", 0.85),
        ("cycleway", 0.60),
        ("residential", 0.50),
        ("service", 0.30),
        ("unclassified", 0.30),
        ("tertiary", 0.20),
        ("tertiary_link", 0.15),  # tertiary - 0.05
        ("secondary", 0.15),
        ("secondary_link", 0.10),
        ("primary", 0.10),
        ("primary_link", 0.05),
        ("trunk", 0.05),
        ("trunk_link", 0.00),
        ("construction", 0.20),
        ("not_a_real_highway", float("nan")),
        (None, float("nan")),
        (float("nan"), float("nan")),
    ],
)
def test_highway_likelihood(highway, expected):
    out = highway_likelihood(highway)
    if isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(out)
    else:
        assert out == pytest.approx(expected, abs=1e-9)


# --- surface_quality -------------------------------------------------------


@pytest.mark.parametrize(
    "surface,expected",
    [
        ("asphalt", 1.0),
        ("paving_stones", 1.0),
        ("concrete", 1.0),
        ("paved", 1.0),
        ("sett", 1.0),
        ("cobblestone", 1.0),
        ("tiles", 1.0),
        ("clinker_plates", 1.0),
        ("stone", 1.0),
        ("cement", 1.0),
        ("concrete:plates", 1.0),
        ("compacted", 0.6),
        ("unpaved", 0.2),
        ("dirt", 0.2),
        ("ground", 0.2),
        ("gravel", 0.2),
        ("rock", 0.2),
        ("mud", 0.2),
        ("ston", 0.2),  # common OSM typo
        ("made_up_material", float("nan")),
        (None, float("nan")),
    ],
)
def test_surface_quality(surface, expected):
    out = surface_quality(surface)
    if isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(out)
    else:
        assert out == pytest.approx(expected, abs=1e-9)


# --- sidewalk_presence (the conditional-logic encoder) ---------------------


@pytest.mark.parametrize(
    "highway,sidewalk,expected",
    [
        # Explicit tag wins regardless of highway.
        ("residential", "separate", "explicit_present"),
        ("residential", "both", "explicit_present"),
        ("residential", "left", "explicit_present"),
        ("residential", "right", "explicit_present"),
        ("primary", "separate", "explicit_present"),  # even on a highway
        ("residential", "no", "explicit_absent"),
        ("primary", "no", "explicit_absent"),
        # No tag on a footpath-like highway -> implicit_present.
        ("footway", None, "implicit_present"),
        ("footway", "no", "explicit_absent"),  # explicit beats implicit
        ("path", None, "implicit_present"),
        ("pedestrian", None, "implicit_present"),
        ("steps", None, "implicit_present"),
        ("cycleway", None, "implicit_present"),
        # No tag on a residential-class road -> likely_present (prior, not a measurement).
        ("residential", None, "likely_present"),
        ("service", None, "likely_present"),
        ("unclassified", None, "likely_present"),
        ("living_street", None, "likely_present"),
        # No tag on a highway -> unlikely.
        ("tertiary", None, "unlikely"),
        ("secondary", None, "unlikely"),
        ("primary", None, "unlikely"),
        ("trunk", None, "unlikely"),
        ("primary_link", None, "unlikely"),
        ("construction", None, "unlikely"),
        # No highway at all -> unlikely.
        (None, None, "unlikely"),
        (float("nan"), None, "unlikely"),
    ],
)
def test_sidewalk_presence(highway, sidewalk, expected):
    assert sidewalk_presence(highway, sidewalk) == expected


# --- build_features integration -------------------------------------------


def _two_segments_gdf() -> gpd.GeoDataFrame:
    """Two LineStrings ~100 m and ~50 m long, in EPSG:4326 around Bengaluru."""
    # 1 degree of latitude is ~111 km, so 0.0009 deg ≈ 100 m
    rows = [
        {
            "osmid": 1,
            "highway": "footway",
            "name": None,
            "sidewalk": None,
            "surface": "asphalt",
            "width": "2",
            "smoothness": None,
            "geometry": LineString([(77.60, 12.97), (77.609, 12.97)]),  # ~93 m E-W
        },
        {
            "osmid": [2, 3],  # multi-id way
            "highway": "residential",
            "name": "5th Cross",
            "sidewalk": "separate",
            "surface": "concrete",
            "width": "20'",  # 20 ft -> 6.096 m
            "smoothness": None,
            "geometry": LineString([(77.60, 12.97), (77.60, 12.9705)]),  # ~55 m N-S
        },
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def test_build_features_reprojects_and_computes_length():
    gdf = _two_segments_gdf()
    out = build_features(gdf)
    assert str(out.crs).upper().endswith("32643")
    # Length should be in metres now. At 12.97°N (UTM 43N), 1° lon
    # is ~108.5 km, so 0.009° lon ≈ 977 m; 0.0005° lat ≈ 55 m.
    assert out["length_m"].iloc[0] == pytest.approx(977.0, abs=10.0)
    assert out["length_m"].iloc[1] == pytest.approx(55.0, abs=5.0)
    # osmid is coerced to a string, multi-id joined.
    assert out["osmid"].iloc[0] == "1"
    assert out["osmid"].iloc[1] == "2,3"
    # Width parser handled both the bare-metres and the foot case.
    assert out["width_m"].iloc[0] == pytest.approx(2.0, abs=1e-9)
    assert out["width_m"].iloc[1] == pytest.approx(6.096, rel=1e-9)
    # Highway + sidewalk -> expected buckets.
    assert out["highway_likelihood"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert out["highway_likelihood"].iloc[1] == pytest.approx(0.5, abs=1e-9)
    assert out["sidewalk_presence"].iloc[0] == "implicit_present"
    assert out["sidewalk_presence"].iloc[1] == "explicit_present"
    # Surface ordinal.
    assert out["surface_quality"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert out["surface_quality"].iloc[1] == pytest.approx(1.0, abs=1e-9)


def test_build_features_handles_missing_columns():
    """A GDF that only has highway + geometry should still produce a valid frame."""
    rows = [
        {
            "highway": "footway",
            "geometry": LineString([(77.60, 12.97), (77.601, 12.97)]),
        }
    ]
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    out = build_features(gdf)
    assert len(out) == 1
    assert out["highway_likelihood"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert math.isnan(out["width_m"].iloc[0])
    assert math.isnan(out["surface_quality"].iloc[0])
    assert out["sidewalk_presence"].iloc[0] == "implicit_present"


def test_build_features_drops_name_column():
    rows = [
        {
            "highway": "footway",
            "name": "should be dropped",
            "sidewalk": None,
            "surface": None,
            "width": None,
            "smoothness": None,
            "geometry": LineString([(77.60, 12.97), (77.601, 12.97)]),
        }
    ]
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    out = build_features(gdf)
    assert "name" not in out.columns


def test_build_features_output_schema():
    gdf = _two_segments_gdf()
    out = build_features(gdf)
    expected_cols = {
        "osmid", "highway", "highway_likelihood", "length_m", "width_m",
        "surface_quality", "sidewalk_presence", "geometry",
    }
    assert set(out.columns) == expected_cols
    assert str(out["highway_likelihood"].dtype) == "float32"
    assert str(out["length_m"].dtype) == "float32"
    assert str(out["width_m"].dtype) == "float32"
    assert str(out["surface_quality"].dtype) == "float32"
    # CRS is the UTM zone.
    assert out.crs.to_epsg() == 32643


# --- feature_coverage_report -----------------------------------------------


def test_feature_coverage_report_includes_all_engineered_features():
    gdf = _two_segments_gdf()
    out = build_features(gdf)
    report = feature_coverage_report(out)
    assert "highway_likelihood" in report
    assert "length_m" in report
    assert "width_m" in report
    assert "surface_quality" in report
    assert "sidewalk_presence" in report
    # Both rows have length + highway + sidewalk; both have width (parsed).
    assert "2/2" in report or "2 / 2" in report  # format may use commas


def test_feature_coverage_report_handles_empty():
    empty = gpd.GeoDataFrame(
        {c: [] for c in ENGINEERED_FEATURES} | {"geometry": []},
        crs="EPSG:32643",
    )
    report = feature_coverage_report(empty)
    assert "0 rows" in report


# --- find_latest_raw (no real data) ---------------------------------------


def test_find_latest_raw_picks_newest(tmp_path):
    older = tmp_path / "older_footpaths.geojson"
    newer = tmp_path / "newer_footpaths.geojson"
    older.write_text("{}")
    newer.write_text("{}")
    # Force older to be older.
    older_mtime = time.time() - 60
    newer_mtime = time.time()
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))
    assert find_latest_raw(tmp_path) == newer


def test_find_latest_raw_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        find_latest_raw(tmp_path / "no_such_dir")


def test_find_latest_raw_no_matching_files(tmp_path):
    (tmp_path / "unrelated.geojson").write_text("{}")
    with pytest.raises(FileNotFoundError, match="no `.*_footpaths.geojson`"):
        find_latest_raw(tmp_path)


# --- process_file (in tmp, never in data/processed) -----------------------


def test_process_file_writes_parquet_with_expected_name(tmp_path):
    gdf = _two_segments_gdf()
    raw_path = tmp_path / "bengaluru_india_footpaths.geojson"
    gdf.to_file(raw_path, driver="GeoJSON")
    out_dir = tmp_path / "processed"
    out_path = process_file(raw_path, out_dir=out_dir)
    assert out_path == out_dir / "bengaluru_india_features.parquet"
    assert out_path.exists()
    # Round-trip.
    loaded = gpd.read_parquet(out_path)
    assert set(loaded.columns) >= set(ENGINEERED_FEATURES)
    assert len(loaded) == 2
