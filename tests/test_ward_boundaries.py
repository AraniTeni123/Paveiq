"""Offline tests for ``paveiq.data_ingestion.ward_boundaries``.

Builds tiny synthetic ward polygons and segment geometries in
memory; never touches the real cached wards file or the network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

from paveiq.data_ingestion import ward_boundaries as wb
from paveiq.data_ingestion.ward_boundaries import (
    WARDS_FILENAME,
    coverage_report,
    fetch_wards,
    join_wards_to_features,
    load_wards,
)


# --- Fixture GeoJSON (KGI Sschema) -----------------------------------------


def _write_kgis_geojson(
    path: Path,
    *,
    ward_id: str = "186",
    ward_name: str = "Koramangala",
    lgd_code: int = 1303124,
    extra_columns: dict | None = None,
) -> Path:
    """Write a minimal valid GeoJSON that mimics the KGIS schema.

    Includes ``geometry`` and the three source columns we read,
    plus a couple of KGIS-specific extras that the loader is
    expected to *drop* during schema coercion.
    """
    props = {
        "KGISWardID": 5120,
        "KGISWardCode": "2003186",
        "KGISWardNo": ward_id,
        "KGISWardName": ward_name,
        "KGISTownCode": "2003",
        "LGD_WardCode": lgd_code,
    }
    if extra_columns:
        props.update(extra_columns)
    geojson = {
        "type": "FeatureCollection",
        "name": "BBMP-test",
        "features": [
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [77.60, 12.93],
                            [77.65, 12.93],
                            [77.65, 12.95],
                            [77.60, 12.95],
                            [77.60, 12.93],
                        ]
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(geojson))
    return path


# --- load_wards (schema coercion) ------------------------------------------


def test_load_wards_standardises_schema(tmp_path):
    path = _write_kgis_geojson(
        tmp_path / "ward.geojson",
        extra_columns={"extra_field": "should be dropped", "notes": "junk"},
    )
    wards = load_wards(path)
    assert set(wards.columns) >= {"ward_id", "ward_name", "ward_lgd_code", "geometry"}
    # Extra KGIS columns should be gone, but geometry stays.
    assert "KGISWardID" not in wards.columns
    assert "extra_field" not in wards.columns


def test_load_wards_preserves_leading_zeros(tmp_path):
    path = _write_kgis_geojson(tmp_path / "ward.geojson", ward_id="001")
    wards = load_wards(path)
    assert wards["ward_id"].iloc[0] == "001"  # not the int 1
    assert isinstance(wards["ward_id"].iloc[0], str)


def test_load_wards_ward_lgd_code_is_nullable_int64(tmp_path):
    path = _write_kgis_geojson(tmp_path / "ward.geojson", lgd_code=1303124)
    wards = load_wards(path)
    assert str(wards["ward_lgd_code"].dtype) == "Int64"
    assert wards["ward_lgd_code"].iloc[0] == 1303124


def test_load_wards_raises_on_missing_source_columns(tmp_path):
    """If upstream schema changes, we fail loudly, not silently."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"KGISWardNo": "1", "KGISWardName": "X"},  # no LGD
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }
    path = tmp_path / "no_lgd.geojson"
    path.write_text(json.dumps(geojson))
    with pytest.raises(ValueError, match="LGD_WardCode"):
        load_wards(path)


# --- fetch_wards (caching) -------------------------------------------------


def test_fetch_wards_caches(tmp_path):
    with patch.object(wb, "_download") as mock_dl:
        first = fetch_wards(tmp_path)
        # The first call writes a real file (via the mock), so subsequent
        # calls within the same test see the cache.
        # Make the mock actually create the file:
        (tmp_path / WARDS_FILENAME).write_text("{}")
        second = fetch_wards(tmp_path)
    assert first == second == tmp_path / WARDS_FILENAME
    assert mock_dl.call_count == 1


def test_fetch_wards_force_redownloads(tmp_path):
    # Pre-populate the cache.
    (tmp_path / WARDS_FILENAME).write_text("{}")
    with patch.object(wb, "_download") as mock_dl:
        # Mock writes a fresh file so the path "exists" after each call.
        def fake_dl(url, dest):
            dest.write_text("{}")
        mock_dl.side_effect = fake_dl
        fetch_wards(tmp_path, force=True)
        fetch_wards(tmp_path, force=True)
    assert mock_dl.call_count == 2


# --- join_wards_to_features (the heart of the stage) ---------------------


def _two_square_wards_gdf() -> gpd.GeoDataFrame:
    """Two non-overlapping unit squares side by side.

    Ward A:  [0,1] x [0,1]   (id="A")
    Ward B:  [1,2] x [0,1]   (id="B")
    """
    rows = [
        {
            "ward_id": "A",
            "ward_name": "Alpha",
            "ward_lgd_code": pd.array([101], dtype="Int64")[0],
            "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        },
        {
            "ward_id": "B",
            "ward_name": "Beta",
            "ward_lgd_code": pd.array([102], dtype="Int64")[0],
            "geometry": Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        },
    ]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def test_join_wards_to_features_one_to_one():
    wards = _two_square_wards_gdf()
    feats = gpd.GeoDataFrame(
        {
            "osmid": ["1", "2", "3"],
            "highway": ["footway", "residential", "service"],
            "geometry": [
                LineString([(0.1, 0.1), (0.2, 0.2)]),  # in A
                LineString([(0.5, 0.7), (0.6, 0.8)]),  # in A
                LineString([(1.3, 0.4), (1.4, 0.5)]),  # in B
            ],
        },
        crs="EPSG:4326",
    )
    out = join_wards_to_features(feats, wards)
    assert len(out) == 3
    assert list(out["ward_id"]) == ["A", "A", "B"]
    assert list(out["ward_name"]) == ["Alpha", "Alpha", "Beta"]
    assert list(out["ward_lgd_code"]) == [101, 101, 102]


def test_join_wards_to_features_crossing_segment_picks_one_ward():
    """A segment that crosses the boundary should be assigned to one specific ward
    (the one containing its representative point) — not duplicated, not random."""
    wards = _two_square_wards_gdf()
    # A horizontal line spanning both wards.
    feats = gpd.GeoDataFrame(
        {
            "osmid": ["x"],
            "highway": ["footway"],
            "geometry": [LineString([(0.4, 0.5), (1.6, 0.5)])],
        },
        crs="EPSG:4326",
    )
    out = join_wards_to_features(feats, wards)
    # shapely's representative_point picks a point guaranteed to be
    # inside the geometry. For a horizontal line, it can be any point
    # along the line — we just need the join to give *one* answer.
    assert len(out) == 1
    assert out["ward_id"].iloc[0] in ("A", "B")  # exactly one, not both


def test_join_wards_to_features_orphan_gets_empty_ward_id():
    wards = _two_square_wards_gdf()
    # A segment far outside both wards.
    feats = gpd.GeoDataFrame(
        {
            "osmid": ["orphan"],
            "highway": ["footway"],
            "geometry": [LineString([(10, 10), (10, 11)])],
        },
        crs="EPSG:4326",
    )
    out = join_wards_to_features(feats, wards)
    assert len(out) == 1
    # Empty / NaN for an orphan — the test is "is this row identifiable
    # as an orphan?" which is the question the coverage report answers.
    assert out["ward_id"].iloc[0] in ("", None) or pd.isna(out["ward_id"].iloc[0])
    assert pd.isna(out["ward_lgd_code"].iloc[0])


def test_join_wards_to_features_preserves_row_count_with_mixed_input():
    wards = _two_square_wards_gdf()
    feats = gpd.GeoDataFrame(
        {
            "osmid": ["1", "2", "3", "4", "5"],
            "highway": ["footway"] * 5,
            "geometry": [
                LineString([(0.1, 0.1), (0.2, 0.2)]),  # A
                LineString([(1.3, 0.4), (1.4, 0.5)]),  # B
                LineString([(10, 10), (10, 11)]),      # orphan
                LineString([(0.4, 0.5), (1.6, 0.5)]),  # crossing
                LineString([(1.8, 0.2), (1.9, 0.3)]),  # B
            ],
        },
        crs="EPSG:4326",
    )
    out = join_wards_to_features(feats, wards)
    assert len(out) == 5
    # The original columns are still there.
    assert "osmid" in out.columns
    assert "highway" in out.columns
    # And the ward columns are present.
    assert {"ward_id", "ward_name", "ward_lgd_code"} <= set(out.columns)


def test_join_wards_to_features_handles_overlapping_wards():
    """Real BBMP wards occasionally overlap along a shared edge (a topology
    error in the source data), so a representative point can intersect two
    wards even though it isn't a boundary-crossing segment. The join must
    still pick exactly one ward instead of raising or duplicating the row."""
    overlapping_wards = gpd.GeoDataFrame(
        [
            {
                "ward_id": "A",
                "ward_name": "Alpha",
                "ward_lgd_code": pd.array([101], dtype="Int64")[0],
                "geometry": Polygon([(0, 0), (1.1, 0), (1.1, 1), (0, 1)]),
            },
            {
                "ward_id": "B",
                "ward_name": "Beta",
                "ward_lgd_code": pd.array([102], dtype="Int64")[0],
                "geometry": Polygon([(0.9, 0), (2, 0), (2, 1), (0.9, 1)]),
            },
        ],
        crs="EPSG:4326",
    )
    feats = gpd.GeoDataFrame(
        {
            "osmid": ["1"],
            "highway": ["footway"],
            # Sits inside the [0.9, 1.1] overlap strip shared by A and B.
            "geometry": [LineString([(0.95, 0.4), (1.0, 0.5)])],
        },
        crs="EPSG:4326",
    )
    out = join_wards_to_features(feats, overlapping_wards)
    assert len(out) == 1
    assert out["ward_id"].iloc[0] in ("A", "B")


def test_join_wards_to_features_reprojects_when_crs_differs():
    """Features in EPSG:32643 (UTM 43N), wards in EPSG:4326 — join must still work.

    The synthetic wards are placed at Bengaluru-ish latitudes
    (12.93-12.95°N, 77.60-77.62°E) so that reprojecting to UTM
    43N lands inside the expected zone. We feed a feature whose
    geometry is the *centroid* of ward A reprojected to UTM 43N
    — guaranteeing the point is inside the ward even though small
    polygons reproject as thin strips rather than axis-aligned
    rectangles.
    """
    rows = [
        {
            "ward_id": "A",
            "ward_name": "Alpha",
            "ward_lgd_code": pd.array([101], dtype="Int64")[0],
            "geometry": Polygon([(77.60, 12.93), (77.61, 12.93), (77.61, 12.94), (77.60, 12.94)]),
        },
        {
            "ward_id": "B",
            "ward_name": "Beta",
            "ward_lgd_code": pd.array([102], dtype="Int64")[0],
            "geometry": Polygon([(77.61, 12.93), (77.62, 12.93), (77.62, 12.94), (77.61, 12.94)]),
        },
    ]
    wards = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    wards_utm = wards.to_crs("EPSG:32643")

    # Use the UTM centroid of ward A as the feature's representative
    # point — by construction that's inside the ward.
    utm_a = wards_utm.loc[wards_utm["ward_id"] == "A", "geometry"].iloc[0]
    cx, cy = utm_a.centroid.x, utm_a.centroid.y
    in_A = LineString([(cx, cy), (cx + 1, cy + 1)])  # tiny segment at centroid

    feats = gpd.GeoDataFrame(
        {"osmid": ["1"], "highway": ["footway"], "geometry": [in_A]},
        crs="EPSG:32643",
    )
    out = join_wards_to_features(feats, wards_utm)
    assert out["ward_id"].iloc[0] == "A"
    # Output CRS should match input features' CRS (UTM 43N), not the wards'.
    assert out.crs == feats.crs


# --- coverage_report ------------------------------------------------------


def test_coverage_report_includes_orphan_count():
    wards = _two_square_wards_gdf()
    feats = gpd.GeoDataFrame(
        {
            "osmid": ["1", "2", "3"],
            "highway": ["footway"] * 3,
            "geometry": [
                LineString([(0.1, 0.1), (0.2, 0.2)]),  # A
                LineString([(1.3, 0.4), (1.4, 0.5)]),  # B
                LineString([(10, 10), (10, 11)]),      # orphan
            ],
        },
        crs="EPSG:4326",
    )
    joined = join_wards_to_features(feats, wards)
    report = coverage_report(joined)
    assert "BBMP 2022" in report
    assert "3" in report and "2" in report  # 3 total, 2 matched
    assert "1" in report                  # 1 orphan
    assert "distinct wards" in report
    assert "Alpha" in report and "Beta" in report


def test_coverage_report_handles_empty_gdf():
    empty = gpd.GeoDataFrame(
        {"ward_id": [], "ward_name": [], "ward_lgd_code": []},
        geometry=[],
        crs="EPSG:4326",
    )
    report = coverage_report(empty)
    assert "0 rows" in report


# --- _find_latest_features ------------------------------------------------


def test_find_latest_features_picks_newest(tmp_path):
    older = tmp_path / "older_features.parquet"
    newer = tmp_path / "newer_features.parquet"
    older.write_text("")  # empty is fine for the mtime test
    newer.write_text("")
    older_mtime = time.time() - 60
    newer_mtime = time.time()
    import os
    os.utime(older, (older_mtime, older_mtime))
    os.utime(newer, (newer_mtime, newer_mtime))
    assert wb._find_latest_features(tmp_path) == newer


def test_find_latest_features_skips_with_wards_outputs(tmp_path):
    """A previous run's *_with_wards.parquet must not be picked up as 'latest features'."""
    features = tmp_path / "real_features.parquet"
    features.write_text("")
    with_wards = tmp_path / "real_features_with_wards.parquet"
    with_wards.write_text("")
    # Make the with_wards file look newer.
    new_mtime = time.time()
    old_mtime = time.time() - 60
    import os
    os.utime(with_wards, (new_mtime, new_mtime))
    os.utime(features, (old_mtime, old_mtime))
    assert wb._find_latest_features(tmp_path) == features


def test_find_latest_features_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        wb._find_latest_features(tmp_path / "no_such_dir")


def test_find_latest_features_no_features_files(tmp_path):
    (tmp_path / "unrelated.txt").write_text("")
    with pytest.raises(FileNotFoundError, match="features.parquet"):
        wb._find_latest_features(tmp_path)
