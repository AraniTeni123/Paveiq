"""Offline tests for ``paveiq.data_ingestion.osm_loader``.

These tests deliberately do **not** hit Overpass or the network —
the goal is to cover the public surface (validation, slugifying,
coverage formatting, and the polygon-resolution dispatch) so
that CI is fast and deterministic. The end-to-end network path
is verified manually in the verification step of the plan.
"""

from __future__ import annotations

from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon, box

from paveiq.data_ingestion import osm_loader
from paveiq.data_ingestion.osm_loader import (
    DEFAULT_PLACE,
    _resolve_polygon,
    _slugify_area,
    coverage_report,
    load_osm_footpaths,
)


# --- Helpers ---------------------------------------------------------------


def _make_synthetic_gdf(sidewalk=3, surface=15, width=2, smoothness=1, total=100):
    """Build a tiny GeoDataFrame with the documented tag sparsity.

    Defaults reflect the *expected* Bengaluru coverage so the
    synthetic fixture stays in step with reality. Each non-null
    value is the string ``"yes"`` — we don't need real values,
    just non-null.
    """
    rows = []
    for i in range(total):
        geom = LineString([(77.5 + i * 1e-4, 12.9), (77.5 + i * 1e-4, 12.91)])
        rows.append(
            {
                "osmid": i,
                "highway": "footway",
                "name": None,
                "sidewalk": "yes" if i < sidewalk else None,
                "surface": "asphalt" if i < surface else None,
                "width": "2" if i < width else None,
                "smoothness": "good" if i < smoothness else None,
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# --- coverage_report -------------------------------------------------------


def test_coverage_report_uses_documented_fractions():
    gdf = _make_synthetic_gdf(sidewalk=3, surface=15, width=2, smoothness=1, total=100)
    text = coverage_report(gdf)
    assert "Fetched 100 ways." in text
    assert "sidewalk" in text and "3/100" in text and "3.0%" in text
    assert "surface" in text and "15/100" in text and "15.0%" in text
    assert "width" in text and "2/100" in text and "2.0%" in text
    assert "smoothness" in text and "1/100" in text and "1.0%" in text


def test_coverage_report_handles_empty_gdf():
    empty = gpd.GeoDataFrame(
        {"sidewalk": [], "surface": [], "width": [], "smoothness": []},
        geometry=[],
        crs="EPSG:4326",
    )
    text = coverage_report(empty)
    assert "0 ways" in text


def test_coverage_report_handles_missing_columns():
    gdf = _make_synthetic_gdf()  # has all four target columns
    # Drop one to simulate a sparse OSM pull.
    gdf = gdf.drop(columns=["width"])
    text = coverage_report(gdf)
    assert "column absent" in text
    assert "width" in text


# --- _slugify_area ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Bengaluru, India", "bengaluru_india"),
        ("  Indiranagar  ", "indiranagar"),
        ("Koramangala 4th Block", "koramangala_4th_block"),
        ("HSR Layout — Sector 1", "hsr_layout_sector_1"),
        ("", "unnamed"),
        ("   ", "unnamed"),
    ],
)
def test_slugify_area(raw, expected):
    assert _slugify_area(raw) == expected


# --- _resolve_polygon dispatch (mocked) ------------------------------------


def test_resolve_polygon_dispatches_place_name_to_geocode():
    fake_polygon = Polygon([(77.5, 12.9), (77.6, 12.9), (77.6, 13.0)])
    # ``_resolve_polygon`` does ``import osmnx as ox`` *inside* the
    # function, so the right place to patch is the osmnx module
    # itself (wherever it's looked up at call time).
    with patch(
        "osmnx.geocode_to_gdf", return_value=_gdf_with_geometry(fake_polygon)
    ) as mock_geo:
        result = _resolve_polygon("Indiranagar, Bengaluru, India")
    assert result.equals(fake_polygon)
    mock_geo.assert_called_once()


def test_resolve_polygon_dispatches_bbox_to_box():
    bbox = (12.93, 77.55, 12.99, 77.65)
    result = _resolve_polygon(bbox)
    # shapely.box(minx=77.55, miny=12.93, maxx=77.65, maxy=12.99)
    assert result.equals(box(77.55, 12.93, 77.65, 12.99))


def test_resolve_polygon_rejects_empty_place_name():
    with pytest.raises(ValueError, match="empty"):
        _resolve_polygon("")


def test_resolve_polygon_rejects_malformed_bbox():
    with pytest.raises(ValueError, match="must be"):
        _resolve_polygon((12.9, 77.5))  # only 2 elements


def test_resolve_polygon_rejects_inverted_bbox():
    with pytest.raises(ValueError, match="min_lat"):
        _resolve_polygon((13.0, 77.5, 12.9, 77.6))  # min_lat > max_lat
    with pytest.raises(ValueError, match="min_lng"):
        _resolve_polygon((12.9, 77.6, 13.0, 77.5))  # min_lng > max_lng


def test_resolve_polygon_rejects_out_of_range():
    with pytest.raises(ValueError, match="lat out of range"):
        _resolve_polygon((12.0, 77.0, 91.0, 78.0))


def test_resolve_polygon_rejects_wrong_type():
    with pytest.raises(TypeError, match="place string or"):
        _resolve_polygon(42)


# --- load_osm_footpaths validation -----------------------------------------


def test_load_osm_footpaths_propagates_validation_error():
    with pytest.raises(ValueError, match="empty"):
        load_osm_footpaths("")


# --- small fixture for the dispatch test ----------------------------------


def _gdf_with_geometry(geom):
    return gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")


# --- smoke: default place string is non-empty -----------------------------


def test_default_place_is_set():
    assert DEFAULT_PLACE
    assert "Bengaluru" in DEFAULT_PLACE
