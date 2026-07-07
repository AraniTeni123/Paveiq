"""Offline tests for ``paveiq.dashboard.mapview``.

Constructs pydeck ``Layer``/``Deck`` objects and checks they serialize
(``.to_json()``) without a browser — that's the whole surface that can
be tested without actually rendering WebGL.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pydeck as pdk
import pytest
from shapely.geometry import LineString, Polygon

from paveiq.dashboard import mapview as mv


# --- score_to_rgba -----------------------------------------------------


def test_score_to_rgba_endpoints():
    rgba = mv.score_to_rgba([0, 100])
    # RdYlGn: low value -> red end, high value -> green end.
    red, green = rgba[0], rgba[1]
    assert red[0] > red[1]  # red channel dominant at score 0
    assert green[1] > green[0]  # green channel dominant at score 100


def test_score_to_rgba_shifts_from_red_to_green_dominance():
    """RdYlGn is a diverging colormap through yellow at the midpoint, so
    individual channels aren't monotonic end-to-end — but red-minus-green
    dominance should still shift monotonically red -> yellow -> green at
    the three natural anchor points."""
    rgba = mv.score_to_rgba([0, 50, 100]).astype(int)
    dominance = rgba[:, 0] - rgba[:, 1]  # R - G
    assert dominance[0] > dominance[1] > dominance[2]


def test_score_to_rgba_shape_and_dtype():
    rgba = mv.score_to_rgba([10, 50, 90])
    assert rgba.shape == (3, 4)
    assert rgba.dtype == np.uint8


# --- score_to_elevation -----------------------------------------------


def test_score_to_elevation_worse_is_taller():
    elev = mv.score_to_elevation([0, 100])
    assert elev[0] > elev[1]
    assert elev[0] == pytest.approx(mv.MAX_ELEVATION_M)
    assert elev[1] == pytest.approx(mv.MIN_ELEVATION_M)


def test_score_to_elevation_clips_out_of_range_scores():
    elev = mv.score_to_elevation([-50, 150])
    assert elev[0] == pytest.approx(mv.MAX_ELEVATION_M)
    assert elev[1] == pytest.approx(mv.MIN_ELEVATION_M)


# --- build_segment_layer -------------------------------------------------


def _scored_segments_utm() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "osmid": ["1", "2"],
            "score": [20.0, 90.0],
            "geometry": [LineString([(0, 0), (0, 100)]), LineString([(50, 0), (50, 100)])],
        },
        crs="EPSG:32643",
    )


def test_build_segment_layer_returns_valid_pdk_layer():
    layer = mv.build_segment_layer(_scored_segments_utm())
    assert isinstance(layer, pdk.Layer)
    assert layer.extruded is True


def test_build_segment_layer_serializes_to_json():
    layer = mv.build_segment_layer(_scored_segments_utm())
    deck = mv.make_deck([layer])
    # Must not raise — this is the whole point of the offline test.
    payload = json.loads(deck.to_json())
    assert "layers" in payload


def test_build_segment_layer_trims_to_score_and_tooltip_columns_only():
    """Regression guard: a real deploy hit a WebSocket message-size limit
    because every scored-table column (17 of them) was carried through as a
    GeoJSON property for ~22k segments. Only score + highway (what the map
    tooltip uses) should survive into the output properties."""
    gdf = gpd.GeoDataFrame(
        {
            "osmid": ["1", "2"],
            "highway": ["footway", "residential"],
            "score": [20.0, 90.0],
            "ward_name": ["Alpha", "Beta"],  # should NOT survive
            "sidewalk_presence": ["explicit_present", "unlikely"],  # should NOT survive
            "geometry": [LineString([(0, 0), (0, 100)]), LineString([(50, 0), (50, 100)])],
        },
        crs="EPSG:32643",
    )
    layer = mv.build_segment_layer(gdf)
    props = layer.data["features"][0]["properties"]
    assert "score" in props
    assert "highway" in props
    assert "elevation" in props
    assert "fill_color" in props
    assert "osmid" not in props
    assert "ward_name" not in props
    assert "sidewalk_presence" not in props


def test_build_segment_layer_simplifies_many_collinear_points():
    """A line with many redundant near-collinear points (like real OSM
    digitization) should end up with far fewer vertices after buffering,
    thanks to simplify() + the mitre join — this is the actual size fix."""
    dense_line = LineString([(0, y) for y in range(0, 101)])  # 101 collinear points
    gdf = gpd.GeoDataFrame({"score": [50.0], "geometry": [dense_line]}, crs="EPSG:32643")
    layer = mv.build_segment_layer(gdf)
    ribbon_coords = layer.data["features"][0]["geometry"]["coordinates"][0]
    # A straight ribbon needs only ~4-5 corners, not 101+ per side.
    assert len(ribbon_coords) < 20


def test_build_segment_layer_raises_without_score_column():
    gdf = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (0, 1)])]}, crs="EPSG:32643"
    )
    with pytest.raises(ValueError, match="score"):
        mv.build_segment_layer(gdf)


# --- build_ward_choropleth_layer -----------------------------------------


def _ward_polygons_with_scores() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "ward_id": ["A", "B"],
            "mean_score": [30.0, 80.0],
            "geometry": [
                Polygon([(77.60, 12.93), (77.61, 12.93), (77.61, 12.94), (77.60, 12.94)]),
                Polygon([(77.61, 12.93), (77.62, 12.93), (77.62, 12.94), (77.61, 12.94)]),
            ],
        },
        crs="EPSG:4326",
    )


def test_build_ward_choropleth_layer_returns_valid_pdk_layer():
    layer = mv.build_ward_choropleth_layer(_ward_polygons_with_scores())
    assert isinstance(layer, pdk.Layer)
    assert layer.extruded is False


def test_build_ward_choropleth_layer_serializes_to_json():
    layer = mv.build_ward_choropleth_layer(_ward_polygons_with_scores())
    deck = mv.make_deck([layer])
    payload = json.loads(deck.to_json())
    assert "layers" in payload


def test_build_ward_choropleth_layer_raises_without_mean_score_column():
    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="mean_score"):
        mv.build_ward_choropleth_layer(gdf)


# --- make_deck -----------------------------------------------------------


def test_make_deck_uses_carto_no_token():
    layer = mv.build_segment_layer(_scored_segments_utm())
    deck = mv.make_deck([layer])
    assert deck.map_provider == "carto"


def test_make_deck_combines_multiple_layers():
    seg_layer = mv.build_segment_layer(_scored_segments_utm())
    ward_layer = mv.build_ward_choropleth_layer(_ward_polygons_with_scores())
    deck = mv.make_deck([seg_layer, ward_layer])
    payload = json.loads(deck.to_json())
    assert len(payload["layers"]) == 2
