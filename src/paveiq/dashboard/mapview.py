"""pydeck layer construction for the 3D map and ward choropleth.

pydeck's ``extruded=True`` elevation only works on polygons, but a
footpath segment is a LineString. We give it volume by buffering
each segment into a thin rectangular ribbon polygon **while still in
a metric CRS** (buffering in degrees would distort ribbon width by
latitude), then reprojecting to WGS84 for pydeck/deck.gl.

Height convention: worse segments are *taller* (an inverse mapping),
so problem footpaths visually spike up without needing a legend to
explain a positive-height-is-good convention — this matches the
project's "worst-off segments surfaced first" goal.

This module never imports ``streamlit`` and constructs plain
``pdk.Layer``/``pdk.Deck`` objects, so it's testable (including
JSON serialization) without a browser.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pydeck as pdk
from matplotlib import colormaps

from paveiq.dashboard.data import to_wgs84

RIBBON_HALFWIDTH_M = 1.5
MIN_ELEVATION_M = 2.0
MAX_ELEVATION_M = 120.0

BENGALURU_VIEW_STATE = pdk.ViewState(latitude=12.9716, longitude=77.5946, zoom=12, pitch=45)

_COLORMAP = colormaps["RdYlGn"]


def score_to_rgba(scores) -> np.ndarray:
    """Map scores in ``[0, 100]`` to an (N, 4) uint8 RGBA array.

    Uses matplotlib's ``RdYlGn`` diverging colormap: score 0 -> red
    end, score 100 -> green end, monotonic in between.
    """
    normalized = np.clip(np.asarray(scores, dtype=float) / 100.0, 0.0, 1.0)
    return (_COLORMAP(normalized) * 255).astype(np.uint8)


def score_to_elevation(
    scores,
    min_elevation_m: float = MIN_ELEVATION_M,
    max_elevation_m: float = MAX_ELEVATION_M,
) -> np.ndarray:
    """Inverse-map scores to elevation: worse score -> taller ribbon."""
    clipped = np.clip(np.asarray(scores, dtype=float), 0.0, 100.0)
    return min_elevation_m + (100.0 - clipped) / 100.0 * (max_elevation_m - min_elevation_m)


def _buffer_to_ribbons(gdf: gpd.GeoDataFrame, halfwidth_m: float) -> gpd.GeoDataFrame:
    """Buffer each (line) geometry into a flat-capped ribbon polygon.

    Must run before any reprojection to WGS84 — buffering in degrees
    would make the ribbon width vary with latitude.
    """
    return gdf.assign(geometry=gdf.buffer(halfwidth_m, cap_style="flat"))


def build_segment_layer(
    scored_gdf: gpd.GeoDataFrame,
    ribbon_halfwidth_m: float = RIBBON_HALFWIDTH_M,
    min_elevation_m: float = MIN_ELEVATION_M,
    max_elevation_m: float = MAX_ELEVATION_M,
) -> pdk.Layer:
    """Build the extruded 3D segment-ribbon layer.

    ``scored_gdf`` must be in a metric CRS (e.g. ``EPSG:32643``) and
    have a ``score`` column; any other columns are carried through as
    GeoJSON properties (useful for tooltips).
    """
    if "score" not in scored_gdf.columns:
        raise ValueError("scored_gdf must have a 'score' column")

    ribbons = _buffer_to_ribbons(scored_gdf, ribbon_halfwidth_m)
    ribbons_wgs84 = to_wgs84(ribbons)
    scores = ribbons_wgs84["score"].to_numpy()
    ribbons_wgs84 = ribbons_wgs84.assign(
        elevation=score_to_elevation(scores, min_elevation_m, max_elevation_m),
        fill_color=[c.tolist() for c in score_to_rgba(scores)],
    )
    geojson = json.loads(ribbons_wgs84.to_json())
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        extruded=True,
        get_elevation="properties.elevation",
        get_fill_color="properties.fill_color",
        pickable=True,
        auto_highlight=True,
    )


def build_ward_choropleth_layer(ward_gdf_with_scores: gpd.GeoDataFrame) -> pdk.Layer:
    """Build a flat, filled choropleth layer over ward polygons.

    ``ward_gdf_with_scores`` must have a ``mean_score`` column (e.g.
    the ward polygons joined against ``dashboard.data.ward_summary``'s
    output). Kept as a separate layer/tab from the extruded segment
    layer rather than combined, to avoid visual noise.
    """
    if "mean_score" not in ward_gdf_with_scores.columns:
        raise ValueError("ward_gdf_with_scores must have a 'mean_score' column")

    wards_wgs84 = to_wgs84(ward_gdf_with_scores)
    colors = score_to_rgba(wards_wgs84["mean_score"].to_numpy())
    wards_wgs84 = wards_wgs84.assign(fill_color=[c.tolist() for c in colors])
    geojson = json.loads(wards_wgs84.to_json())
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        extruded=False,
        stroked=True,
        get_fill_color="properties.fill_color",
        get_line_color=[80, 80, 80, 200],
        line_width_min_pixels=1,
        pickable=True,
    )


def make_deck(layers, view_state: pdk.ViewState = None, map_style=None, tooltip=None) -> pdk.Deck:
    """Assemble a ``pdk.Deck`` on the CARTO basemap (no Mapbox token needed).

    ``map_style`` defaults to the light CARTO tiles; pass e.g.
    ``pdk.map_styles.CARTO_DARK`` for a dark basemap (both are free,
    same ``map_provider="carto"``, no Mapbox token either way).
    """
    return pdk.Deck(
        layers=list(layers),
        initial_view_state=view_state or BENGALURU_VIEW_STATE,
        map_provider="carto",
        map_style=map_style or pdk.map_styles.LIGHT,
        tooltip=tooltip if tooltip is not None else True,
    )
