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
import shapely
from matplotlib import colormaps

from paveiq.dashboard.data import to_wgs84

RIBBON_HALFWIDTH_M = 1.5
MIN_ELEVATION_M = 2.0
MAX_ELEVATION_M = 120.0

# A real deploy hit a WebSocket message-size limit (~64 MiB) sending the
# per-segment ribbon GeoJSON for ~22k segments: pydeck's to_json() always
# pretty-prints with indent=2, and carrying every scored-table column
# through as a GeoJSON property multiplies an already-large coordinate
# payload (measured ~69 MB for this dataset). These two levers plus
# trimming properties (in build_segment_layer) cut that to ~27 MB with no
# visible fidelity loss at map-viewing scale:
SEGMENT_SIMPLIFY_TOLERANCE_M = 1.0  # collapse near-collinear points on the source line
COORDINATE_PRECISION_DEG = 1e-6  # ~11cm; ribbons are already 3m wide

# Raw columns pulled in only to *derive* human-readable tooltip fields below
# (the raw ordinal/tag values themselves don't survive into the output
# properties -- see build_segment_layer).
_SEGMENT_TOOLTIP_INPUT_COLS = ("highway", "surface_quality")

# Same 3-tier mapping build_features.surface_quality() encodes, inverted for
# display. Cast to float64 + round before mapping: surface_quality is stored
# as float32, and a raw float32(0.6)/float32(0.2) can hash differently from
# the Python float literal below even though `==` says they're equal --
# a dict-based .map() silently misses and returns NaN for every row that
# isn't exactly 1.0 unless the dtype is normalized first.
_SURFACE_QUALITY_LABELS = {1.0: "Paved", 0.6: "Compacted", 0.2: "Unpaved"}

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
    would make the ribbon width vary with latitude. ``join_style="mitre"``
    (vs. shapely's default ``"round"``) avoids adding extra points at
    each bend to approximate a circular arc — a real vertex-count saving
    at city scale, invisible on a 3m-wide ribbon.
    """
    return gdf.assign(geometry=gdf.buffer(halfwidth_m, cap_style="flat", join_style="mitre"))


def build_segment_layer(
    scored_gdf: gpd.GeoDataFrame,
    ribbon_halfwidth_m: float = RIBBON_HALFWIDTH_M,
    min_elevation_m: float = MIN_ELEVATION_M,
    max_elevation_m: float = MAX_ELEVATION_M,
) -> pdk.Layer:
    """Build the extruded 3D segment-ribbon layer.

    ``scored_gdf`` must be in a metric CRS (e.g. ``EPSG:32643``) and
    have a ``score`` column. Only ``score`` (rounded to 1 decimal for
    display) and human-readable tooltip fields derived from
    ``_SEGMENT_TOOLTIP_INPUT_COLS`` (``highway_label``, ``surface_label``)
    are carried through as GeoJSON properties — see the module-level
    comment on ``SEGMENT_SIMPLIFY_TOLERANCE_M`` for why: at ~22k
    segments, the full scored table's payload is large enough to
    exceed the browser's WebSocket message-size limit.
    """
    if "score" not in scored_gdf.columns:
        raise ValueError("scored_gdf must have a 'score' column")

    input_cols = [c for c in _SEGMENT_TOOLTIP_INPUT_COLS if c in scored_gdf.columns]
    trimmed = scored_gdf[["score", "geometry"] + input_cols].copy()
    trimmed["geometry"] = trimmed.geometry.simplify(SEGMENT_SIMPLIFY_TOLERANCE_M)

    if "highway" in trimmed.columns:
        trimmed["highway_label"] = trimmed.pop("highway").fillna("Unknown")
    if "surface_quality" in trimmed.columns:
        trimmed["surface_label"] = (
            trimmed.pop("surface_quality").astype("float64").round(1).map(_SURFACE_QUALITY_LABELS).fillna("Unknown")
        )

    ribbons = _buffer_to_ribbons(trimmed, ribbon_halfwidth_m)
    ribbons_wgs84 = to_wgs84(ribbons)
    ribbons_wgs84 = ribbons_wgs84.set_geometry(
        shapely.set_precision(ribbons_wgs84.geometry.values, grid_size=COORDINATE_PRECISION_DEG)
    )
    scores = ribbons_wgs84["score"].to_numpy()
    ribbons_wgs84 = ribbons_wgs84.assign(
        score=np.round(scores, 1),
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
