"""OpenStreetMap footpath loader.

Stage 1 of the PaveIQ pipeline. Downloads OSM ways for a given
Bengaluru area, keeps the four quality-relevant tags (``sidewalk``,
``surface``, ``width``, ``smoothness``) plus geometry, writes a
GeoJSON to ``data/raw/``, and prints a tag-coverage report.

Bengaluru's OSM coverage of these tags is known to be sparse
(<5% of ways carry a ``sidewalk`` tag); the coverage report
makes that sparsity visible at a glance so downstream stages
know what they're working with.

Usage
-----
As a module::

    from paveiq.data_ingestion.osm_loader import load_osm_footpaths
    gdf = load_osm_footpaths("Indiranagar, Bengaluru, India")

From the CLI (from the project root)::

    python -m paveiq.data_ingestion.osm_loader
    python -m paveiq.data_ingestion.osm_loader --place "Koramangala, Bengaluru, India"
    python -m paveiq.data_ingestion.osm_loader --bbox 12.93 77.55 12.99 77.65
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
from shapely.geometry import Polygon, box

# Target tags we care about for footpath health scoring. Order is
# preserved in the coverage report so the printed output is stable.
TARGET_TAGS = ("sidewalk", "surface", "width", "smoothness")

# Highway values that *might* host a footpath or be a footpath. We
# include the full set so that a residential way carrying a
# ``sidewalk=*`` tag is captured. ``highway=footway`` alone is too
# narrow; many Bengaluru sidewalks are tagged on the parent road.
FOOTPATH_HIGHWAY_VALUES = (
    "footway",
    "path",
    "pedestrian",
    "living_street",
    "residential",
    "service",
    "unclassified",
    "tertiary",
    "tertiary_link",
    "secondary",
    "secondary_link",
    "primary",
    "primary_link",
    "trunk",
    "trunk_link",
)

DEFAULT_PLACE = "Bengaluru, India"

# Public alias so callers can type-hint against the union.
PlaceSpec = Union[str, Tuple[float, float, float, float]]


# --- Input parsing ---------------------------------------------------------


def _slugify_area(area: str) -> str:
    """Turn a human area name into a filesystem-safe slug.

    Examples
    --------
    >>> _slugify_area("Bengaluru, India")
    'bengaluru_india'
    >>> _slugify_area("  Koramangala 4th Block  ")
    'koramangala_4th_block'
    """
    s = area.strip().lower()
    # Replace any run of non-alphanumeric chars with a single underscore.
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unnamed"


def _validate_bbox(
    min_lat: float, min_lng: float, max_lat: float, max_lng: float
) -> None:
    """Reject bboxes that can't be turned into a sensible polygon.

    OSMnx / Overpass will happily send a query for a degenerate
    bbox; we catch obvious mistakes early with a clear error.
    """
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError(f"lat out of range: {min_lat}, {max_lat}")
    if not (-180 <= min_lng <= 180 and -180 <= max_lng <= 180):
        raise ValueError(f"lng out of range: {min_lng}, {max_lng}")
    if min_lat >= max_lat:
        raise ValueError(
            f"min_lat ({min_lat}) must be < max_lat ({max_lat}); "
            "remember the order is south, west, north, east"
        )
    if min_lng >= max_lng:
        raise ValueError(
            f"min_lng ({min_lng}) must be < max_lng ({max_lng})"
        )


# --- OSMnx interaction -----------------------------------------------------
# These are wrapped (not inlined into the public functions) so that
# tests can mock them with ``unittest.mock.patch`` without touching
# the network.


def _tags_of_interest() -> dict:
    """Build the OSMnx 2.x ``tags`` filter dict.

    In OSMnx 2.x, ``features.features_from_polygon(polygon, tags=...)``
    translates a Python dict into a single Overpass query. A value
    of ``True`` means "this key exists with any value", which is
    the way to OR-combine the highway-list filter with the
    "has a sidewalk tag" filter.
    """
    return {
        "highway": list(FOOTPATH_HIGHWAY_VALUES),
        "sidewalk": True,
        "surface": True,
        "width": True,
        "smoothness": True,
    }


def _resolve_polygon(area: PlaceSpec) -> Polygon:
    """Convert a place string or bbox tuple to a shapely Polygon.

    For a place string: geocode via OSMnx. For a bbox tuple
    ``(min_lat, min_lng, max_lat, max_lng)``: build a ``shapely.box``
    directly (south, west, north, east order — ``shapely.box`` is
    ``(minx, miny, maxx, maxy)`` = ``(min_lng, min_lat, max_lng, max_lat)``).
    """
    import osmnx as ox  # imported lazily so module import is cheap

    if isinstance(area, str):
        if not area.strip():
            raise ValueError("place name is empty")
        gdf = ox.geocode_to_gdf(area)
        # geocode_to_gdf returns the unioned geometry; take the first row.
        return gdf.geometry.iloc[0]
    elif isinstance(area, (tuple, list)):
        if len(area) != 4:
            raise ValueError(
                f"bbox must be (min_lat, min_lng, max_lat, max_lng); got {area!r}"
            )
        min_lat, min_lng, max_lat, max_lng = (float(x) for x in area)
        _validate_bbox(min_lat, min_lng, max_lat, max_lng)
        # shapely.box signature: (minx, miny, maxx, maxy) = (min_lng, min_lat, max_lng, max_lat)
        return box(min_lng, min_lat, max_lng, max_lat)
    else:
        raise TypeError(
            f"area must be a place string or (min_lat, min_lng, max_lat, max_lng) "
            f"tuple; got {type(area).__name__}"
        )


def _fetch_features(polygon: Polygon) -> gpd.GeoDataFrame:
    """Thin wrapper around ``osmnx.features.features_from_polygon``.

    Kept as a separate function so tests can patch it cleanly.
    """
    import osmnx as ox  # lazy import; the public path doesn't need it

    return ox.features.features_from_polygon(polygon, tags=_tags_of_interest())


def _filter_to_ways(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop nodes/relations; we only want ways (LineString / MultiLineString).

    OSMnx returns a mix of geometries tagged with the
    ``element_type`` column. We only want ways for network scoring.
    """
    if "element_type" not in gdf.columns:
        # Older OSMnx versions: filter on geometry type instead.
        return gdf.loc[
            gdf.geometry.geom_type.isin(("LineString", "MultiLineString"))
        ].copy()

    ways = gdf.loc[gdf["element_type"] == "way"].copy()
    if ways.empty:
        # Fall back to the geometry-type filter if the column is
        # somehow present but empty after filtering.
        return gdf.loc[
            gdf.geometry.geom_type.isin(("LineString", "MultiLineString"))
        ].copy()
    return ways


# --- Public API ------------------------------------------------------------


@dataclass
class LoadResult:
    """Bundle of a successful load — handy for callers and tests."""

    gdf: gpd.GeoDataFrame
    out_path: Path
    coverage_text: str


def load_osm_footpaths(area: PlaceSpec = DEFAULT_PLACE) -> gpd.GeoDataFrame:
    """Download OSM ways for ``area`` and return a filtered GeoDataFrame.

    Parameters
    ----------
    area : str or 4-tuple of float
        Either a place name (geocoded via OSMnx) or a bbox tuple
        ``(min_lat, min_lng, max_lat, max_lng)``.

    Returns
    -------
    geopandas.GeoDataFrame
        CRS is EPSG:4326 (OSM native). Reprojection is the
        features stage's job, not this one's. Columns include
        ``osmid``, ``highway``, ``name``, ``geometry``, and the
        four target tags where populated.
    """
    polygon = _resolve_polygon(area)
    raw = _fetch_features(polygon)
    ways = _filter_to_ways(raw)

    # Make sure the four target-tag columns exist even if no way
    # populated them; downstream code (and the coverage report)
    # can rely on them.
    for col in TARGET_TAGS:
        if col not in ways.columns:
            ways[col] = None

    # Keep only the columns that matter for the project, plus geometry.
    keep = [c for c in ("osmid", "highway", "name", *TARGET_TAGS) if c in ways.columns]
    ways = ways[keep + ["geometry"]].reset_index(drop=True)
    return ways


def save_geojson(gdf: gpd.GeoDataFrame, out_path: Union[str, Path]) -> Path:
    """Write ``gdf`` to ``out_path`` as GeoJSON; return the resolved Path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def coverage_report(gdf: gpd.GeoDataFrame) -> str:
    """Return a multi-line string summarising tag coverage.

    Pure function — easy to unit-test against a synthetic GeoDataFrame.
    """
    n = len(gdf)
    if n == 0:
        return "Tag coverage: 0 ways fetched; nothing to report."

    lines = [f"Fetched {n:,} ways.", "Tag coverage (non-null count / total):"]
    for tag in TARGET_TAGS:
        if tag not in gdf.columns:
            lines.append(f"  {tag:<10} 0/{n}  (0.0%)  [column absent]")
            continue
        non_null = int(gdf[tag].notna().sum())
        pct = 100.0 * non_null / n
        lines.append(f"  {tag:<10} {non_null:>6,}/{n:<,}  ({pct:5.1f}%)")
    return "\n".join(lines)


def load_and_save(
    area: PlaceSpec = DEFAULT_PLACE,
    out_dir: Optional[Union[str, Path]] = None,
) -> LoadResult:
    """End-to-end helper: load, save, and report in one call.

    ``out_dir`` defaults to ``data/raw/`` (resolved from the project
    root via :mod:`paveiq.config`).
    """
    from paveiq.config import RAW_DATA_DIR  # local import avoids a cycle at import time

    gdf = load_osm_footpaths(area)

    if out_dir is None:
        out_dir = RAW_DATA_DIR
    out_dir = Path(out_dir)

    if isinstance(area, str):
        slug = _slugify_area(area)
    else:
        # bbox tuple: 12.93_77.55_12.99_77.65
        slug = "_".join(f"{x:.4f}" for x in area)
    out_path = out_dir / f"{slug}_footpaths.geojson"

    save_geojson(gdf, out_path)
    text = coverage_report(gdf)
    return LoadResult(gdf=gdf, out_path=out_path, coverage_text=text)


# --- CLI -------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paveiq.data_ingestion.osm_loader",
        description=(
            "Download OSM ways for a Bengaluru area, save as GeoJSON, "
            "and print a tag-coverage report."
        ),
    )
    area = p.add_mutually_exclusive_group()
    area.add_argument(
        "--place",
        default=DEFAULT_PLACE,
        help=(
            f'Place name to geocode (default: "{DEFAULT_PLACE}"). '
            "Mutually exclusive with --bbox."
        ),
    )
    area.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MIN_LNG", "MAX_LAT", "MAX_LNG"),
        help="Bounding box in (south, west, north, east) order. Overrides --place.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: data/raw/ under the project root).",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    area: PlaceSpec
    if args.bbox is not None:
        area = tuple(args.bbox)
    else:
        area = args.place

    try:
        result = load_and_save(area=area, out_dir=args.out_dir)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # network errors, Overpass timeouts, etc.
        print(f"error: failed to fetch OSM data: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {result.out_path}")
    print(result.coverage_text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
