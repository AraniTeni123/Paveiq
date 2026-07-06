"""Feature engineering over OSM footpath segments.

Stage 2 of the PaveIQ pipeline. Reads the raw GeoJSON produced by
``osm_loader`` and emits a per-segment feature Parquet that the
modeling and scoring stages consume.

What the module does
--------------------
- Reprojects the input to the project target CRS (``EPSG:32643``)
  so lengths are in metres.
- Parses the ``width`` tag (which is dirty in OSM: ``"20'"``,
  ``"25 ft"``, etc.) into a float metres column.
- Encodes ``highway`` as a 0–1 footpath-likelihood score.
- Encodes ``surface`` as an ordinal quality score (paved / compacted
  / unpaved / null).
- Encodes ``sidewalk`` as a 5-bucket categorical that is conditional
  on ``highway`` (a ``footway`` with no sidewalk tag is implicitly a
  footpath; a ``primary`` with no sidewalk tag is "unlikely").
- Computes ``length_m`` in UTM 43N metres.

The output is a Parquet with the schema documented below; downstream
stages can read it with ``geopandas.read_parquet(...)``.

Output schema
-------------
================  =========  ==========================================
Column            dtype      Notes
================  =========  ==========================================
osmid             str        comma-joined if a way has multiple ids
highway           str        raw OSM value, may be null
highway_likelihood float32   0..1, NaN where highway is unknown
length_m          float32    geometry length in UTM 43N metres
width_m           float32    parsed to metres, NaN where unparseable
surface_quality   float32    1.0 (paved) / 0.6 (compacted) / 0.2 (unpaved) / NaN
sidewalk_presence str        5-bucket categorical
geometry          LineString EPSG:32643
================  =========  ==========================================

CLI
---
From the project root::

    python -m paveiq.features.build_features
    python -m paveiq.features.build_features --file path/to/raw.geojson
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from paveiq.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TARGET_CRS


# --- Pure transform helpers (the encoders) --------------------------------
# These are top-level functions so tests can target each one directly
# without instantiating a GeoDataFrame.


# Width: clamp to plausible physical range. <0.1 m and >100 m are
# almost always data-entry errors (footpath widths in the wild run
# roughly 0.5 m to ~10 m, road widths top out around 50 m).
_WIDTH_MIN_M = 0.1
_WIDTH_MAX_M = 100.0

# Captures: number (int or float, optional decimals), optional unit.
# Unit tokens we accept: m, cm, ft, single-quote (foot), double-quote (inch).
# Anything else fails the match and the parser returns NaN.
_WIDTH_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|ft|\'|\")?\s*$",
    flags=re.IGNORECASE,
)

# m -> 1, cm -> 0.01, mm -> 0.001, ft -> 0.3048, ' -> 0.3048, " -> 0.0254
_UNIT_TO_M = {
    None: 1.0,
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "ft": 0.3048,
    "'": 0.3048,
    '"': 0.0254,
}


def parse_width_m(raw) -> float:
    """Parse an OSM width tag value into float metres.

    Accepts bare numbers (``"2.5"``), explicit metres (``"2 m"``),
    centimetres (``"100 cm"``), feet (``"25 ft"`` or ``"20'"``),
    inches (``"24\""``), and millimetres (``"500 mm"``). Returns
    ``float('nan')`` for null/empty/garbage/implausible values.

    Parameters
    ----------
    raw : str or None or float
        The raw tag value. Numerics are passed through.

    Returns
    -------
    float
        Width in metres, or NaN.
    """
    if raw is None:
        return float("nan")
    if isinstance(raw, (int, float)):
        v = float(raw)
        return v if _WIDTH_MIN_M <= v <= _WIDTH_MAX_M else float("nan")
    if not isinstance(raw, str):
        return float("nan")
    s = raw.strip()
    if not s:
        return float("nan")
    m = _WIDTH_RE.match(s)
    if not m:
        return float("nan")
    value = float(m.group(1))
    unit = m.group(2)
    unit_factor = _UNIT_TO_M.get(unit.lower() if unit else None, 1.0)
    metres = value * unit_factor
    if not (_WIDTH_MIN_M <= metres <= _WIDTH_MAX_M):
        return float("nan")
    return metres


# Highway -> footpath likelihood. 0..1; 1 = "this is a footpath",
# 0 = "this is a highway where footpaths are very unlikely to live".
# *_link values are one step below their parent class.
_BASE_HIGHWAY_LIKELIHOOD = {
    "footway": 1.00,
    "path": 0.95,
    "pedestrian": 0.90,
    "living_street": 0.85,
    "steps": 0.85,
    "cycleway": 0.60,
    "residential": 0.50,
    "service": 0.30,
    "unclassified": 0.30,
    "tertiary": 0.20,
    "secondary": 0.15,
    "primary": 0.10,
    "trunk": 0.05,
    "construction": 0.20,  # ambiguous; treat like tertiary
}

_LINK_PARENT = {
    "primary_link": "primary",
    "secondary_link": "secondary",
    "tertiary_link": "tertiary",
    "trunk_link": "trunk",
}


def highway_likelihood(highway) -> float:
    """Map an OSM ``highway`` value to a 0..1 footpath likelihood.

    Returns NaN for null/unknown values so the model can treat them
    as missing rather than zero (zero would be a strong negative
    signal, not a missingness signal).
    """
    if highway is None or (isinstance(highway, float) and pd.isna(highway)):
        return float("nan")
    s = str(highway).strip().lower()
    if not s:
        return float("nan")
    if s in _BASE_HIGHWAY_LIKELIHOOD:
        return _BASE_HIGHWAY_LIKELIHOOD[s]
    if s in _LINK_PARENT:
        parent = _LINK_PARENT[s]
        return max(0.0, _BASE_HIGHWAY_LIKELIHOOD[parent] - 0.05)
    return float("nan")


# Surface -> ordinal quality. Three tiers:
#   paved      (1.0) — surfaces a wheelchair / stroller can roll on
#   compacted  (0.6) — hard to roll on but not loose
#   unpaved    (0.2) — dirt, gravel, mud; roll-resistant
_SURFACE_PAVED = {
    "asphalt", "paving_stones", "concrete", "paved", "sett", "cobblestone",
    "tiles", "clinker_plates", "stone", "cement", "concrete:plates",
}
_SURFACE_COMPACTED = {"compacted"}
_SURFACE_UNPAVED = {"unpaved", "dirt", "ground", "gravel", "rock", "mud", "ston"}


def surface_quality(surface) -> float:
    """Map an OSM ``surface`` value to an ordinal quality score in [0, 1]."""
    if surface is None or (isinstance(surface, float) and pd.isna(surface)):
        return float("nan")
    s = str(surface).strip().lower()
    if not s:
        return float("nan")
    if s in _SURFACE_PAVED:
        return 1.0
    if s in _SURFACE_COMPACTED:
        return 0.6
    if s in _SURFACE_UNPAVED:
        return 0.2
    return float("nan")


# Sidewalk presence: 5 mutually-exclusive buckets. The logic:
#   1. If sidewalk tag says yes (both/left/right/separate) -> explicit_present
#   2. If sidewalk tag says no -> explicit_absent
#   3. If highway is a "definitely-a-footpath" type with no sidewalk tag
#      -> implicit_present (the way IS the footpath; you don't tag it)
#   4. If highway is residential/service/etc (a road that often has a sidewalk)
#      -> likely_present (a prior, not a measurement)
#   5. Everything else -> unlikely (highways, primary, trunk, null highway)
_SIDEWALK_PRESENT_VALUES = {"both", "left", "right", "separate"}
_IMPLICIT_FOOTPATH_HIGHWAY = {"footway", "path", "pedestrian", "steps", "cycleway"}
_LIKELY_FOOTPATH_HIGHWAY = {"living_street", "residential", "service", "unclassified"}
_UNLIKELY_FOOTPATH_HIGHWAY = {
    "tertiary", "secondary", "primary", "trunk",
    "tertiary_link", "secondary_link", "primary_link", "trunk_link",
    "construction",
}


def sidewalk_presence(highway, sidewalk) -> str:
    """Compute the 5-bucket sidewalk presence categorical.

    The bucket captures the *measurement* of whether a sidewalk is
    present; the prior based on highway type is folded in so the
    model can distinguish "footway without explicit tag" (definitely
    a footpath) from "primary without explicit tag" (almost certainly
    no footpath).
    """
    if sidewalk is not None and not (isinstance(sidewalk, float) and pd.isna(sidewalk)):
        s = str(sidewalk).strip().lower()
        if s in _SIDEWALK_PRESENT_VALUES:
            return "explicit_present"
        if s == "no":
            return "explicit_absent"
        # Any other sidewalk value (e.g. "yes" — rare) is treated as explicit_present.
        if s:
            return "explicit_present"

    # No informative sidewalk tag; fall through to the highway prior.
    h = "" if highway is None or (isinstance(highway, float) and pd.isna(highway)) else str(highway).strip().lower()
    if h in _IMPLICIT_FOOTPATH_HIGHWAY:
        return "implicit_present"
    if h in _LIKELY_FOOTPATH_HIGHWAY:
        return "likely_present"
    return "unlikely"


# --- Pipeline --------------------------------------------------------------


# Engineered features that go into the coverage report, in display order.
ENGINEERED_FEATURES = (
    "highway_likelihood",
    "length_m",
    "width_m",
    "surface_quality",
    "sidewalk_presence",
)


def _ensure_crs(gdf: gpd.GeoDataFrame, expected: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Reproject ``gdf`` to ``expected`` if it has a different CRS.

    OSMnx returns EPSG:4326, but bbox-derived pulls can occasionally
    inherit a non-WGS84 CRS — we don't want to silently compute
    wrong lengths in that case.
    """
    if gdf.crs is None:
        return gdf.set_crs(expected)
    if gdf.crs.to_epsg() != 4326:
        return gdf.to_crs(expected)
    return gdf


def _coerce_osmid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Turn the ``osmid`` column (often a list of ints) into a string."""
    if "osmid" not in gdf.columns:
        gdf["osmid"] = None
        return gdf

    def _to_str(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, (list, tuple, set)):
            return ",".join(str(int(x)) for x in sorted(v) if x is not None)
        return str(int(v))

    gdf["osmid"] = gdf["osmid"].apply(_to_str)
    return gdf


def build_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Pure transform: raw OSM GDF -> engineered feature GDF.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Raw OSM ways. CRS should be EPSG:4326 (auto-reprojected if not).

    Returns
    -------
    gpd.GeoDataFrame
        Reprojected to ``TARGET_CRS`` (EPSG:32643) with the schema
        documented at module top.
    """
    gdf = _ensure_crs(gdf)
    gdf = gdf.to_crs(TARGET_CRS)
    gdf = _coerce_osmid(gdf)

    # Vectorised column-wise encoding. Each row gets a scalar value;
    # the column is float32 / str as appropriate.
    out = pd.DataFrame(index=gdf.index)
    out["osmid"] = gdf["osmid"]
    out["highway"] = gdf.get("highway")
    out["highway_likelihood"] = gdf["highway"].apply(highway_likelihood).astype("float32")
    out["length_m"] = gdf.geometry.length.astype("float32")
    out["width_m"] = gdf.get("width", pd.Series([None] * len(gdf))).apply(parse_width_m).astype("float32")
    out["surface_quality"] = gdf.get("surface", pd.Series([None] * len(gdf))).apply(surface_quality).astype("float32")
    out["sidewalk_presence"] = [
        sidewalk_presence(h, s)
        for h, s in zip(out["highway"], gdf.get("sidewalk", pd.Series([None] * len(gdf))))
    ]
    out = gpd.GeoDataFrame(out, geometry=gdf.geometry, crs=TARGET_CRS)
    return out


def feature_coverage_report(feat_gdf: gpd.GeoDataFrame) -> str:
    """Multi-line coverage report for the engineered feature columns."""
    n = len(feat_gdf)
    if n == 0:
        return "Engineered feature coverage: 0 rows."
    lines = [f"Engineered feature coverage (non-null count / total = {n:,}):"]
    for col in ENGINEERED_FEATURES:
        if col not in feat_gdf.columns:
            lines.append(f"  {col:<22} [column absent]")
            continue
        non_null = int(feat_gdf[col].notna().sum())
        pct = 100.0 * non_null / n
        lines.append(f"  {col:<22} {non_null:>7,}/{n:<,}  ({pct:5.2f}%)")
    return "\n".join(lines)


def find_latest_raw(raw_dir: Optional[Path] = None) -> Path:
    """Return the most-recently-modified ``*_footpaths.geojson`` in ``raw_dir``."""
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir does not exist: {raw_dir}")
    candidates = sorted(raw_dir.glob("*_footpaths.geojson"))
    if not candidates:
        raise FileNotFoundError(
            f"no `*_footpaths.geojson` files in {raw_dir}; run the OSM loader first"
        )
    # Newest mtime wins.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_latest_raw(raw_dir: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Find the latest raw GeoJSON and read it as a GeoDataFrame."""
    return gpd.read_file(find_latest_raw(raw_dir))


def process_file(in_path: Path, out_dir: Optional[Path] = None) -> Path:
    """Read ``in_path``, build features, write Parquet to ``out_dir``.

    Returns the Parquet path. Also prints the coverage report to stdout.
    """
    out_dir = Path(out_dir) if out_dir is not None else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = gpd.read_file(in_path)
    feat = build_features(raw)

    out_name = in_path.stem.replace("_footpaths", "") + "_features.parquet"
    out_path = out_dir / out_name
    feat.to_parquet(out_path)
    return out_path


# --- CLI -------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paveiq.features.build_features",
        description=(
            "Build per-segment features from the raw OSM GeoJSON, "
            "write to data/processed/, and print a feature-coverage report."
        ),
    )
    p.add_argument(
        "--in-dir",
        default=None,
        help="Directory containing raw `*_footpaths.geojson` (default: data/raw/).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write features Parquet (default: data/processed/).",
    )
    p.add_argument(
        "--file",
        default=None,
        help="Explicit input file. Overrides the 'latest' pick.",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    in_dir = Path(args.in_dir) if args.in_dir else None
    out_dir = Path(args.out_dir) if args.out_dir else None

    try:
        in_path = Path(args.file) if args.file else find_latest_raw(in_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        out_path = process_file(in_path, out_dir=out_dir)
    except Exception as e:
        print(f"error: failed to build features: {e}", file=sys.stderr)
        return 1

    # Re-read to print the coverage report (cheap on Parquet; keeps the
    # contract simple — process_file returns a path, the caller decides
    # what to do with the GDF).
    feat = gpd.read_parquet(out_path)
    print(f"Wrote {out_path}")
    print(feature_coverage_report(feat))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
