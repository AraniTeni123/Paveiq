"""BBMP ward boundary ingestion and spatial join.

Stage 1.5 of the PaveIQ pipeline (or stage 3 of the data path).
Fetches the official BBMP 2022 ward polygons from the DataMeet
Municipal Spatial Data repository, caches them locally, and
joins each footpath segment to the ward it lies in.

The DataMeet file is preferred over OpenStreetMap's
``admin_level=6`` query because OSM only carries a partial /
ambiguous mapping of BBMP wards. The DataMeet file is the
KSRSAC 2022 delimitation (243 wards), CC-BY-SA 2.5 India
licensed, and is the canonical civic-data source for Bengaluru
wards.

Spatial-join strategy
--------------------
A segment that crosses a ward boundary would match multiple
wards under a naive ``intersects``. We avoid that by computing
each segment's ``shapely.representative_point`` (a guaranteed-
interior point) and joining the *point* to the ward polygon.
A point is in at most one ward for a clean polygon layer, but
the real BBMP 2022 file has a handful of slivers where two
*individually valid* adjacent ward polygons overlap along their
shared edge (a topology error in the source data, distinct from
the ~6 wards that fail ``is_valid`` outright). A representative
point landing in one of those slivers matches more than one
ward; we break the tie deterministically by keeping the first
match, so every segment still gets exactly one ward. Segments
whose representative point falls outside every ward (rare for
a bbox inside Bengaluru) are kept in the output with empty
``ward_id`` / ``ward_name`` and ``NaN`` ``ward_lgd_code``; the
coverage report surfaces them.

Output schema
-------------
The joined GeoDataFrame has the original 8 columns from
``build_features`` plus three new ones:

================  ======  ========================================
Column            dtype   Notes
================  ======  ========================================
ward_id           str     KGISWardNo, e.g. "186" (leading zeros kept)
ward_name         str     KGISWardName, e.g. "Koramangala"
ward_lgd_code     Int64   LGD_WardCode, the national-canonical code
================  ======  ========================================

CLI
---
::

    python -m paveiq.data_ingestion.ward_boundaries
    python -m paveiq.data_ingestion.ward_boundaries --features path/to/features.parquet
    python -m paveiq.data_ingestion.ward_boundaries --force-download
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from paveiq.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


# --- Constants -------------------------------------------------------------

# DataMeet / Municipal Spatial Data — Bangalore / BBMP.geojson
# 2022 delimitation: 243 wards. CC-BY-SA 2.5 India.
WARDS_URL = (
    "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/"
    "master/Bangalore/BBMP.geojson"
)
WARDS_FILENAME = "bbmp_wards_2022.geojson"
WARDS_VERSION = "2022"

# Final ward columns in the joined Parquet, in this order.
WARD_OUTPUT_COLS = ("ward_id", "ward_name", "ward_lgd_code")

# Source column names in the DataMeet GeoJSON. Kept as constants so
# the loader fails loudly if the upstream schema ever changes.
_SRC_WARD_ID = "KGISWardNo"
_SRC_WARD_NAME = "KGISWardName"
_SRC_WARD_LGD = "LGD_WardCode"


# --- Fetch / cache ---------------------------------------------------------


def _ward_path(out_dir: Path) -> Path:
    return Path(out_dir) / WARDS_FILENAME


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` with a sensible User-Agent.

    Uses urllib (stdlib) — no extra dep. Writes atomically via a
    temp file so a partial download doesn't leave a corrupt cache.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "paveiq/0.1 (https://github.com/bengawalk/paveiq)"},
    )
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def fetch_wards(out_dir: Optional[Path] = None, *, force: bool = False) -> Path:
    """Download the BBMP wards GeoJSON to ``out_dir`` and return its path.

    Caches locally: a second call with the same ``out_dir`` won't
    re-download unless ``force=True``.
    """
    if out_dir is None:
        out_dir = RAW_DATA_DIR
    out_dir = Path(out_dir)
    dest = _ward_path(out_dir)
    if dest.exists() and not force:
        return dest
    _download(WARDS_URL, dest)
    return dest


# --- Load / standardise ----------------------------------------------------


def _coerce_ward_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Pick the 3 KGIS columns we care about, rename, and coerce dtypes.

    LGD_WardCode uses pandas' nullable Int64 (not the default int64)
    because some wards in older DataMeet revisions have a null
    LGD code; casting to int64 would convert that to ``float('nan')``
    and break the downcast to plain int. Int64 keeps the NaN intact.
    """
    missing = [c for c in (_SRC_WARD_ID, _SRC_WARD_NAME, _SRC_WARD_LGD) if c not in gdf.columns]
    if missing:
        raise ValueError(
            f"ward GeoJSON is missing expected source columns: {missing}. "
            f"Has the upstream schema changed? See {WARDS_URL}."
        )

    out = gpd.GeoDataFrame(
        {
            "ward_id": gdf[_SRC_WARD_ID].astype(str),
            "ward_name": gdf[_SRC_WARD_NAME].astype(str),
            # Int64 (capital I) is the nullable integer extension dtype.
            "ward_lgd_code": pd.array(gdf[_SRC_WARD_LGD], dtype="Int64"),
        },
        geometry=gdf.geometry,
        crs=gdf.crs,
    )
    return out


def load_wards(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Read the cached ward GeoJSON and return a standardised GeoDataFrame.

    ``path`` defaults to the cached location (``data/raw/bbmp_wards_2022.geojson``).
    Downloads on demand if the file is missing.
    """
    if path is None:
        path = _ward_path(RAW_DATA_DIR)
        if not path.exists():
            fetch_wards(RAW_DATA_DIR)
    return _coerce_ward_schema(gpd.read_file(path))


# --- Spatial join ----------------------------------------------------------


def _representative_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a POINT GeoDataFrame at each feature's representative point.

    ``shapely.representative_point`` is guaranteed to be in the
    interior of a non-empty geometry, which means a single point
    is in at most one ward. We attach the original index so we
    can re-merge the ward assignment back onto the source GDF.
    """
    pts = [geom.representative_point() if geom is not None and not geom.is_empty else None
           for geom in gdf.geometry]
    out = gpd.GeoDataFrame(
        {"_orig_index": list(gdf.index)},
        geometry=[Point(*p.coords[0]) if p is not None else None for p in pts],
        crs=gdf.crs,
    )
    return out


def join_wards_to_features(
    features_gdf: gpd.GeoDataFrame,
    wards_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Augment ``features_gdf`` with ``ward_id`` / ``ward_name`` / ``ward_lgd_code``.

    Returns a new GeoDataFrame with the same row order and index as
    the input, plus the three ward columns. The output CRS is the
    input features' CRS (the wards are reprojected to it on demand).

    Segments whose representative point falls outside every ward
    get empty ``ward_id`` / ``ward_name`` and ``<NA>`` ``ward_lgd_code``.
    """
    if features_gdf.crs is None:
        raise ValueError("features_gdf has no CRS; cannot spatial-join")

    # Reproject wards to features' CRS if needed.
    if wards_gdf.crs is None:
        raise ValueError("wards_gdf has no CRS; cannot spatial-join")
    if wards_gdf.crs != features_gdf.crs:
        wards = wards_gdf.to_crs(features_gdf.crs)
    else:
        wards = wards_gdf

    points = _representative_points(features_gdf)

    # Spatial join: point in polygon. Use intersects (a point inside
    # a polygon is an intersection). how='left' so every feature keeps
    # a row even if it doesn't match a ward.
    joined = gpd.sjoin(
        points,
        wards[list(WARD_OUTPUT_COLS) + ["geometry"]],
        how="left",
        predicate="intersects",
    )

    # Drop the sjoin-generated point geometry and the right index;
    # we only want the ward columns. We index by the original
    # feature index (which we carried through as ``_orig_index``).
    joined = joined.drop(columns=["geometry"]).set_index("_orig_index")
    joined.index.name = features_gdf.index.name

    # A handful of real BBMP wards overlap along their shared edge
    # (see module docstring), which can make one representative point
    # match two wards. Keep the first match so every segment maps to
    # exactly one ward and `reindex` below never sees a duplicate label.
    joined = joined[~joined.index.duplicated(keep="first")]

    # Reindex to the input's full index so orphans (and segments
    # whose point is null) all appear in the result.
    ward_cols = joined.reindex(features_gdf.index)

    out = features_gdf.copy()
    for col in WARD_OUTPUT_COLS:
        out[col] = ward_cols[col].values
    return out


# --- File I/O helpers ------------------------------------------------------


def _find_latest_features(processed_dir: Optional[Path] = None) -> Path:
    """Return the newest ``*features.parquet`` in ``processed_dir``."""
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed dir does not exist: {processed_dir}")
    candidates = sorted(processed_dir.glob("*features.parquet"))
    # Exclude the with-wards output of a previous run if it slipped in.
    candidates = [p for p in candidates if "_with_wards" not in p.name]
    if not candidates:
        raise FileNotFoundError(
            f"no `*features.parquet` files in {processed_dir}; run build_features first"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def join_and_save(
    features_path: Path,
    wards_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """End-to-end: load features, load wards, join, write Parquet.

    Output path: ``<out_dir>/<features_stem>_with_wards.parquet``,
    e.g. ``bengaluru_india_features_with_wards.parquet``.
    """
    out_dir = Path(out_dir) if out_dir is not None else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    features = gpd.read_parquet(features_path)
    wards = load_wards(wards_path)
    joined = join_wards_to_features(features, wards)

    out_name = features_path.stem + "_with_wards.parquet"
    out_path = out_dir / out_name
    joined.to_parquet(out_path)
    return out_path


# --- Coverage report -------------------------------------------------------


def coverage_report(joined_gdf: gpd.GeoDataFrame) -> str:
    """Multi-line string summarising the ward join."""
    n = len(joined_gdf)
    if n == 0:
        return f"Ward join summary (BBMP {WARDS_VERSION}): 0 rows."

    matched_mask = joined_gdf["ward_id"].notna() & (joined_gdf["ward_id"] != "")
    n_matched = int(matched_mask.sum())
    n_orphan = n - n_matched
    distinct = int(joined_gdf.loc[matched_mask, "ward_id"].nunique())

    lines = [
        f"Ward join summary (BBMP {WARDS_VERSION}):",
        f"  total segments:                  {n:,}",
        f"  matched to a ward:               {n_matched:,}  ({100 * n_matched / n:5.2f}%)",
        f"  orphaned (no ward):              {n_orphan:,}  ({100 * n_orphan / n:5.2f}%)",
        f"  distinct wards touched:          {distinct:,}",
    ]

    if n_matched > 0:
        # ward_id is a str; show "<id> <name>" in the top list.
        matched = joined_gdf.loc[matched_mask, ["ward_id", "ward_name"]]
        top = (
            matched.groupby(["ward_id", "ward_name"])
            .size()
            .sort_values(ascending=False)
            .head(5)
        )
        lines.append("  top 5 wards by segment count:")
        for (wid, wname), count in top.items():
            lines.append(f"    Ward {wid} {wname:<30} {count:>7,}")
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paveiq.data_ingestion.ward_boundaries",
        description=(
            "Download BBMP ward boundaries and spatial-join them to "
            "the latest features Parquet."
        ),
    )
    p.add_argument(
        "--features",
        default=None,
        help=(
            "Path to a features Parquet (default: latest *features.parquet in data/processed/)."
        ),
    )
    p.add_argument(
        "--wards",
        default=None,
        help=(
            "Path to a cached wards GeoJSON (default: data/raw/bbmp_wards_2022.geojson; "
            "downloaded on first use)."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: data/processed/).",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the ward GeoJSON even if a cached copy exists.",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else None

    # Resolve features path.
    try:
        features_path = (
            Path(args.features) if args.features else _find_latest_features(out_dir)
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Resolve wards path; optionally force a re-download.
    wards_path: Optional[Path] = Path(args.wards) if args.wards else None
    if wards_path is None and args.force_download:
        fetch_wards(RAW_DATA_DIR, force=True)

    try:
        out_path = join_and_save(features_path, wards_path=wards_path, out_dir=out_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: failed to join wards: {e}", file=sys.stderr)
        return 1

    joined = gpd.read_parquet(out_path)
    print(f"Wrote {out_path}")
    print(coverage_report(joined))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
