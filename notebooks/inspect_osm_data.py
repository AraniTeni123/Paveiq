"""Quick structural inspection of an OSM footpaths GeoJSON.

Reports:
  1. Total feature count and geometry-type distribution.
  2. Value counts (including nulls) for highway, sidewalk, surface, width, smoothness.
  3. A few example rows: the "richest" tag combinations and a few sparse ones.
  4. Geometry quality flags: orphaned segments, duplicate ways, self-intersections,
     empty geometries, zero-length segments.

Designed to print to stdout — no files written. Run from project root.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString

PATH = Path("data/raw/12.9300_77.5500_12.9900_77.6500_footpaths.geojson")

TAGS = ("highway", "sidewalk", "surface", "width", "smoothness")


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def value_counts(gdf: gpd.GeoDataFrame, col: str) -> None:
    """Print value counts including a null bucket."""
    if col not in gdf.columns:
        print(f"  (column '{col}' not present)")
        return
    n_total = len(gdf)
    n_null = int(gdf[col].isna().sum())
    counts = gdf[col].value_counts(dropna=True)
    print(f"  total={n_total:,}  null={n_null:,}  ({100 * n_null / n_total:.1f}%)")
    print(f"  non-null distinct values: {len(counts)}")
    # Top 15 non-null values
    for val, n in counts.head(15).items():
        pct = 100 * n / n_total
        print(f"    {n:>6,}  ({pct:5.2f}%)  {val!r}")
    if len(counts) > 15:
        print(f"    ... and {len(counts) - 15} more values")


def show_rows(gdf: gpd.GeoDataFrame, indices, label: str) -> None:
    print(f"\n  --- {label} ---")
    display_cols = [c for c in TAGS if c in gdf.columns] + ["osmid"]
    # Keep order: highway, sidewalk, surface, width, smoothness, osmid
    ordered = [c for c in ("highway", "sidewalk", "surface", "width", "smoothness") if c in gdf.columns]
    if "osmid" in gdf.columns:
        ordered.append("osmid")
    # Truncate long osmid lists
    for idx in indices:
        row = gdf.loc[idx]
        bits = []
        for c in ordered:
            v = row[c]
            if c == "osmid" and isinstance(v, list):
                v = f"[{len(v)} ids, first={v[0]}]"
            bits.append(f"{c}={v!r}")
        print(f"  [{idx}] " + ", ".join(bits))


def main() -> None:
    if not PATH.exists():
        raise SystemExit(f"not found: {PATH}")

    print(f"Reading {PATH} ...")
    gdf = gpd.read_file(PATH)
    print(f"  rows: {len(gdf):,}")
    print(f"  columns ({len(gdf.columns)}): {list(gdf.columns)}")
    print(f"  CRS: {gdf.crs}")

    # --- 1. Geometry types ------------------------------------------------
    section("1. Geometry types")
    geom_types = gdf.geometry.geom_type.value_counts(dropna=False)
    for gt, n in geom_types.items():
        print(f"  {gt:<20} {n:>7,}  ({100 * n / len(gdf):5.2f}%)")

    # --- 2. Tag value counts ----------------------------------------------
    section("2. Tag value counts (highway / sidewalk / surface / width / smoothness)")
    for col in TAGS:
        print(f"\n[{col}]")
        value_counts(gdf, col)

    # --- 3. Example rows --------------------------------------------------
    section("3. Example rows")

    # "Richness score": count of non-null values across the 4 quality tags (not highway).
    quality_cols = [c for c in ("sidewalk", "surface", "width", "smoothness") if c in gdf.columns]
    if quality_cols:
        gdf["_n_quality_tags"] = gdf[quality_cols].notna().sum(axis=1)
    else:
        gdf["_n_quality_tags"] = 0

    n_with_any = int((gdf["_n_quality_tags"] > 0).sum())
    n_full = int((gdf["_n_quality_tags"] == len(quality_cols)).sum())
    print(f"\n  ways with at least one quality tag: {n_with_any:,}  ({100 * n_with_any / len(gdf):.1f}%)")
    print(f"  ways with ALL {len(quality_cols)} quality tags populated:  {n_full:,}  ({100 * n_full / len(gdf):.1f}%)")
    print(f"  ways with ZERO quality tags: {len(gdf) - n_with_any:,}  ({100 * (len(gdf) - n_with_any) / len(gdf):.1f}%)")

    # Richest 3
    richest_idx = gdf.sort_values("_n_quality_tags", ascending=False).head(3).index.tolist()
    show_rows(gdf, richest_idx, "3 richest (most quality tags populated)")

    # Fully tagged 3 random sample
    if n_full > 3:
        full_sample = gdf[gdf["_n_quality_tags"] == len(quality_cols)].sample(
            min(3, n_full), random_state=0
        ).index.tolist()
        show_rows(gdf, full_sample, "3 random rows with all 4 quality tags")

    # 3 with only one quality tag
    one_tag = gdf[gdf["_n_quality_tags"] == 1]
    if len(one_tag) >= 3:
        show_rows(gdf, one_tag.sample(3, random_state=1).index.tolist(), "3 random rows with exactly 1 quality tag")

    # 3 with zero quality tags
    zero_tag = gdf[gdf["_n_quality_tags"] == 0]
    if len(zero_tag) >= 3:
        show_rows(gdf, zero_tag.head(3).index.tolist(), "3 rows with NO quality tags (first 3)")

    # --- 4. Geometry quality issues ---------------------------------------
    section("4. Geometry quality issues")

    # Empty / null geometries
    is_empty = gdf.geometry.is_empty
    is_null = gdf.geometry.isna()
    print(f"  empty geometries:  {int(is_empty.sum()):,}")
    print(f"  null geometries:   {int(is_null.sum()):,}")

    # Zero-length linestrings
    def is_zero_length(geom):
        if geom is None or geom.is_empty:
            return False
        if isinstance(geom, LineString):
            return geom.length == 0
        if isinstance(geom, MultiLineString):
            return all(part.length == 0 for part in geom.geoms)
        return False

    zero_len_mask = gdf.geometry.apply(is_zero_length)
    print(f"  zero-length LineStrings:  {int(zero_len_mask.sum()):,}")

    # Invalid geometries
    is_valid = gdf.geometry.is_valid
    print(f"  invalid geometries:  {int((~is_valid).sum()):,}")
    if (~is_valid).any():
        # Show 3 invalid examples
        for idx in gdf[~is_valid].head(3).index:
            print(f"    [{idx}] reason: {gdf.geometry.loc[idx].is_valid_reason}")

    # Duplicate geometries (by WKB). Use bytes for a robust comparison.
    wkb = gdf.geometry.to_wkb()
    n_dup = int(wkb.duplicated().sum())
    print(f"  duplicate geometries (by WKB):  {n_dup:,}")
    if n_dup > 0:
        # Show 3 duplicate examples
        dup_wkb = wkb[wkb.duplicated(keep=False)]
        for v in dup_wkb.value_counts().head(3).index:
            matching = gdf[wkb == v]
            print(f"    example appears {len(matching)} times, first 3 indices: {matching.index[:3].tolist()}")
            row0 = matching.iloc[0]
            bits = ", ".join(f"{c}={row0[c]!r}" for c in TAGS if c in gdf.columns)
            print(f"      first: {bits}")

    # Self-intersections: only computed for LineString (cheap-ish).
    # MultiLineString and Point are skipped (Point can't self-intersect; MLS would need per-part).
    lines = gdf[gdf.geometry.geom_type == "LineString"]
    if len(lines) > 0:
        # is_simple returns False if self-intersecting or duplicate consecutive points
        is_simple_mask = lines.geometry.is_simple
        n_non_simple = int((~is_simple_mask).sum())
        print(f"  non-simple LineStrings (self-intersect or duplicate verts):  {n_non_simple:,} of {len(lines):,}")
    else:
        print("  no LineString geometries to check for self-intersections")

    # Heuristic: "orphaned" — LineString with length 0 OR length < 1 metre at EPSG:4326 (essentially degenerate).
    # Better metric: project to UTM 43N and count sub-1m segments. We do that in a separate run.

    print()
    print("=" * 78)
    print("Inspection complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
