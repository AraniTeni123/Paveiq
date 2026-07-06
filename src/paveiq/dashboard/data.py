"""Loading and aggregation for the dashboard. No ``streamlit`` import here —
this module is plain geopandas/pandas so it's testable without a browser.

``app.py`` wraps the loaders below in ``st.cache_data``; keeping the
caching decorator out of this module means it stays a normal,
directly-callable library for tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from paveiq.config import PROCESSED_DATA_DIR
from paveiq.data_ingestion.ward_boundaries import load_wards

POOR_SCORE_THRESHOLD = 40


def _find_latest_scored(processed_dir: Optional[Path] = None) -> Path:
    """Return the newest ``*_scored.parquet`` in ``processed_dir``."""
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed dir does not exist: {processed_dir}")
    candidates = sorted(processed_dir.glob("*_scored.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"no `*_scored.parquet` files in {processed_dir}; run scoring.score_city first"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_scored_segments(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load the latest (or given) scored segments Parquet."""
    path = Path(path) if path is not None else _find_latest_scored()
    return gpd.read_parquet(path)


def load_ward_polygons() -> gpd.GeoDataFrame:
    """Load the cached BBMP ward polygons (reuses ``ward_boundaries.load_wards``)."""
    return load_wards()


def to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to EPSG:4326 for pydeck/deck.gl, which expect lon/lat."""
    if gdf.crs is None:
        raise ValueError("gdf has no CRS; cannot reproject to EPSG:4326")
    if gdf.crs.to_epsg() == 4326:
        return gdf
    return gdf.to_crs("EPSG:4326")


def ward_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Per-ward aggregate: mean score, segment count, % poor, total length.

    Sorted ascending by ``mean_score`` (worst ward first), matching the
    README's "worst-off segments surfaced first" advocacy goal.
    """
    poor = (scored_df["score"] < POOR_SCORE_THRESHOLD).astype(float)
    grouped = scored_df.assign(_poor=poor).groupby(["ward_id", "ward_name"], dropna=False)
    summary = grouped.agg(
        mean_score=("score", "mean"),
        segment_count=("score", "size"),
        pct_poor=("_poor", lambda s: 100.0 * s.mean()),
        total_length_m=("length_m", "sum"),
    ).reset_index()
    return summary.sort_values("mean_score", ascending=True).reset_index(drop=True)


def leaderboard(scored_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """The ``n`` worst-scoring individual segments, worst first."""
    cols = [c for c in ("osmid", "highway", "ward_name", "score", "length_m") if c in scored_df.columns]
    return scored_df.sort_values("score", ascending=True).head(n)[cols].reset_index(drop=True)
