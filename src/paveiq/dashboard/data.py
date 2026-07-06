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

from paveiq.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from paveiq.data_ingestion.ward_boundaries import load_wards

POOR_SCORE_THRESHOLD = 40

# Bundled fixed snapshot for the one Bengaluru neighborhood (Koramangala)
# committed to the repo — see README "Demo dataset". data/processed/ is
# gitignored, so a fresh deploy (e.g. Streamlit Cloud) has nothing there;
# this is the fallback that gives the deployed app something to show.
DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo"

# Same idea for the scorer artifact: artifacts/ is gitignored too, so a
# fresh deploy has no trained/heuristic scorer for the what-if panel to
# call. artifacts/demo/ bundles the artifact matching the demo dataset.
DEMO_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "demo"


def _find_latest_scored(processed_dir: Optional[Path] = None) -> Path:
    """Return the newest ``*_scored.parquet``.

    Looks in ``processed_dir`` (default: ``data/processed/``) first —
    that's the real, freshly-scored output of a local pipeline run.
    Only when ``processed_dir`` was *not* explicitly overridden and
    nothing is found there does this fall back to the bundled demo
    snapshot in ``data/demo/``, so local dev running the full pipeline
    always sees its own output, while a deploy with an empty/gitignored
    ``data/processed/`` still has something to display.
    """
    using_default = processed_dir is None
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DATA_DIR
    candidates = sorted(processed_dir.glob("*_scored.parquet")) if processed_dir.exists() else []
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    if using_default and DEMO_DATA_DIR.exists():
        demo_candidates = sorted(DEMO_DATA_DIR.glob("*_scored.parquet"))
        if demo_candidates:
            return max(demo_candidates, key=lambda p: p.stat().st_mtime)

    searched = f"{processed_dir}" + (f" or {DEMO_DATA_DIR}" if using_default else "")
    raise FileNotFoundError(f"no `*_scored.parquet` files in {searched}; run scoring.score_city first")


def _find_latest_artifact(artifacts_dir: Optional[Path] = None) -> Path:
    """Return the newest ``*_scorer*.json``.

    Same fallback rule as ``_find_latest_scored``: looks in
    ``artifacts_dir`` (default: ``artifacts/``) first, and only falls
    back to the bundled demo artifact in ``artifacts/demo/`` when
    ``artifacts_dir`` was *not* explicitly overridden and nothing is
    found there.
    """
    using_default = artifacts_dir is None
    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
    candidates = sorted(artifacts_dir.glob("*_scorer*.json")) if artifacts_dir.exists() else []
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    if using_default and DEMO_ARTIFACTS_DIR.exists():
        demo_candidates = sorted(DEMO_ARTIFACTS_DIR.glob("*_scorer*.json"))
        if demo_candidates:
            return max(demo_candidates, key=lambda p: p.stat().st_mtime)

    searched = f"{artifacts_dir}" + (f" or {DEMO_ARTIFACTS_DIR}" if using_default else "")
    raise FileNotFoundError(f"no scorer artifact in {searched}; run `python -m paveiq.models.train` first")


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
