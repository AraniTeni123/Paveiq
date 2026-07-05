"""Shared configuration constants for the PaveIQ pipeline.

Centralizes paths, target CRS, score range, and other
project-wide defaults so individual stages can stay focused
on logic.
"""

from pathlib import Path

# --- Paths ------------------------------------------------------------------

# Project root is the parent of the ``src/`` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Geospatial defaults ----------------------------------------------------

# Bengaluru and surrounding area; WGS84 / UTM zone 43N is convenient
# for metric-area / distance calculations.
TARGET_CRS = "EPSG:32643"

# Default score range produced by the model.
SCORE_MIN = 0
SCORE_MAX = 100
